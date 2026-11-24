# -*- coding: utf-8 -*-
"""
main.py'deki IlanLSTM + modeli_egit döngüsünün NumPy ile birebir yeniden yazımı.
Bu deney ortamında PyTorch kurulamadığı için rapor sayıları bununla üretilmiştir.
Mimari, başlatma, kayıp, optimizer, gradyan kırpma, erken durdurma ve veri hattı (Sozluk,
veriyi_bol, metrikler, grafikler) main.py ile aynıdır; veri hattı doğrudan main.py'den çağrılır.
"""
import sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), ".."))
import main as M  # noqa: E402

DT = np.float64


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyIlanLSTM:
    def __init__(self, V, D=M.GOMME_BOYUTU, H=M.GIZLI_BOYUT, n_is=len(M.IS_TIPLERI), n_alan=len(M.ALANLAR), rng=None):
        rng = rng or np.random.default_rng(M.SEED)
        k = 1 / np.sqrt(H)
        self.p = {"E": rng.standard_normal((V, D)).astype(DT)}
        self.p["E"][0] = 0.0                                    # padding_idx=0
        for d in ("f", "b"):                                    # ileri / geri yön
            self.p["W" + d] = rng.uniform(-k, k, (D + H, 4 * H)).astype(DT)
            self.p["b" + d] = rng.uniform(-k, k, 4 * H).astype(DT) + rng.uniform(-k, k, 4 * H)  # b_ih + b_hh
        kk = 1 / np.sqrt(2 * H)
        self.p["Wi"] = rng.uniform(-kk, kk, (2 * H, n_is)); self.p["bi"] = rng.uniform(-kk, kk, n_is)
        self.p["Wa"] = rng.uniform(-kk, kk, (2 * H, n_alan)); self.p["ba"] = rng.uniform(-kk, kk, n_alan)
        self.H, self.D = H, D

    # ----- tek yön LSTM (maskeli = packed sequence ile eşdeğer) -----
    def _yon(self, x, m, W, b, ters):
        B, T, _ = x.shape
        H = self.H
        h = np.zeros((B, H), DT); c = np.zeros((B, H), DT)
        out = np.zeros((B, T, H), DT)
        cache = [None] * T
        sira = range(T - 1, -1, -1) if ters else range(T)
        for t in sira:
            xh = np.concatenate([x[:, t], h], 1)
            z = xh @ W + b
            i, f, g, o = sig(z[:, :H]), sig(z[:, H:2 * H]), np.tanh(z[:, 2 * H:3 * H]), sig(z[:, 3 * H:])
            cn = f * c + i * g
            tc = np.tanh(cn)
            hn = o * tc
            mt = m[:, t:t + 1]
            cache[t] = (xh, i, f, g, o, c, tc, mt)
            c = mt * cn + (1 - mt) * c
            h = mt * hn + (1 - mt) * h
            out[:, t] = mt * hn
        return out, cache

    def _yon_geri(self, dout, cache, W, ters):
        B, T, H = dout.shape
        dW = np.zeros_like(W); db = np.zeros(W.shape[1], DT)
        dx = np.zeros((B, T, self.D), DT)
        gh = np.zeros((B, H), DT); gc = np.zeros((B, H), DT)
        sira = range(T) if ters else range(T - 1, -1, -1)       # işlem sırasının tersi
        for t in sira:
            xh, i, f, g, o, cp, tc, mt = cache[t]
            dhn = mt * (gh + dout[:, t])
            dcn = mt * gc + dhn * o * (1 - tc ** 2)
            do = dhn * tc; di = dcn * g; dg = dcn * i; df = dcn * cp
            dz = np.concatenate([di * i * (1 - i), df * f * (1 - f), dg * (1 - g ** 2), do * o * (1 - o)], 1)
            dW += xh.T @ dz; db += dz.sum(0)
            dxh = dz @ W.T
            dx[:, t] = dxh[:, :self.D]
            gh = (1 - mt) * gh + dxh[:, self.D:]
            gc = (1 - mt) * gc + dcn * f
        return dW, db, dx

    def ileri(self, X, egitim=False, rng=None, p_drop=M.DROPOUT):
        m = (X != 0).astype(DT)
        x = self.p["E"][X]
        of, cf = self._yon(x, m, self.p["Wf"], self.p["bf"], False)
        ob, cb = self._yon(x, m, self.p["Wb"], self.p["bb"], True)
        out = np.concatenate([of, ob], 2)
        L = np.maximum(m.sum(1, keepdims=True), 1)
        oz = out.sum(1) / L
        dm = (rng.random(oz.shape) >= p_drop) / (1 - p_drop) if egitim else np.ones_like(oz)
        ozd = oz * dm
        li = ozd @ self.p["Wi"] + self.p["bi"]
        la = ozd @ self.p["Wa"] + self.p["ba"]
        self._c = (X, m, cf, cb, L, dm, ozd)
        return li, la

    def geri(self, dli, dla):
        X, m, cf, cb, L, dm, ozd = self._c
        g = {"Wi": ozd.T @ dli, "bi": dli.sum(0), "Wa": ozd.T @ dla, "ba": dla.sum(0)}
        doz = (dli @ self.p["Wi"].T + dla @ self.p["Wa"].T) * dm
        dout = (doz / L)[:, None, :] * m[:, :, None]
        H = self.H
        g["Wf"], g["bf"], dxf = self._yon_geri(dout[:, :, :H], cf, self.p["Wf"], False)
        g["Wb"], g["bb"], dxb = self._yon_geri(dout[:, :, H:], cb, self.p["Wb"], True)
        dE = np.zeros_like(self.p["E"])
        np.add.at(dE, X, dxf + dxb)
        dE[0] = 0.0
        g["E"] = dE
        return g


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def agirlikli_ce(logit, y, w):
    """torch.nn.CrossEntropyLoss(weight=w, ignore_index=-100) ile aynı (ağırlıklı ortalama)."""
    mask = y != M.YOKSAY
    if not mask.any():
        return 0.0, np.zeros_like(logit)
    p = softmax(logit)
    yy = np.where(mask, y, 0)
    wi = w[yy] * mask
    S = wi.sum()
    kayip = -(wi * np.log(p[np.arange(len(y)), yy] + 1e-12)).sum() / S
    d = p.copy(); d[np.arange(len(y)), yy] -= 1
    d *= (wi / S)[:, None]
    return kayip, d


def sinif_agirliklari(etiketler, K):
    etiketler = [e for e in etiketler if e != M.YOKSAY]
    say = np.bincount(etiketler, minlength=K).astype(float)
    return np.where(say > 0, np.sqrt(len(etiketler) / (K * np.maximum(say, 1))), 0.0)


class Adam:
    def __init__(self, p, lr, wd=1e-5, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.wd, self.b1, self.b2, self.eps = lr, wd, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in p.items()}
        self.v = {k: np.zeros_like(v) for k, v in p.items()}
        self.t = 0

    def adim(self, p, g):
        self.t += 1
        for k in p:
            gk = g[k] + self.wd * p[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gk
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * gk ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t); vh = self.v[k] / (1 - self.b2 ** self.t)
            p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def kodla(df, s):
    x = [s.kodla(t) for t in df["ilan_metni"]]
    yi = np.array([M.IS_IDX[e] for e in df["etiket_is_tipi"]])
    ya = np.array([M.ALAN_IDX.get(e, M.YOKSAY) for e in df["etiket_alan"]])
    return x, yi, ya


def doldur(xs):
    T = max(len(x) for x in xs)
    X = np.zeros((len(xs), T), dtype=np.int64)
    for i, x in enumerate(xs):
        X[i, :len(x)] = x
    return X


def tahmin(model, x, bs=128):
    li_all, la_all = [], []
    for s in range(0, len(x), bs):
        li, la = model.ileri(doldur(x[s:s + bs]))
        li_all.append(li); la_all.append(la)
    return np.vstack(li_all), np.vstack(la_all)


VERBOSE = True


def egit(df_e, df_d, s, epok=M.MAKS_EPOK, sabir=M.SABIR, lr=M.OGRENME_HIZI):
    rng = np.random.default_rng(M.SEED)
    model = NumpyIlanLSTM(len(s), rng=rng)
    xe, yie, yae = kodla(df_e, s); xd, yid, yad = kodla(df_d, s)
    wi, wa = sinif_agirliklari(yie, len(M.IS_TIPLERI)), sinif_agirliklari(yae, len(M.ALANLAR))
    opt = Adam(model.p, lr)
    gecmis, en_iyi, en_iyi_p, bek = [], -1, None, 0
    for ep in range(1, epok + 1):
        sira = rng.permutation(len(xe)); top, n = 0.0, 0
        for b in range(0, len(sira), M.BATCH):
            idx = sira[b:b + M.BATCH]
            X = doldur([xe[i] for i in idx])
            li, la = model.ileri(X, egitim=True, rng=rng)
            k1, d1 = agirlikli_ce(li, yie[idx], wi)
            k2, d2 = agirlikli_ce(la, yae[idx], wa)
            g = model.geri(d1, d2)
            norm = np.sqrt(sum((v ** 2).sum() for v in g.values()))
            if norm > 1.0:
                g = {k: v / (norm + 1e-6) for k, v in g.items()}
            opt.adim(model.p, g)
            top += k1 + k2; n += 1
        li, la = tahmin(model, xd)
        dk = agirlikli_ce(li, yid, wi)[0] + agirlikli_ce(la, yad, wa)[0]
        sk = M._skorlar(yid, li.argmax(1), yad, la.argmax(1))
        kayit = {"epok": ep, "egitim_kaybi": top / n, "dogrulama_kaybi": dk, **sk}
        gecmis.append(kayit)
        if VERBOSE: print(f"  epok {ep:02d} | eğitim {kayit['egitim_kaybi']:.3f} | doğr {dk:.3f} | F1 iş {sk['f1_is']:.3f} | F1 alan {sk['f1_alan']:.3f}", flush=True)
        if sk["skor"] > en_iyi + 1e-4:
            en_iyi, bek = sk["skor"], 0
            en_iyi_p = {k: v.copy() for k, v in model.p.items()}
        else:
            bek += 1
            if bek >= sabir:
                break
    model.p = en_iyi_p
    return model, gecmis


def gradyan_kontrol():
    rng = np.random.default_rng(0)
    m = NumpyIlanLSTM(12, D=5, H=4, rng=rng)
    X = np.array([[3, 4, 5, 0], [6, 7, 0, 0], [2, 9, 10, 11]])
    yi = np.array([0, 2, 1]); ya = np.array([1, M.YOKSAY, 3])
    wi = np.array([1., 2., .5, 1., 1.]); wa = np.ones(6)

    def kayip():
        li, la = m.ileri(X)
        return agirlikli_ce(li, yi, wi)[0] + agirlikli_ce(la, ya, wa)[0]
    li, la = m.ileri(X)
    g = m.geri(agirlikli_ce(li, yi, wi)[1], agirlikli_ce(la, ya, wa)[1])
    eps, maks = 1e-6, 0
    for k in m.p:
        for _ in range(6):
            idx = tuple(rng.integers(0, s) for s in m.p[k].shape)
            if k == "E" and idx[0] == 0:
                continue
            o = m.p[k][idx]
            m.p[k][idx] = o + eps; a = kayip()
            m.p[k][idx] = o - eps; b = kayip()
            m.p[k][idx] = o
            num = (a - b) / (2 * eps)
            maks = max(maks, abs(num - g[k][idx]) / max(1e-8, abs(num) + abs(g[k][idx])))
    return maks


if __name__ == "__main__":
    print("Gradyan kontrolü (göreli hata, maks):", gradyan_kontrol())
    df = pd.read_csv(M.EGITIM_CSV, keep_default_na=False)
    # --- main.capraz_dogrulama ile aynı döngü ---
    from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
    cv = []
    for kat, (ed_idx, t_idx) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=M.SEED)
                                          .split(df, df["etiket_is_tipi"], df["grup"]), start=1):
        ed, tt = df.iloc[ed_idx].reset_index(drop=True), df.iloc[t_idx].reset_index(drop=True)
        e_i, d_i = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=M.SEED).split(ed, groups=ed["grup"]))
        sk_ = M.Sozluk.kur(ed.iloc[e_i]["ilan_metni"])
        mdl, _ = egit(ed.iloc[e_i], ed.iloc[d_i], sk_)
        x_, yi_, ya_ = kodla(tt, sk_)
        l1, l2 = tahmin(mdl, x_)
        r = M._skorlar(yi_, l1.argmax(1), ya_, l2.argmax(1)); r["kat"] = kat; r["n_test"] = len(tt)
        cv.append(r); print("KAT", kat, r, flush=True)
    df_e, df_d, df_t = M.veriyi_bol(df)
    s = M.Sozluk.kur(df_e["ilan_metni"])
    print("Bölme:", len(df_e), len(df_d), len(df_t), "| sözlük:", len(s))
    t0 = time.time()
    model, gecmis = egit(df_e, df_d, s)
    print("Eğitim süresi (s):", round(time.time() - t0, 1))
    xt, yit, yat = kodla(df_t, s)
    li, la = tahmin(model, xt)
    pi, pa = li.argmax(1), la.argmax(1)
    print("TEST:", M._skorlar(yit, pi, yat, pa))
    sonuc = M.rapor_ciktilari_uret(df, df_e, df_d, df_t, gecmis, yit, pi, yat, pa)
    sonuc["capraz_dogrulama"] = cv
    with open(M.os.path.join(M.RAPOR_KLASORU, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2, default=float)
    # hata analizi için test tahminlerini kaydet
    out = df_t.copy()
    out["tahmin_is"] = [M.IS_TIPLERI[i] for i in pi]
    out["guven_is"] = softmax(li).max(1)
    out["tahmin_alan"] = [M.ALANLAR[i] for i in pa]
    out["guven_alan"] = softmax(la).max(1)
    out.to_csv(M.os.path.join(M.RAPOR_KLASORU, "test_predictions.csv"), index=False)
    np.savez(M.os.path.join(M.RAPOR_KLASORU, "numpy_model.npz"), **model.p)
    with open(M.os.path.join(M.RAPOR_KLASORU, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(s.k2i, f, ensure_ascii=False)
    print(json.dumps(sonuc["temel_model"], indent=1))

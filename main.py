# -*- coding: utf-8 -*-
"""
==============================================================================
 İLAN.GOV.TR İÇİN DOĞAL DİL DESTEKLİ VE OTONOM YAPAY ZEKA ASİSTANI
 (Natural-Language Driven Autonomous AI Assistant for ilan.gov.tr)
==============================================================================

Sistem üç ayaktan oluşur:

  1) OTONOM KAZIYICI (Selenium + SQLite)
     Arka planda (threading) 10 dakikada bir ilan.gov.tr "Kamu-Akademik Personel"
     kategorisini tarar, yeni ilanların detay sayfalarına girer, kadro tablolarını
     satır satır ayrıştırır ve CANLI veritabanına (live_ads.db) yazar.

  2) PYTORCH LSTM BEYNİ (çoklu görev / multi-task)
     training_data.csv (eski ilanlardan üretilmiş, canlı veriden bağımsız) ile eğitilir.
     Tek bir gövde (Embedding + çift yönlü LSTM) ve iki çıkış başı vardır:
        - İş Tipi : argor / ogrgor / ogruye / idari / ilan_disi
        - Alan    : muhendislik / fen_bilimleri / saglik / sosyal_bilimler /
                    egitim / sanat_mimarlik
     Model yeni linklerden de öğrenir: canlı ilanlardan güvenilir etiketli kadrolar
     AYRI bir öğrenme havuzuna (learning_pool.db) kopyalanır ve model periyodik
     olarak ince ayar (fine-tune) ile güncellenir. Yeni model, sabit test kümesinde
     eskisinden kötü değilse devreye alınır (champion / challenger).

  3) DOĞAL DİL ARAYÜZÜ (LLM API - Google Gemini, yedek: Groq; ikisi de ücretsiz katman)
     Kullanıcı cümlesini JSON filtreye çevirir, sonuçları 5'erli gruplar halinde,
     tıklanabilir linkleriyle ve sıcak bir asistan diliyle sunar. API anahtarı yoksa
     veya bağlantı koparsa kural tabanlı yedek çözümleyici ve şablon cevaplar devreye
     girer; kullanıcı hiçbir zaman Python hata mesajı görmez.

Kesin kurallar (koda gömülüdür):
  * Katı alt filtre: LLM'in çıkardığı "spesifik_kelime", kadro metninde metin bazlı
    (string contains) aranır. "bilgisayar" arayana "inşaat" kadrosu gösterilmez.
  * Veri ayrımı: training_data.csv + learning_pool.db  <->  live_ads.db
    birbirine asla karışmaz. Kullanıcıya yalnızca canlı DB'deki linkler verilir.
  * 7 gün kuralı: yayın tarihi 7 günden eski ilan kimseye gösterilmez; bakım
    iş parçacığı bunları canlı DB'den silip öğrenme havuzuna arşivler.
  * Sıfır print hatası: tüm hatalar assistant.log dosyasına yazılır, ekrana asla.
  * Maksimum 5'erli gösterim, "devam" ile sonraki 5.

Kullanım:
  pip install torch selenium beautifulsoup4 pandas numpy scikit-learn matplotlib google-genai
  python main.py --anahtar          # ücretsiz Gemini ve/veya Groq anahtarını gir (ilk açılışta da sorulur)
  (alternatif: GEMINI_API_KEY / GROQ_API_KEY ortam değişkenleri)

  python main.py --veri-hazirla     # academic_ads_clean.json + legacy_scrape.db -> training_data.csv
  python main.py --egit             # modeli eğitir, metrik + grafikleri report_outputs/ altına yazar
  python main.py --tek-tarama       # tek bir kazıma turu (test için)
  python main.py                    # tam sistem: kazıyıcı + öğrenme + sohbet
==============================================================================
"""

# ==============================================================================
# BÖLÜM 0: İMPORTLAR, AYARLAR VE SESSİZ GÜNLÜKLEME
# ==============================================================================
import os
import re
import sys
import json
import time
import random
import sqlite3
import logging
import argparse
import threading
import warnings
import unicodedata
from datetime import datetime, timedelta, date
from collections import Counter

import numpy as np
import pandas as pd

try:
    # --- Derin öğrenme (PyTorch) ---
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    # --- Web kazıma (Selenium + BeautifulSoup) ---
    from bs4 import BeautifulSoup
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
except ImportError as _eksik:
    # Kullanıcıya Python hata kodu yerine anlaşılır bir yönlendirme gösteriyoruz.
    print(f"Merhaba! Asistanın çalışması için bir kütüphane eksik görünüyor ({_eksik.name}).\n"
          "Lütfen şu komutu çalıştırıp tekrar dene:\n"
          "  pip install torch selenium beautifulsoup4 pandas numpy scikit-learn matplotlib google-genai")
    sys.exit(1)

# Uyarıların (FutureWarning vb.) kullanıcı ekranına düşmesini engelliyoruz.
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

# ---- Dosya yolları ----
KLASOR = os.path.dirname(os.path.abspath(__file__))
EGITIM_JSON = os.path.join(KLASOR, "data", "academic_ads_clean.json")   # eski akademik ilanlar (tablolu)
ESKI_DB = os.path.join(KLASOR, "data", "legacy_scrape.db")                # eski kazıma (ihale ağırlıklı)
EGITIM_CSV = os.path.join(KLASOR, "data", "training_data.csv")              # MODELİN EĞİTİM VERİSİ
CANLI_DB = os.path.join(KLASOR, "live_ads.db")                # KULLANICIYA GÖSTERİLEN CANLI VERİ
HAVUZ_DB = os.path.join(KLASOR, "learning_pool.db")                # yeni linklerden öğrenme havuzu
MODEL_KLASORU = os.path.join(KLASOR, "model")
RAPOR_KLASORU = os.path.join(KLASOR, "report_outputs")
LOG_DOSYASI = os.path.join(KLASOR, "assistant.log")

# ---- Zamanlama ve iş kuralları ----
KAZIMA_ARALIGI_SN = 10 * 60          # 10 dakika
OGRENME_ARALIGI_SN = 24 * 60 * 60    # günde bir ince ayar denemesi
OGRENME_MIN_YENI_ORNEK = 50          # en az bu kadar yeni etiketli örnek birikince
GOSTERIM_GUN_SINIRI = 7              # 7 günden eski ilan gösterilmez
SAYFA_BOYUTU = 5                     # 5'erli gösterim
KAZIMA_SAYFA_SAYISI = 3              # her turda listelemenin ilk 3 sayfası
TUR_BASINA_MAKS_DETAY = 40           # siteye yük bindirmemek için tur başına detay sınırı

LISTE_URL = "https://www.ilan.gov.tr/ilan/kategori/8/kamu-akademik-personel?ats=5&currentPage={sayfa}"
SITE = "https://www.ilan.gov.tr"

# ---- LLM ayarları (iki ÜCRETSİZ sağlayıcı: önce Gemini, kota/bağlantı sorununda Groq) ----
# Anahtarlar koda GÖMÜLMEZ. Sıra: ortam değişkeni -> api_keys.json (ilk açılışta sihirbazla
# oluşturulur, .gitignore'dadır). Her kullanıcı kendi ücretsiz anahtarını kullanır; kimseden ücret alınmaz.
#   Gemini : https://aistudio.google.com  -> "Get API key"  (faturalandırma AÇMAYIN = ücretsiz katman)
#   Groq   : https://console.groq.com/keys                 (kart bilgisi istemez)
ANAHTAR_DOSYASI = os.path.join(KLASOR, "api_keys.json")
GEMINI_MODELLERI = [m for m in [os.environ.get("GEMINI_MODEL", "").strip(),
                                "gemini-3.5-flash", "gemini-2.5-flash"] if m]
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELLERI = [m for m in [os.environ.get("GROQ_MODEL", "").strip(),
                              "openai/gpt-oss-120b", "openai/gpt-oss-20b"] if m]   # llama-3.3 16.08.2026'da kapatıldı

# ---- Sessiz günlükleme: her şey dosyaya, ekrana hiçbir hata basılmaz ----
logging.basicConfig(filename=LOG_DOSYASI, level=logging.INFO, encoding="utf-8",
                    format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s")
LOG = logging.getLogger("asistan")
for _gurultulu in ("selenium", "urllib3", "httpx", "google", "WDM"):
    logging.getLogger(_gurultulu).setLevel(logging.WARNING)

# ---- Sınıf tanımları (iki görev) ----
IS_TIPLERI = ["argor", "ogrgor", "ogruye", "idari", "ilan_disi"]
ALANLAR = ["muhendislik", "fen_bilimleri", "saglik", "sosyal_bilimler", "egitim", "sanat_mimarlik"]
IS_IDX = {k: i for i, k in enumerate(IS_TIPLERI)}
ALAN_IDX = {k: i for i, k in enumerate(ALANLAR)}
YOKSAY = -100                         # bu etiketin kaybı hesaplanmaz (ignore_index)

IS_TIPI_ACIKLAMA = {"argor": "Araştırma Görevlisi", "ogrgor": "Öğretim Görevlisi",
                    "ogruye": "Öğretim Üyesi (Prof./Doç./Dr. Öğr. Üyesi)",
                    "idari": "İdari / Kamu Personeli", "ilan_disi": "Personel ilanı değil"}
ALAN_ACIKLAMA = {"muhendislik": "Mühendislik & Teknoloji", "fen_bilimleri": "Fen & Doğa Bilimleri",
                 "saglik": "Sağlık Bilimleri", "sosyal_bilimler": "Sosyal & Beşeri Bilimler",
                 "egitim": "Eğitim & Spor", "sanat_mimarlik": "Sanat, Tasarım & Mimarlık"}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ==============================================================================
# BÖLÜM 1: TÜRKÇE METİN ARAÇLARI
# ==============================================================================
def tr_kucuk(metin):
    """Türkçe'ye uygun küçük harf: 'I'->'ı', 'İ'->'i' (Python'un lower()'ı İ'yi bozar)."""
    metin = str(metin or "").replace("I", "ı").replace("İ", "i")
    return unicodedata.normalize("NFC", metin.lower())


_KATLAMA = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def katla(metin):
    """Arama için Türkçe karakterleri ASCII'ye katlar: 'Mühendisliği' -> 'muhendisligi'.
    Kullanıcı 'muhendislik' yazsa da 'mühendislik' yazsa da aynı sonucu buluruz."""
    metin = tr_kucuk(metin).translate(_KATLAMA)
    metin = "".join(c for c in unicodedata.normalize("NFD", metin) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", metin).strip()


def kelimelere_bol(metin):
    """Model girdisi için tokenizasyon: yalnızca harflerden oluşan, en az 2 harfli kelimeler."""
    return [k for k in re.findall(r"[a-zçğıöşüâîû]+", tr_kucuk(metin)) if len(k) >= 2]


def kok(kelime):
    """Basit Türkçe kök kırpma (çekim eklerini tolere etmek için):
    'mühendisliği' -> 'muhendisl' ; 'hemşire' -> 'hemsi' ; 'bilgisayar' -> 'bilgisa'."""
    k = katla(kelime)
    return k if len(k) <= 5 else k[:max(5, len(k) - 3)]


def tarih_coz(metin):
    """'13.11.2026' veya '2026-11-13' biçimlerini date nesnesine çevirir; olmazsa None."""
    if not metin:
        return None
    m = re.search(r"(\d{1,2})[./](\d{1,2})[./](\d{4})", str(metin))
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(metin))
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


# ==============================================================================
# BÖLÜM 2: ZAYIF DENETİM (WEAK SUPERVISION) - ETİKETLEME KURALLARI VE TABLO AYRIŞTIRMA
# ==============================================================================
# Etiketler, ilan tablolarındaki YAPISAL sütunlardan türetilir:
#   - İş tipi  <- "Ünvan / Kadro Ünvanı" sütunu
#   - Alan     <- "Birim / Fakülte / Bölüm / Anabilim Dalı" sütunları
# Etiket sızıntısını (label leakage) önlemek için ÜNVAN SÜTUNU MODEL GİRDİSİNDEN ÇIKARILIR:
# model, iş tipini kadronun şartlarından/bağlamından ("tezli yüksek lisans yapıyor olmak",
# "doçentlik unvanını almış olmak", "ALES 70" ...) öğrenmek zorunda kalır.

UNVAN_KURALLARI = [  # sıra önemlidir: 'Öğr. Gör. Dr.' öğretim görevlisidir, öğretim üyesi değil
    ("argor", re.compile(r"ara[sş]t[ıi]rma\s*g[öo]revli|ar[sş]\.?\s*g[öo]r|ara[sş]\.\s*g[öo]r")),
    ("ogrgor", re.compile(r"[öo][ğg]retim\s*g[öo]revli|[öo][ğg]r\.?\s*g[öo]r|okutman")),
    ("ogruye", re.compile(r"prof|do[çc]|[öo][ğg]retim\s*[üu]yes|[öo][ğg]r\.?\s*[üu]yes")),
    ("idari", re.compile(r"memur|s[öo]zle[şs]meli|uzman\s*yard|kontrol[öo]r|m[üu]fetti[şs]|teknisyen|"
                         r"tekniker|[şs]of[öo]r|g[üu]venlik\s*g[öo]rev|temizlik|destek\s*personel|"
                         r"koruma|i[şs][çc]i|hem[şs]ire|ebe\b|sa[ğg]l[ıi]k\s*teknik|programc[ıi]|"
                         r"m[üu]hendis\b|avukat|veteriner\s*hekim|aday\s*memur|personel|sekreter|"
                         r"bilgisayar\s*i[şs]letmen|veri\s*haz[ıi]rlama|a[şs][çc][ıi]|bah[çc][ıi]van|"
                         r"elektrik[çc]i|tesisat[çc]|kaloriferci|garson|bekçi|itfaiye|zab[ıi]ta")),
]


def kural_is_tipi(unvan_metni):
    """Ünvan hücresinden iş tipini çıkarır. Eşleşme yoksa None."""
    s = tr_kucuk(unvan_metni)
    for etiket, desen in UNVAN_KURALLARI:
        if desen.search(s):
            return etiket
    return None


ALAN_ANAHTARLARI = {
    "saglik": ["sanatoryum", "suam", "fizik tedavi", "fiziksel tıp", "rehabilitasyon", "hastalıkları", "hastane",
               "tıp", "tıbbi", "sağlık", "hemşire", "diş hek", "ağız", "eczacı", "veteriner", "fizyoterapi",
               "ebelik", "anestezi", "cerrahi", "hastalık", "radyoloji", "farmakoloji", "patoloji", "pediatri",
               "çocuk sağ", "kardiyoloji", "onkoloji", "nöroloji", "psikiyatri", "ortopedi", "beslenme",
               "odyoloji", "acil", "anatomi", "fizyoloji", "histoloji", "mikrobiyoloji", "biyokimya", "üroloji",
               "dermatoloji", "göz hast", "kulak", "nükleer", "ameliyat", "laborant", "diyaliz", "perfüzyon",
               "hematoloji", "endokrin", "gastroenteroloji", "nefroloji", "romatoloji", "immünoloji",
               "aile hek", "halk sağ", "biyoistatistik", "tıp eğitimi", "adli tıp", "spor hekim", "kadın hast",
               "dahili", "eczane", "optisyen", "paramedik", "yaşlı bakım", "evde hasta", "radyoterapi"],
    "muhendislik": ["mühendis", "teknoloji fakültesi", "bilgisayar", "yazılım", "elektrik", "elektronik",
                    "makine", "makina", "inşaat", "endüstri", "otomasyon", "mekatronik", "bilişim", "harita",
                    "metalurji", "enerji", "otomotiv", "teknik bilimler", "mühendislik", "yapay zeka",
                    "veri bilimi", "siber", "havacılık", "uzay", "denizcilik", "gemi", "tekstil", "maden",
                    "petrol", "biyomedikal", "kontrol ve", "telekom", "elektronik haberleşme", "inşaat teknolojisi",
                    "kimya teknolojisi", "malzeme", "nanoteknoloji", "iklimlendirme", "raylı sistem", "mobilya",
                    "dijital dönüşüm", "web tasarım", "bilgi güvenliği"],
    "fen_bilimleri": ["fen fakültesi", "fen-edebiyat", "fen edebiyat", "matematik", "fizik", "kimya",
                      "biyoloji", "istatistik", "moleküler", "genetik", "astronomi", "ziraat", "tarım", "bitki",
                      "su ürünleri", "orman", "zootekni", "toprak", "bahçe", "tarla", "hayvansal", "gıda",
                      "jeoloji", "jeofizik", "çevre", "aktüerya", "biyoteknoloji", "fen bilimleri"],
    "egitim": ["eğitim fakültesi", "öğretmenliği", "eğitim bilimleri", "eğitimi anabilim", "spor bilimleri",
               "beden eğitimi", "rehberlik", "psikolojik danışma", "okul öncesi", "sınıf eğitimi",
               "özel eğitim", "antrenörlük", "rekreasyon", "spor yöneticiliği", "egzersiz", "hareket ve antrenman",
               "eğitim programları", "eğitim yönetimi", "spor", "eğitimi bölümü"],
    "sanat_mimarlik": ["mimarlık", "mimari", "güzel sanatlar", "tasarım", "müzik", "konservatuvar", "resim",
                       "grafik", "sahne", "sinema", "iç mimar", "şehir ve bölge", "moda", "seramik", "heykel",
                       "fotoğraf", "geleneksel türk sanat", "peyzaj", "el sanatları", "tiyatro", "opera",
                       "bale", "görsel sanat", "animasyon", "çalgı", "kompozisyon", "sanat"],
    "sosyal_bilimler": ["iktisat", "işletme", "hukuk", "maliye", "ekonomi", "siyaset", "kamu yönetimi",
                        "uluslararası", "sosyoloji", "psikoloji", "tarih", "felsefe", "edebiyat", "dilbilim",
                        "ilahiyat", "islami", "islam", "din ", "dinler", "tefsir", "hadis", "kelam", "iletişim",
                        "gazetecilik", "halkla ilişkiler", "turizm", "gastronomi", "muhasebe", "lojistik", "büro",
                        "yönetim bilişim", "coğrafya", "arkeoloji", "sosyal hizmet", "yabancı dil", "ingilizce",
                        "mütercim", "çeviri", "dili ve", "bankacılık", "finans", "pazarlama", "insan kaynakları",
                        "sosyal bilimler", "beşeri", "antropoloji", "kütüphane", "bilgi ve belge", "çalışma ekonomisi",
                        "ekonometri", "adalet", "sigortacılık", "dış ticaret", "medya", "radyo", "reklam",
                        "türk dili", "kafkas", "çocuk gelişimi", "yönetim", "otel", "seyahat", "emlak", "işletmecilik"],
}
# Anahtar kelimeleri katlanmış biçimde ön-derliyoruz (aramada katla() kullanılır).
_ALAN_DESENLERI = [(alan, re.sub(r"[^a-z0-9]+", " ", katla(k)).strip())
                   for alan, liste in ALAN_ANAHTARLARI.items() for k in liste]


def kural_alan(*metinler):
    """Verilen metinlerde (öncelik sırasıyla) EN ERKEN geçen alan anahtarına göre alanı seçer.
    Örn. 'Eğitim Fakültesi / Matematik Eğitimi' -> önce 'eğitim fakültesi' geçtiği için 'egitim'.
    Anahtarlar yalnızca KELİME BAŞINDA eşleşir: 'kadın ve' içindeki 'din' dinler sayılmaz.
    Aynı konumda iki eşleşme varsa sözlükte önce gelen (ör. 'sanatoryum' > 'sanat') kazanır."""
    for metin in metinler:
        m = " " + re.sub(r"[^a-z0-9]+", " ", katla(metin)) + " "
        en_iyi, en_erken = None, 10 ** 9
        for alan, anahtar in _ALAN_DESENLERI:
            p = m.find(" " + anahtar)
            if 0 <= p < en_erken:
                en_iyi, en_erken = alan, p
        if en_iyi:
            return en_iyi
    return None


# Tablo başlık hücrelerinin rollerini belirleyen desenler
_BASLIK_UNVAN = re.compile(r"[üu]nvan|pozisyon|kadro\s*ad")
_BASLIK_YAPISAL = re.compile(r"birim|fak[üu]lte|b[öo]l[üu]m|anabilim|ana\s*bilim|abd|program|y[üu]ksekokul|"
                             r"enstit[üu]|g[öo]revlendir|anasanat|ana\s*sanat|myo|meslek")
_BASLIK_ACIKLAMA = re.compile(r"a[çc][ıi]klama|[şs]art|nitelik|ko[şs]ul|kriter")


def _bos_mu(h):
    return not re.search(r"[A-Za-zÇĞİÖŞÜçğıöşü]{2,}", h or "")


def tablolardan_kadrolar(tablolar):
    """İlan tablolarını (liste[liste[liste[str]]]) kadro satırlarına çevirir.
    Her kadro için: model girdisi metni (ünvan hariç), ham ünvan, kural iş tipi, kural alan.
    Hem eğitim verisi hazırlarken hem de canlı kazıyıcıda AYNI fonksiyon kullanılır."""
    kadrolar = []
    for tablo in tablolar or []:
        if not tablo or len(tablo) < 2:
            continue
        # 1) Başlık satırını bul: ilk 3 satırda kısa bir "Ünvan" hücresi olan satır
        bas_i = None
        for i, satir in enumerate(tablo[:3]):
            if any(_BASLIK_UNVAN.search(tr_kucuk(h)) and len(h) < 40 for h in satir):
                bas_i = i
                break
        if bas_i is None:
            continue
        baslik = [tr_kucuk(h) for h in tablo[bas_i]]
        n = len(baslik)
        u_j = next(j for j, h in enumerate(baslik) if _BASLIK_UNVAN.search(h) and len(h) < 40)
        yap_j = [j for j, h in enumerate(baslik) if j != u_j and _BASLIK_YAPISAL.search(h)]
        acik_j = [j for j, h in enumerate(baslik) if j != u_j and j not in yap_j and _BASLIK_ACIKLAMA.search(h)]
        onceki = [""] * n
        for satir in tablo[bas_i + 1:]:
            satir = [str(h or "").strip() for h in satir]
            if len(satir) > n or len(satir) < max(2, n - 3):
                continue
            # 2) Birleştirilmiş (rowspan) hücreler yüzünden kısa kalan satırları sağa hizala,
            #    eksik baştaki hücreleri bir önceki satırdan doldur.
            if len(satir) < n:
                satir = onceki[:n - len(satir)] + satir
            onceki = satir
            unvan = satir[u_j]
            is_tipi = kural_is_tipi(unvan)
            if not is_tipi:
                continue
            yapisal = " ".join(satir[j] for j in yap_j if not _bos_mu(satir[j]))
            aciklama = " ".join(satir[j] for j in acik_j if not _bos_mu(satir[j]))
            metin = re.sub(r"\s+", " ", f"{yapisal} {aciklama}").strip()
            if len(kelimelere_bol(metin)) < 2 or "toplam" == katla(yapisal):
                continue
            kadrolar.append({"metin": metin, "unvan_ham": unvan, "is_tipi_kural": is_tipi,
                             "alan_kural": kural_alan(yapisal, aciklama)})
    return kadrolar


# ==============================================================================
# BÖLÜM 3: EĞİTİM VERİSİNİN HAZIRLANMASI (training_data.csv)
# ==============================================================================
ILAN_BASINA_MAKS = 40


def egitim_verisi_hazirla(json_yolu=EGITIM_JSON, eski_db_yolu=ESKI_DB, cikti=EGITIM_CSV):
    """Eski verilerden etiketli eğitim kümesi üretir. CANLI VERİTABANINA HİÇ DOKUNMAZ.
      * academic_ads_clean.json -> kadro düzeyinde örnekler (argor/ogrgor/ogruye + alan)
      * legacy_scrape.db          -> eski kazıyıcının yanlışlıkla topladığı ihale başlıkları
                                       = 'ilan_disi' (personel ilanı olmayan) negatif sınıfı."""
    satirlar = []
    if os.path.exists(json_yolu):
        with open(json_yolu, encoding="utf-8") as f:
            ilanlar = json.load(f)
        for ilan in ilanlar:
            for k in tablolardan_kadrolar(ilan.get("tablolar")):
                satirlar.append({"grup": f"ilan_{ilan.get('id')}", "kaynak": "akademik_json",
                                 "ilan_metni": k["metin"], "unvan_ham": k["unvan_ham"],
                                 "etiket_is_tipi": k["is_tipi_kural"], "etiket_alan": k["alan_kural"] or "",
                                 "sehir": ilan.get("sehir") or "", "link": ilan.get("link") or ""})
    if os.path.exists(eski_db_yolu):
        con = sqlite3.connect(f"file:{eski_db_yolu}?mode=ro", uri=True)
        for metin, link in con.execute("SELECT ilan_metni, link FROM ilanlar"):
            # Menü/gezinme linkleri ilan değildir, atlıyoruz.
            if not re.search(r"/ilan/\d{6,}", link or ""):
                continue
            temiz = re.sub(r"\s*-\s*[A-Z]{2,4}\d{5,}\s*$", "", metin or "").strip()
            temiz = re.sub(r"[​-‏]", "", temiz)
            if not temiz:
                continue
            satirlar.append({"grup": "eski_" + re.search(r"/ilan/(\d+)", link).group(1), "kaynak": "eski_db",
                             "ilan_metni": temiz, "unvan_ham": "", "etiket_is_tipi": "ilan_disi",
                             "etiket_alan": "", "sehir": "", "link": link})
        con.close()
    df = pd.DataFrame(satirlar).drop_duplicates(subset=["ilan_metni", "etiket_is_tipi"]).reset_index(drop=True)
    # 10'dan az örneği olan iş tipi sınıfı (ör. eski veride 'idari' yalnızca 1 kez geçiyor) güvenilir
    # öğrenilemez ve bölmeyi bozar; bu sınıflar sürekli öğrenme havuzu doldukça modele katılır.
    say = df["etiket_is_tipi"].value_counts()
    df = df[df["etiket_is_tipi"].map(say) >= 10].reset_index(drop=True)
    # Tek bir ilanın veriye hükmetmesini önle: bir ilandan en fazla ILAN_BASINA_MAKS kadro alınır
    # (ör. bir sağlık üniversitesi ilanı tek başına 236 neredeyse aynı kalıpta kadro içeriyor).
    karisik = df.sample(frac=1.0, random_state=SEED)
    df = karisik[karisik.groupby("grup").cumcount() < ILAN_BASINA_MAKS].sort_index().reset_index(drop=True)
    df.to_csv(cikti, index=False, encoding="utf-8")
    LOG.info("Eğitim verisi hazırlandı: %d satır -> %s", len(df), cikti)
    return df


# ==============================================================================
# BÖLÜM 4: PYTORCH ÇOKLU GÖREV LSTM MODELİ
# ==============================================================================
MAKS_UZUNLUK = 80        # kelime
GOMME_BOYUTU = 64
GIZLI_BOYUT = 64
DROPOUT = 0.3
BATCH = 32
OGRENME_HIZI = 2e-3
MAKS_EPOK = 30
SABIR = 5                # erken durdurma sabrı (epok)


class Sozluk:
    """Kelime -> indeks sözlüğü. 0=<PAD>, 1=<UNK>. Yalnızca EĞİTİM kümesinden kurulur
    (doğrulama/test kelimelerini önceden görmek sızıntı olurdu)."""

    def __init__(self, kelime2id=None):
        self.k2i = kelime2id or {"<PAD>": 0, "<UNK>": 1}

    @classmethod
    def kur(cls, metinler, min_frekans=2):
        sayac = Counter(k for m in metinler for k in kelimelere_bol(m))
        s = cls()
        for kelime, frekans in sayac.most_common():
            if frekans >= min_frekans:
                s.k2i[kelime] = len(s.k2i)
        return s

    def genislet(self, metinler, min_frekans=2):
        """Öğrenme havuzundan gelen yeni kelimeleri sözlüğe ekler (yeni linklerden öğrenme)."""
        eski = len(self.k2i)
        sayac = Counter(k for m in metinler for k in kelimelere_bol(m))
        for kelime, frekans in sayac.most_common():
            if frekans >= min_frekans and kelime not in self.k2i:
                self.k2i[kelime] = len(self.k2i)
        return len(self.k2i) - eski

    def kodla(self, metin):
        ids = [self.k2i.get(k, 1) for k in kelimelere_bol(metin)][:MAKS_UZUNLUK]
        return ids or [1]

    def __len__(self):
        return len(self.k2i)


class IlanVeriseti(Dataset):
    def __init__(self, df, sozluk):
        self.x = [sozluk.kodla(m) for m in df["ilan_metni"]]
        self.y_is = [IS_IDX[e] for e in df["etiket_is_tipi"]]
        self.y_alan = [ALAN_IDX.get(e, YOKSAY) if isinstance(e, str) else YOKSAY for e in df["etiket_alan"]]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y_is[i], self.y_alan[i]


def topla(batch):
    """Değişken uzunluklu dizileri sağdan 0 (PAD) ile doldurup tensöre çevirir."""
    xs, yis, yalan = zip(*batch)
    uz = torch.tensor([len(x) for x in xs], dtype=torch.long)
    X = torch.zeros(len(xs), int(uz.max()), dtype=torch.long)
    for i, x in enumerate(xs):
        X[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return X, uz, torch.tensor(yis, dtype=torch.long), torch.tensor(yalan, dtype=torch.long)


class IlanLSTM(nn.Module):
    """Embedding -> Çift yönlü LSTM -> maskeli ortalama havuzlama -> Dropout -> 2 çıkış başı.
    Paylaşılan gövde iki görevin ortak dil temsillerini öğrenir (multi-task learning);
    az örnekli sınıflar (argor, ogrgor) diğer görevden gelen sinyalden yararlanır."""

    def __init__(self, sozluk_boyutu, gomme=GOMME_BOYUTU, gizli=GIZLI_BOYUT, dropout=DROPOUT):
        super().__init__()
        self.gomme = nn.Embedding(sozluk_boyutu, gomme, padding_idx=0)
        self.lstm = nn.LSTM(gomme, gizli, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.bas_is = nn.Linear(2 * gizli, len(IS_TIPLERI))     # softmax CrossEntropy içinde
        self.bas_alan = nn.Linear(2 * gizli, len(ALANLAR))

    def forward(self, X, uzunluklar):
        g = self.gomme(X)                                                     # (B, T, E)
        paket = pack_padded_sequence(g, uzunluklar.cpu(), batch_first=True, enforce_sorted=False)
        cikti, _ = self.lstm(paket)
        cikti, _ = pad_packed_sequence(cikti, batch_first=True, total_length=X.size(1))  # (B, T, 2H)
        maske = (X != 0).unsqueeze(-1).float()
        ozet = (cikti * maske).sum(1) / maske.sum(1).clamp(min=1.0)          # maskeli ortalama
        ozet = self.dropout(ozet)
        return self.bas_is(ozet), self.bas_alan(ozet)

    def sozlugu_buyut(self, yeni_boyut):
        """Sözlük genişleyince gömme matrisini eski ağırlıkları koruyarak büyütür."""
        eski = self.gomme.weight.data
        if yeni_boyut <= eski.size(0):
            return
        yeni = nn.Embedding(yeni_boyut, eski.size(1), padding_idx=0)
        yeni.weight.data[:eski.size(0)] = eski
        self.gomme = yeni


def sinif_agirliklari(etiketler, sinif_sayisi):
    """Dengesiz sınıflar için yumuşatılmış ters frekans ağırlığı: w_c = sqrt(N / (K * n_c))."""
    etiketler = [e for e in etiketler if e != YOKSAY]
    say = np.bincount(etiketler, minlength=sinif_sayisi).astype(float)
    w = np.where(say > 0, np.sqrt(len(etiketler) / (sinif_sayisi * np.maximum(say, 1))), 0.0)
    return torch.tensor(w, dtype=torch.float)


def veriyi_bol(df, seed=SEED):
    """Grup (ilan) bazlı, iş tipine göre tabakalı bölme: %70 eğitim / %15 doğrulama / %15 test.
    Aynı ilanın kadroları ortak açıklama metinleri taşıdığından, bir ilanın tüm satırları
    yalnızca TEK bir kümeye düşer (aksi halde test başarısı yapay olarak şişerdi)."""
    from sklearn.model_selection import StratifiedGroupKFold
    y = df["etiket_is_tipi"].values
    g = df["grup"].values
    dis = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    egitim_dogr_idx, test_idx = next(dis.split(df, y, g))
    ic_df = df.iloc[egitim_dogr_idx]
    ic = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    e_idx, d_idx = next(ic.split(ic_df, ic_df["etiket_is_tipi"].values, ic_df["grup"].values))
    return (ic_df.iloc[e_idx].reset_index(drop=True), ic_df.iloc[d_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True))


def _tahmin_et(model, yukleyici):
    model.eval()
    ti, ta, gi, ga, pi, pa, kayip = [], [], [], [], [], [], []
    with torch.no_grad():
        for X, uz, yi, ya in yukleyici:
            li, la = model(X, uz)
            ti += yi.tolist(); ta += ya.tolist()
            pi += li.argmax(1).tolist(); pa += la.argmax(1).tolist()
            gi += torch.softmax(li, 1).max(1).values.tolist()
            ga += torch.softmax(la, 1).max(1).values.tolist()
    return np.array(ti), np.array(pi), np.array(ta), np.array(pa)


def _skorlar(ti, pi, ta, pa):
    from sklearn.metrics import f1_score, accuracy_score
    m = ta != YOKSAY
    f_is = f1_score(ti, pi, average="macro", labels=sorted(set(ti)), zero_division=0)
    f_alan = f1_score(ta[m], pa[m], average="macro", labels=sorted(set(ta[m])), zero_division=0) if m.any() else 0.0
    return {"acc_is": accuracy_score(ti, pi), "f1_is": f_is,
            "acc_alan": accuracy_score(ta[m], pa[m]) if m.any() else 0.0, "f1_alan": f_alan,
            "skor": (f_is + f_alan) / 2}


def modeli_egit(df_egitim, df_dogr, sozluk=None, baslangic_modeli=None, epok=MAKS_EPOK, sabir=SABIR,
                ogrenme_hizi=OGRENME_HIZI, sessiz=False):
    """Çoklu görev LSTM'i eğitir. Toplam kayıp = CE(iş tipi) + CE(alan, ignore_index=-100).
    Doğrulama kümesindeki (makro-F1_iş + makro-F1_alan)/2 skoruna göre erken durdurma uygular
    ve en iyi epoktaki ağırlıkları geri yükler."""
    torch.manual_seed(SEED)
    sozluk = sozluk or Sozluk.kur(df_egitim["ilan_metni"])
    model = baslangic_modeli or IlanLSTM(len(sozluk))
    model.sozlugu_buyut(len(sozluk))
    e_ds, d_ds = IlanVeriseti(df_egitim, sozluk), IlanVeriseti(df_dogr, sozluk)
    e_yuk = DataLoader(e_ds, batch_size=BATCH, shuffle=True, collate_fn=topla)
    d_yuk = DataLoader(d_ds, batch_size=128, shuffle=False, collate_fn=topla)
    kriter_is = nn.CrossEntropyLoss(weight=sinif_agirliklari(e_ds.y_is, len(IS_TIPLERI)))
    kriter_alan = nn.CrossEntropyLoss(weight=sinif_agirliklari(e_ds.y_alan, len(ALANLAR)), ignore_index=YOKSAY)
    opt = torch.optim.Adam(model.parameters(), lr=ogrenme_hizi, weight_decay=1e-5)

    gecmis, en_iyi, en_iyi_durum, bekleme = [], -1.0, None, 0
    for ep in range(1, epok + 1):
        model.train()
        toplam, adim = 0.0, 0
        for X, uz, yi, ya in e_yuk:
            opt.zero_grad()
            li, la = model(X, uz)
            kayip = kriter_is(li, yi)
            if (ya != YOKSAY).any():
                kayip = kayip + kriter_alan(la, ya)
            kayip.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # patlayan gradyanlara karşı
            opt.step()
            toplam += kayip.item(); adim += 1
        # doğrulama kaybı
        model.eval()
        with torch.no_grad():
            dk, dn = 0.0, 0
            for X, uz, yi, ya in d_yuk:
                li, la = model(X, uz)
                k = kriter_is(li, yi)
                if (ya != YOKSAY).any():
                    k = k + kriter_alan(la, ya)
                dk += k.item(); dn += 1
        sk = _skorlar(*_tahmin_et(model, d_yuk))
        kayit = {"epok": ep, "egitim_kaybi": toplam / max(adim, 1), "dogrulama_kaybi": dk / max(dn, 1), **sk}
        gecmis.append(kayit)
        if not sessiz:
            print(f"  epok {ep:02d} | eğitim kaybı {kayit['egitim_kaybi']:.3f} | doğrulama kaybı "
                  f"{kayit['dogrulama_kaybi']:.3f} | F1(iş) {sk['f1_is']:.3f} | F1(alan) {sk['f1_alan']:.3f}")
        if sk["skor"] > en_iyi + 1e-4:
            en_iyi, bekleme = sk["skor"], 0
            en_iyi_durum = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bekleme += 1
            if bekleme >= sabir:
                break
    if en_iyi_durum is not None:
        model.load_state_dict(en_iyi_durum)
    return model, sozluk, gecmis


def model_kaydet(model, sozluk, meta=None, klasor=MODEL_KLASORU):
    os.makedirs(klasor, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(klasor, "ad_lstm.pt"))
    with open(os.path.join(klasor, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(sozluk.k2i, f, ensure_ascii=False)
    with open(os.path.join(klasor, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta or {}, f, ensure_ascii=False, indent=2)


def model_yukle(klasor=MODEL_KLASORU):
    with open(os.path.join(klasor, "vocab.json"), encoding="utf-8") as f:
        sozluk = Sozluk(json.load(f))
    model = IlanLSTM(len(sozluk))
    model.load_state_dict(torch.load(os.path.join(klasor, "ad_lstm.pt"), map_location="cpu"))
    model.eval()
    meta = {}
    if os.path.exists(os.path.join(klasor, "meta.json")):
        with open(os.path.join(klasor, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
    return model, sozluk, meta


class ModelYoneticisi:
    """Modeli iş parçacıkları arasında güvenle paylaştırır (tahmin sırasında model
    güncellenirse kilit sayesinde çakışma olmaz)."""

    def __init__(self, model, sozluk, meta):
        self.model, self.sozluk, self.meta = model, sozluk, meta
        self.kilit = threading.Lock()

    def tahmin(self, metinler):
        if not metinler:
            return []
        with self.kilit:
            self.model.eval()
            ds = [(self.sozluk.kodla(m), 0, 0) for m in metinler]
            X, uz, _, _ = topla(ds)
            with torch.no_grad():
                li, la = self.model(X, uz)
            pi, pa = torch.softmax(li, 1), torch.softmax(la, 1)
        return [(IS_TIPLERI[int(pi[i].argmax())], float(pi[i].max()),
                 ALANLAR[int(pa[i].argmax())], float(pa[i].max())) for i in range(len(metinler))]

    def degistir(self, model, sozluk, meta):
        with self.kilit:
            self.model, self.sozluk, self.meta = model, sozluk, meta


# ---------------- Değerlendirme, grafikler ve rapor çıktıları ----------------
def rapor_ciktilari_uret(df, df_e, df_d, df_t, gecmis, ti, pi, ta, pa, klasor=RAPOR_KLASORU):
    """Yönergenin istediği tüm görsel ve tabloları üretir: veri keşfi (3+ grafik),
    eğitim/doğrulama eğrileri, karışıklık matrisleri, sınıf bazlı P/R/F1 ve temel model kıyası."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix
    os.makedirs(klasor, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    MAVI, TURUNCU, GRI = "#2a6fb0", "#e08a2c", "#8c8c8c"

    # --- Şekil 1: sınıf dağılımları (iki görev) ---
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    s1 = df["etiket_is_tipi"].value_counts().reindex(IS_TIPLERI).fillna(0)
    ax[0].barh(s1.index[::-1], s1.values[::-1], color=MAVI)
    for i, v in enumerate(s1.values[::-1]):
        ax[0].text(v + 5, i, str(int(v)), va="center", fontsize=8)
    ax[0].set_title("İş tipi dağılımı (tüm veri)")
    s2 = df.loc[df["etiket_alan"].isin(ALANLAR), "etiket_alan"].value_counts().reindex(ALANLAR).fillna(0)
    ax[1].barh(s2.index[::-1], s2.values[::-1], color=TURUNCU)
    for i, v in enumerate(s2.values[::-1]):
        ax[1].text(v + 3, i, str(int(v)), va="center", fontsize=8)
    ax[1].set_title("Alan dağılımı (etiketli kadrolar)")
    fig.tight_layout(); fig.savefig(os.path.join(klasor, "figure1_class_distribution.png")); plt.close(fig)

    # --- Şekil 2: metin uzunluğu dağılımı (kelime) ---
    uz = df["ilan_metni"].map(lambda m: len(kelimelere_bol(m)))
    fig, ax = plt.subplots(figsize=(6, 3))
    for etiket, renk in [("ogruye", MAVI), ("argor", TURUNCU), ("ogrgor", "#4a9b5f"), ("ilan_disi", GRI)]:
        ax.hist(uz[df["etiket_is_tipi"] == etiket].clip(upper=150), bins=30, alpha=0.6, label=etiket, color=renk)
    ax.axvline(MAKS_UZUNLUK, color="black", ls="--", lw=1)
    ax.text(MAKS_UZUNLUK + 2, ax.get_ylim()[1] * 0.85, f"kesme = {MAKS_UZUNLUK}", fontsize=8)
    ax.set_xlabel("kelime sayısı (150'de kırpıldı)"); ax.set_ylabel("örnek sayısı"); ax.legend(frameon=False)
    ax.set_title("Metin uzunluğu dağılımı")
    fig.tight_layout(); fig.savefig(os.path.join(klasor, "figure2_text_length.png")); plt.close(fig)

    # --- Şekil 3: iş tipine göre ayırt edici kelimeler ---
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4))
    genel = Counter(k for m in df["ilan_metni"] for k in set(kelimelere_bol(m)))
    for ax, etiket in zip(axs, ["argor", "ogrgor", "ogruye"]):
        alt_df = df[df["etiket_is_tipi"] == etiket]
        alt = alt_df["ilan_metni"]
        c = Counter(k for m in alt for k in set(kelimelere_bol(m)))
        # kelimenin bu sınıfta kaç FARKLI ilanda geçtiği (tek ilana özgü kalıpları elemek için)
        ilan_say = Counter(k for _, g in alt_df.groupby("grup")
                           for k in set(w for m in g["ilan_metni"] for w in kelimelere_bol(m)))
        # lift = sınıf içi oran / genel oran ; en az 3 farklı ilanda ve 5 kez geçen kelimeler
        lift = {k: (v / len(alt)) / (genel[k] / len(df)) for k, v in c.items()
                if v >= 5 and ilan_say[k] >= 3 and len(k) > 3}
        top = sorted(lift.items(), key=lambda x: (-round(x[1], 2), -c[x[0]]))[:10]
        ax.barh([t[0] for t in top][::-1], [t[1] for t in top][::-1], color=MAVI)
        ax.set_title(f"{etiket}: ayırt edici kelimeler"); ax.set_xlabel("lift")
    fig.tight_layout(); fig.savefig(os.path.join(klasor, "figure3_distinctive_words.png")); plt.close(fig)

    # --- Şekil 4: eğitim eğrileri ---
    g = pd.DataFrame(gecmis)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(g["epok"], g["egitim_kaybi"], color=MAVI, label="eğitim")
    ax[0].plot(g["epok"], g["dogrulama_kaybi"], color=TURUNCU, label="doğrulama")
    ax[0].set_title("Toplam kayıp (CE_iş + CE_alan)"); ax[0].set_xlabel("epok"); ax[0].legend(frameon=False)
    ax[1].plot(g["epok"], g["f1_is"], color=MAVI, label="makro-F1 iş tipi")
    ax[1].plot(g["epok"], g["f1_alan"], color=TURUNCU, label="makro-F1 alan")
    eniyi = int(g["skor"].idxmax())
    ax[1].axvline(g.loc[eniyi, "epok"], color=GRI, ls="--", lw=1)
    ax[1].set_title("Doğrulama başarımı"); ax[1].set_xlabel("epok"); ax[1].set_ylim(0, 1.02)
    ax[1].legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(klasor, "figure4_training_curves.png")); plt.close(fig)

    # --- Şekil 5: karışıklık matrisleri (test) ---
    m = ta != YOKSAY
    is_lab = [i for i in range(len(IS_TIPLERI)) if i in set(ti) | set(pi)]
    alan_lab = [i for i in range(len(ALANLAR)) if i in set(ta[m]) | set(pa[m])]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for a, (yt, yp, lab, adlar, baslik) in zip(ax, [
            (ti, pi, is_lab, IS_TIPLERI, "İş tipi (test)"),
            (ta[m], pa[m], alan_lab, ALANLAR, "Alan (test)")]):
        cm = confusion_matrix(yt, yp, labels=lab)
        a.imshow(cm, cmap="Blues")
        for i in range(len(lab)):
            for j in range(len(lab)):
                a.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
        a.set_xticks(range(len(lab))); a.set_xticklabels([adlar[i] for i in lab], rotation=40, ha="right")
        a.set_yticks(range(len(lab))); a.set_yticklabels([adlar[i] for i in lab])
        a.set_xlabel("tahmin"); a.set_ylabel("gerçek"); a.set_title(baslik)
    fig.tight_layout(); fig.savefig(os.path.join(klasor, "figure5_confusion_matrices.png")); plt.close(fig)

    # --- Metin raporları + temel model (TF-IDF + Lojistik Regresyon) kıyası ---
    rapor_is = classification_report(ti, pi, labels=is_lab, target_names=[IS_TIPLERI[i] for i in is_lab],
                                     zero_division=0, output_dict=True)
    rapor_alan = classification_report(ta[m], pa[m], labels=alan_lab, target_names=[ALANLAR[i] for i in alan_lab],
                                       zero_division=0, output_dict=True)
    temel = temel_model_kiyasla(df_e, df_d, df_t)
    sonuc = {"ornek_sayilari": {"toplam": len(df), "egitim": len(df_e), "dogrulama": len(df_d), "test": len(df_t)},
             "sinif_dagilimi_is": df["etiket_is_tipi"].value_counts().to_dict(),
             "sinif_dagilimi_alan": df["etiket_alan"].replace("", "etiketsiz").value_counts().to_dict(),
             "test_is_tipi": rapor_is, "test_alan": rapor_alan, "gecmis": gecmis, "temel_model": temel,
             "en_iyi_epok": int(g.loc[eniyi, "epok"])}
    with open(os.path.join(klasor, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2, default=float)
    return sonuc


def temel_model_kiyasla(df_e, df_d, df_t):
    """Karşılaştırma için iki temel yöntem:
      (a) Çoğunluk sınıfı (her örneğe en sık sınıfı ver) - alt sınır,
      (b) TF-IDF (1-2 gram) + Lojistik Regresyon (klasik makine öğrenmesi)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, accuracy_score
    sonuc = {}
    # (b) TF-IDF + LR, her görev için ayrı
    vek = TfidfVectorizer(tokenizer=kelimelere_bol, lowercase=False, min_df=2, ngram_range=(1, 2), token_pattern=None)
    Xe = vek.fit_transform(df_e["ilan_metni"]); Xt = vek.transform(df_t["ilan_metni"])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xe, df_e["etiket_is_tipi"])
    p = lr.predict(Xt)
    sonuc["tfidf_lr_is"] = {"acc": accuracy_score(df_t["etiket_is_tipi"], p),
                            "f1_macro": f1_score(df_t["etiket_is_tipi"], p, average="macro")}
    me, mt = df_e["etiket_alan"].isin(ALANLAR), df_t["etiket_alan"].isin(ALANLAR)
    lr2 = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xe[me.values], df_e.loc[me, "etiket_alan"])
    p2 = lr2.predict(Xt[mt.values])
    sonuc["tfidf_lr_alan"] = {"acc": accuracy_score(df_t.loc[mt, "etiket_alan"], p2),
                              "f1_macro": f1_score(df_t.loc[mt, "etiket_alan"], p2, average="macro")}
    # (a) çoğunluk sınıfı
    for gorev, sutun, maske in (("is", "etiket_is_tipi", None), ("alan", "etiket_alan", mt)):
        yt = df_t[sutun] if maske is None else df_t.loc[maske, sutun]
        ye = df_e[sutun] if maske is None else df_e.loc[me, sutun]
        cogunluk = ye.value_counts().idxmax()
        sonuc[f"cogunluk_{gorev}"] = {"acc": accuracy_score(yt, [cogunluk] * len(yt)),
                                      "f1_macro": f1_score(yt, [cogunluk] * len(yt), average="macro")}
    return sonuc


def capraz_dogrulama(df, k=5):
    """Grup bazlı (ilan bazlı) tabakalı k-katlı çapraz doğrulama. Veri küçük ve ilan boyutları
    çok farklı olduğundan tek bir test bölmesi şansa açıktır; k katın ortalaması ± std raporlanır."""
    from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
    sonuclar = []
    for kat, (ed_idx, t_idx) in enumerate(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
                                          .split(df, df["etiket_is_tipi"], df["grup"]), start=1):
        ed, t = df.iloc[ed_idx].reset_index(drop=True), df.iloc[t_idx].reset_index(drop=True)
        e_i, d_i = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED).split(ed, groups=ed["grup"]))
        model, sozluk, _ = modeli_egit(ed.iloc[e_i], ed.iloc[d_i], sessiz=True)
        yuk = DataLoader(IlanVeriseti(t, sozluk), batch_size=128, collate_fn=topla)
        sk = _skorlar(*_tahmin_et(model, yuk))
        sonuclar.append({"kat": kat, **sk})
        print(f"  kat {kat}: F1(iş) {sk['f1_is']:.3f} | F1(alan) {sk['f1_alan']:.3f}")
    return sonuclar


def egitim_hattini_calistir(sessiz=False):
    """--egit: CSV'yi okur, böler, eğitir, test eder, rapor çıktılarını ve modeli kaydeder."""
    if not os.path.exists(EGITIM_CSV):
        egitim_verisi_hazirla()
    df = pd.read_csv(EGITIM_CSV, keep_default_na=False)
    if not sessiz:
        print("5 katlı grup çapraz doğrulama:")
        cv = capraz_dogrulama(df)
    df_e, df_d, df_t = veriyi_bol(df)
    if not sessiz:
        print("Nihai model (eğitim / doğrulama / test bölmesi):")
    model, sozluk, gecmis = modeli_egit(df_e, df_d, sessiz=sessiz)
    t_yuk = DataLoader(IlanVeriseti(df_t, sozluk), batch_size=128, collate_fn=topla)
    ti, pi, ta, pa = _tahmin_et(model, t_yuk)
    test = _skorlar(ti, pi, ta, pa)
    sonuc = rapor_ciktilari_uret(df, df_e, df_d, df_t, gecmis, ti, pi, ta, pa)
    if not sessiz:
        sonuc["capraz_dogrulama"] = cv
        with open(os.path.join(RAPOR_KLASORU, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(sonuc, f, ensure_ascii=False, indent=2, default=float)
    # Sabit test kümesinin grup kimlikleri saklanır; ileride yeni model bununla kıyaslanır.
    meta = {"egitim_tarihi": datetime.now().isoformat(timespec="seconds"), "test_skoru": test,
            "test_gruplari": sorted(df_t["grup"].unique().tolist()), "surum": 1}
    model_kaydet(model, sozluk, meta)
    return model, sozluk, meta


# ==============================================================================
# BÖLÜM 5: VERİTABANLARI (CANLI DB ve ÖĞRENME HAVUZU AYRI DOSYALARDA)
# ==============================================================================
def baglan(yol):
    con = sqlite3.connect(yol, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    return con


def veritabanlarini_kur():
    with baglan(CANLI_DB) as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS ilanlar (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            link TEXT UNIQUE, baslik TEXT, kurum TEXT, sehir TEXT,
            site_kategori TEXT, yayin_tarihi DATE, son_basvuru DATE,
            aciklama TEXT, eklenme_zamani DATETIME);
        CREATE TABLE IF NOT EXISTS kadrolar (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ilan_id INTEGER REFERENCES ilanlar(id) ON DELETE CASCADE,
            metin TEXT, metin_katli TEXT, unvan_ham TEXT,
            is_tipi TEXT, is_kaynak TEXT, is_guven REAL, alan TEXT, alan_guven REAL);
        CREATE INDEX IF NOT EXISTS ix_ilan_tarih ON ilanlar(yayin_tarihi);
        CREATE INDEX IF NOT EXISTS ix_kadro_ilan ON kadrolar(ilan_id);
        """)
    with baglan(HAVUZ_DB) as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS havuz (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            grup TEXT, ilan_metni TEXT, etiket_is_tipi TEXT, etiket_alan TEXT,
            etiket_kaynagi TEXT, eklenme_zamani DATETIME, kullanildi INTEGER DEFAULT 0,
            UNIQUE(grup, ilan_metni));
        """)


def havuza_ekle(kayitlar):
    """Canlı ilanlardan gelen GÜVENİLİR etiketli kadroları öğrenme havuzuna kopyalar.
    Canlı DB ile eğitim verisi asla aynı tabloyu paylaşmaz; havuz ayrı bir dosyadır."""
    if not kayitlar:
        return 0
    su_an = datetime.now().isoformat(timespec="seconds")
    with baglan(HAVUZ_DB) as con:
        once = con.total_changes
        con.executemany("INSERT OR IGNORE INTO havuz (grup, ilan_metni, etiket_is_tipi, etiket_alan, "
                        "etiket_kaynagi, eklenme_zamani) VALUES (?,?,?,?,?,?)",
                        [(k["grup"], k["metin"], k["is_tipi"], k["alan"], k["kaynak"], su_an) for k in kayitlar])
        return con.total_changes - once


# ==============================================================================
# BÖLÜM 6: OTONOM KAZIYICI (SELENIUM)
# ==============================================================================
# NOT: Eski kodlarda kullanılan ".../kategori/9/personel-alimi-akademik-kadro..." adresi
# sitede aslında "İhale Duyuruları" kategorisini açıyor; bu yüzden eski legacy_scrape.db
# neredeyse tamamen ihale ilanlarıyla dolmuştu. Doğru kategori: kategori/8/kamu-akademik-personel.

def tarayici_ac():
    secenekler = Options()
    for arg in ("--headless=new", "--log-level=3", "--disable-gpu", "--no-sandbox",
                "--disable-dev-shm-usage", "--window-size=1366,900", "--lang=tr-TR"):
        secenekler.add_argument(arg)
    secenekler.add_experimental_option("excludeSwitches", ["enable-logging"])
    # Selenium 4.6+ sürücüyü (chromedriver) kendisi indirir (Selenium Manager).
    return webdriver.Chrome(options=secenekler)


def listeyi_tara(surucu, sayfa):
    """Listeleme sayfasındaki ilan kartlarını toplar: (link, başlık, kurum)."""
    surucu.get(LISTE_URL.format(sayfa=sayfa))
    WebDriverWait(surucu, 25).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a[href^='/ilan/2'], a[href*='ilan.gov.tr/ilan/2']")))
    time.sleep(1.5)  # Angular kartları tamamlasın
    kartlar = []
    for a in surucu.find_elements(By.CSS_SELECTOR, "a[href*='/ilan/']"):
        href = a.get_attribute("href") or ""
        if not re.search(r"/ilan/\d{6,}/", href):
            continue
        baslik = re.sub(r"\s*-\s*[A-Z]{2,4}\d{5,}\s*$", "", (a.get_attribute("title") or "").strip())
        satirlar = [s.strip() for s in (a.text or "").split("\n") if s.strip()]
        kurum = satirlar[0] if len(satirlar) > 1 else ""
        kartlar.append({"link": href.split("?")[0], "baslik": baslik or (satirlar[-1] if satirlar else ""),
                        "kurum": kurum})
    return kartlar


def site_kategorisi(link):
    """URL kısaltmasından ilanın site kategorisi: akademik / kamu personeli / askeri."""
    if "akademik-personel-alimlari" in link:
        return "akademik"
    if "kamu-personel-alim" in link or "askeri-personel" in link:
        return "idari"
    return "diger"


def detayi_tara(surucu, link):
    """İlan detay sayfasını açar; künye bilgilerini, açıklamayı ve tabloları ayrıştırır."""
    surucu.get(link)
    WebDriverWait(surucu, 25).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".tab-content")))
    time.sleep(1.0)
    soup = BeautifulSoup(surucu.page_source, "html.parser")
    kunye = {}
    for d in soup.select(".list-desc"):
        etiket = d.find_previous_sibling()
        if etiket:
            kunye[etiket.get_text(" ", strip=True)] = d.get_text(" ", strip=True)
    h1 = soup.find("h1")
    icerik = soup.select_one(".tab-content")
    aciklama = icerik.get_text("\n", strip=True) if icerik else ""
    tablolar = []
    for t in (icerik.find_all("table") if icerik else []):
        satirlar = [[h.get_text(" ", strip=True) for h in tr.find_all(["td", "th"])] for tr in t.find_all("tr")]
        tablolar.append([s for s in satirlar if any(s)])
    yayin = None
    for anahtar, deger in kunye.items():
        if "tarih" in tr_kucuk(anahtar):
            yayin = yayin or tarih_coz(deger)
    son = re.search(r"son\s*ba[şs]vuru\s*tarihi\W{0,5}(\d{1,2}[./]\d{1,2}[./]\d{4})", tr_kucuk(aciklama))
    return {"baslik": h1.get_text(" ", strip=True) if h1 else "",
            "kurum": kunye.get("İlan Sahibi", ""), "sehir": kunye.get("Şehir", ""),
            "yayin_tarihi": yayin, "son_basvuru": tarih_coz(son.group(1)) if son else None,
            "aciklama": aciklama, "tablolar": tablolar}


def ilani_isle(detay, link, kart, yonetici):
    """Detay sayfasını kadrolara böler, her kadroyu hibrit biçimde etiketler:
      İş tipi : tablo ünvanından (kural) > site kategorisinden (idari) > LSTM tahmini
      Alan    : LSTM tahmini (modelin asıl katkısı; tablolarda alan sütunu yoktur)."""
    kadrolar = tablolardan_kadrolar(detay["tablolar"])
    kategori = site_kategorisi(link)
    if not kadrolar:
        # Tablosuz ilan (ör. sözleşmeli personel duyurusu): başlık + açıklamanın başı tek kadrodur.
        metin = f"{detay['baslik'] or kart.get('baslik', '')} {detay['aciklama'][:1500]}"
        kadrolar = [{"metin": metin, "unvan_ham": "", "is_tipi_kural": kural_is_tipi(detay["baslik"]),
                     "alan_kural": None}]
    tahminler = yonetici.tahmin([k["metin"] for k in kadrolar])
    sonuc = []
    for k, (p_is, g_is, p_alan, g_alan) in zip(kadrolar, tahminler):
        if k["is_tipi_kural"] and k["unvan_ham"]:
            is_tipi, kaynak, guven = k["is_tipi_kural"], "tablo_unvani", 1.0
        elif kategori == "idari":
            is_tipi, kaynak, guven = "idari", "site_kategorisi", 1.0
        elif k["is_tipi_kural"]:
            is_tipi, kaynak, guven = k["is_tipi_kural"], "baslik_kurali", 0.9
        else:
            is_tipi, kaynak, guven = p_is, "lstm", g_is
        sonuc.append({**k, "is_tipi": is_tipi, "is_kaynak": kaynak, "is_guven": guven,
                      "alan": p_alan, "alan_guven": g_alan})
    return sonuc


def kazima_turu(yonetici):
    """Bir tam kazıma turu: listeleme -> yeni ilanlar -> detay -> canlı DB.
    Ayrıca güvenilir etiketli yeni kadroları öğrenme havuzuna kopyalar."""
    surucu, yeni_ilan, havuz_kayitlari = None, 0, []
    try:
        surucu = tarayici_ac()
        with baglan(CANLI_DB) as con:
            bilinen = {r[0] for r in con.execute("SELECT link FROM ilanlar")}
        kartlar = []
        for s in range(1, KAZIMA_SAYFA_SAYISI + 1):
            try:
                kartlar += listeyi_tara(surucu, s)
            except Exception as e:
                LOG.warning("Liste sayfası %s okunamadı: %s", s, e)
        gorulen = set()
        for kart in kartlar:
            if yeni_ilan >= TUR_BASINA_MAKS_DETAY:
                break
            link = kart["link"]
            if link in bilinen or link in gorulen:
                continue
            gorulen.add(link)
            try:
                detay = detayi_tara(surucu, link)
                kadrolar = ilani_isle(detay, link, kart, yonetici)
                # İlan kapısı: model ilanı yüksek güvenle "personel ilanı değil" sayıyorsa kaydetme
                if all(k["is_tipi"] == "ilan_disi" and k["is_guven"] > 0.9 for k in kadrolar):
                    continue
                yayin = detay["yayin_tarihi"] or date.today()   # tarih yoksa: ilk görülme günü
                with baglan(CANLI_DB) as con:
                    cur = con.execute(
                        "INSERT OR IGNORE INTO ilanlar (link, baslik, kurum, sehir, site_kategori, yayin_tarihi, "
                        "son_basvuru, aciklama, eklenme_zamani) VALUES (?,?,?,?,?,?,?,?,?)",
                        (link, detay["baslik"] or kart["baslik"], detay["kurum"] or kart["kurum"], detay["sehir"],
                         site_kategorisi(link), yayin.isoformat(),
                         detay["son_basvuru"].isoformat() if detay["son_basvuru"] else None,
                         detay["aciklama"][:20000], datetime.now().isoformat(timespec="seconds")))
                    ilan_id = cur.lastrowid
                    con.executemany(
                        "INSERT INTO kadrolar (ilan_id, metin, metin_katli, unvan_ham, is_tipi, is_kaynak, is_guven, "
                        "alan, alan_guven) VALUES (?,?,?,?,?,?,?,?,?)",
                        [(ilan_id, k["metin"], katla(k["metin"]), k["unvan_ham"], k["is_tipi"], k["is_kaynak"],
                          k["is_guven"], k["alan"], k["alan_guven"]) for k in kadrolar])
                yeni_ilan += 1
                # Yeni linklerden öğrenme: yalnızca güvenilir iş tipi etiketleri havuza gider.
                # Alan etiketi ise kural ile bulunabiliyorsa (zayıf denetim) eklenir; aksi halde boş.
                grup = "canli_" + (re.search(r"/ilan/(\d+)", link).group(1))
                for k in kadrolar:
                    if k["is_kaynak"] in ("tablo_unvani", "site_kategorisi"):
                        havuz_kayitlari.append({"grup": grup, "metin": k["metin"], "is_tipi": k["is_tipi"],
                                                "alan": k["alan_kural"] or "", "kaynak": k["is_kaynak"]})
                time.sleep(random.uniform(1.0, 2.5))  # siteye karşı nazik ol
            except Exception as e:
                LOG.warning("Detay okunamadı (%s): %s", link, e)
        eklenen = havuza_ekle(havuz_kayitlari)
        LOG.info("Kazıma turu bitti: %d yeni ilan, havuza %d yeni örnek.", yeni_ilan, eklenen)
    except Exception as e:
        LOG.error("Kazıma turu başarısız (internet/Chrome?): %s", e)
    finally:
        try:
            if surucu:
                surucu.quit()   # RAM şişmesin
        except Exception:
            pass
    return yeni_ilan


# ==============================================================================
# BÖLÜM 7: ARKA PLAN İŞ PARÇACIKLARI (KAZIMA, BAKIM, SÜREKLİ ÖĞRENME)
# ==============================================================================
DURUM = {"son_kazima": None, "kazima_basarili": None, "toplam_tur": 0}
DURDUR = threading.Event()


def bakim_yap():
    """7 gün kuralı: yayın tarihi 7 günden eski ilanlar canlı DB'den silinir.
    Silinmeden önce güvenilir etiketli kadroları öğrenme havuzuna arşivlenir."""
    sinir = (date.today() - timedelta(days=GOSTERIM_GUN_SINIRI)).isoformat()
    try:
        with baglan(CANLI_DB) as con:
            eski = con.execute("SELECT k.metin, k.is_tipi, k.is_kaynak, i.link FROM kadrolar k JOIN ilanlar i "
                               "ON i.id = k.ilan_id WHERE i.yayin_tarihi < ?", (sinir,)).fetchall()
            havuza_ekle([{"grup": "canli_" + (re.search(r"/ilan/(\d+)", l).group(1) if re.search(r"/ilan/(\d+)", l) else l),
                          "metin": m, "is_tipi": it, "alan": kural_alan(m) or "", "kaynak": ik}
                         for m, it, ik, l in eski if ik in ("tablo_unvani", "site_kategorisi")])
            con.execute("DELETE FROM kadrolar WHERE ilan_id IN (SELECT id FROM ilanlar WHERE yayin_tarihi < ?)", (sinir,))
            con.execute("DELETE FROM ilanlar WHERE yayin_tarihi < ?", (sinir,))
    except Exception as e:
        LOG.warning("Bakım hatası: %s", e)


def kazima_dongusu(yonetici):
    while not DURDUR.is_set():
        n = kazima_turu(yonetici)
        DURUM.update(son_kazima=datetime.now(), kazima_basarili=n >= 0, toplam_tur=DURUM["toplam_tur"] + 1)
        bakim_yap()
        DURDUR.wait(KAZIMA_ARALIGI_SN)


def surekli_ogrenme_turu(yonetici):
    """Champion/Challenger ince ayar:
      1) training_data.csv (eğitim+doğrulama kısmı) + öğrenme havuzu birleştirilir,
      2) sözlük yeni kelimelerle genişletilir, mevcut ağırlıklardan başlanarak birkaç epok eğitilir,
      3) ORİJİNAL SABİT TEST kümesinde yeni model eskisinden kötü değilse kaydedilip devreye alınır."""
    try:
        with baglan(HAVUZ_DB) as con:
            havuz = pd.read_sql_query("SELECT * FROM havuz", con)
        yeni = int((havuz["kullanildi"] == 0).sum()) if len(havuz) else 0
        if yeni < OGRENME_MIN_YENI_ORNEK:
            return False
        df = pd.read_csv(EGITIM_CSV, keep_default_na=False)
        test_gruplari = set(yonetici.meta.get("test_gruplari", []))
        df_t = df[df["grup"].isin(test_gruplari)]
        df_ed = df[~df["grup"].isin(test_gruplari)]
        havuz_df = havuz.rename(columns={"ilan_metni": "ilan_metni"})[["grup", "ilan_metni", "etiket_is_tipi", "etiket_alan"]]
        birlesik = pd.concat([df_ed[["grup", "ilan_metni", "etiket_is_tipi", "etiket_alan"]], havuz_df], ignore_index=True)
        from sklearn.model_selection import GroupShuffleSplit
        e_idx, d_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
                            .split(birlesik, groups=birlesik["grup"]))
        df_e, df_d = birlesik.iloc[e_idx], birlesik.iloc[d_idx]
        # mevcut modelin kopyası üzerinde ince ayar
        eski_model, eski_sozluk = yonetici.model, yonetici.sozluk
        sozluk = Sozluk(dict(eski_sozluk.k2i))
        sozluk.genislet(havuz_df["ilan_metni"])
        aday = IlanLSTM(len(eski_sozluk))
        aday.load_state_dict(eski_model.state_dict())
        aday, sozluk, _ = modeli_egit(df_e, df_d, sozluk=sozluk, baslangic_modeli=aday, epok=6, sabir=2,
                                      ogrenme_hizi=5e-4, sessiz=True)

        def test_skoru(m, s):
            yuk = DataLoader(IlanVeriseti(df_t, s), batch_size=128, collate_fn=topla)
            return _skorlar(*_tahmin_et(m, yuk))["skor"]

        with yonetici.kilit:
            eski_skor = test_skoru(eski_model, eski_sozluk)
        yeni_skor = test_skoru(aday, sozluk)
        LOG.info("Sürekli öğrenme: eski skor %.4f, aday skor %.4f (havuz %d örnek)", eski_skor, yeni_skor, len(havuz))
        with baglan(HAVUZ_DB) as con:
            con.execute("UPDATE havuz SET kullanildi = 1")
        if yeni_skor >= eski_skor - 0.01:
            meta = {**yonetici.meta, "surum": yonetici.meta.get("surum", 1) + 1,
                    "egitim_tarihi": datetime.now().isoformat(timespec="seconds"),
                    "havuz_ornek_sayisi": int(len(havuz)), "test_skoru_ozet": yeni_skor}
            model_kaydet(aday, sozluk, meta)
            yonetici.degistir(aday, sozluk, meta)
            return True
    except Exception as e:
        LOG.warning("Sürekli öğrenme turu atlandı: %s", e)
    return False


def ogrenme_dongusu(yonetici):
    while not DURDUR.wait(OGRENME_ARALIGI_SN):
        surekli_ogrenme_turu(yonetici)


# ==============================================================================
# BÖLÜM 8: DOĞAL DİL KATMANI (LLM - GEMINI) + KURAL TABANLI YEDEK
# ==============================================================================
SEHIRLER = ["adana", "adıyaman", "afyonkarahisar", "ağrı", "amasya", "ankara", "antalya", "artvin", "aydın",
            "balıkesir", "bilecik", "bingöl", "bitlis", "bolu", "burdur", "bursa", "çanakkale", "çankırı", "çorum",
            "denizli", "diyarbakır", "edirne", "elazığ", "erzincan", "erzurum", "eskişehir", "gaziantep", "giresun",
            "gümüşhane", "hakkari", "hatay", "ısparta", "mersin", "istanbul", "izmir", "kars", "kastamonu",
            "kayseri", "kırklareli", "kırşehir", "kocaeli", "konya", "kütahya", "malatya", "manisa", "kahramanmaraş",
            "mardin", "muğla", "muş", "nevşehir", "niğde", "ordu", "rize", "sakarya", "samsun", "siirt", "sinop",
            "sivas", "tekirdağ", "tokat", "trabzon", "tunceli", "şanlıurfa", "uşak", "van", "yozgat", "zonguldak",
            "aksaray", "bayburt", "karaman", "kırıkkale", "batman", "şırnak", "bartın", "ardahan", "ığdır",
            "yalova", "karabük", "kilis", "osmaniye", "düzce", "isparta"]
_SEHIR_KATLI = {katla(s): s for s in SEHIRLER}

NIYET_SISTEM_PROMPTU = """Sen ilan.gov.tr üzerindeki akademik ve kamu personel ilanları için çalışan bir
arama asistanının "anlama" modülüsün. Görevin, kullanıcının Türkçe mesajını SADECE geçerli bir JSON
nesnesine çevirmektir. Asla açıklama, markdown veya ek metin yazma.

JSON şeması:
{
  "niyet": "arama" | "devam" | "selam" | "yardim" | "tesekkur" | "alakasiz" | "cikis",
  "sehir": string | null,              // küçük harf, Türkçe karakterli il adı: "istanbul", "izmir", "şanlıurfa"
  "spesifik_kelime": string | null,    // bölüm/meslek KÖKÜ, küçük harf (aşağıdaki kurallara bak)
  "is_tipi": "argor" | "ogrgor" | "ogruye" | "idari" | null,
  "alan": "muhendislik" | "fen_bilimleri" | "saglik" | "sosyal_bilimler" | "egitim" | "sanat_mimarlik" | null
}

İş tipi sözlüğü:
- argor  : araştırma görevlisi, arş. gör., asistan (akademik)
- ogrgor : öğretim görevlisi, öğr. gör., okutman, ders verecek öğretim elemanı
- ogruye : öğretim üyesi, profesör, doçent, dr. öğr. üyesi, "hoca kadrosu"
- idari  : memur, sözleşmeli personel, 4/B, uzman yardımcısı, teknisyen, temizlik/hademe, güvenlik, şoför, büro

spesifik_kelime kuralları (ÇOK ÖNEMLİ - veritabanında birebir metin olarak aranacak):
- Kullanıcının istediği EN DAR bölüm/meslek ifadesinin kökünü yaz; "mühendisliği", "bölümü", "ilanı",
  "kadrosu", "alımı" gibi genel ekleri ve kelimeleri at.
  "bilgisayar mühendisliği" -> "bilgisayar" ; "inşaat mühendisi" -> "inşaat" ; "elektrik-elektronik" -> "elektrik"
  "hemşirelik" -> "hemşire" ; "hademe" veya "temizlikçi" -> "temizlik" ; "tarih bölümü" -> "tarih"
  "türk dili ve edebiyatı" -> "türk dili" ; "makine" -> "makine"
- Kullanıcı yalnızca GENİŞ bir alan söylediyse ("mühendislik ilanları", "sosyal bilimler", "sağlık alanı")
  spesifik_kelime null olur ve "alan" doldurulur.
- Şehir, iş tipi ve alan kelimelerini spesifik_kelime'ye koyma.

Bağlam: Sana önceki aramanın filtresi verilecek. Kullanıcı "peki ankara'da?", "bir de doçent olanlar"
gibi eksik bir takip sorusu sorarsa önceki filtreyi koru, yalnızca değişeni güncelle.
"devam", "daha fazla", "sonrakiler", "başka var mı" -> niyet "devam".
İlanlarla ilgisi olmayan sorular (hava durumu, futbol, kod yazdırma vb.) -> niyet "alakasiz".

Örnekler:
Kullanıcı: "İstanbul'daki bilgisayar mühendisliği araştırma görevlisi ilanlarını bul"
{"niyet":"arama","sehir":"istanbul","spesifik_kelime":"bilgisayar","is_tipi":"argor","alan":"muhendislik"}
Kullanıcı: "ankarada hemşire alımı var mı"
{"niyet":"arama","sehir":"ankara","spesifik_kelime":"hemşire","is_tipi":null,"alan":"saglik"}
Kullanıcı: "sosyal bilimlerde doçent kadroları"
{"niyet":"arama","sehir":null,"spesifik_kelime":null,"is_tipi":"ogruye","alan":"sosyal_bilimler"}
Kullanıcı: "devam et"
{"niyet":"devam","sehir":null,"spesifik_kelime":null,"is_tipi":null,"alan":null}
Kullanıcı: "bugün hava nasıl?"
{"niyet":"alakasiz","sehir":null,"spesifik_kelime":null,"is_tipi":null,"alan":null}"""

SUNUM_SISTEM_PROMPTU = """Sen ilan.gov.tr için çalışan sıcak, nazik ve yardımsever bir kariyer asistanısın.
Kullanıcıyla "sen" diye, samimi ama saygılı konuşursun. Robotik kalıplar, JSON, teknik terim,
hata kodu veya İngilizce kullanmazsın. Sana verilen bilgilerin DIŞINDA hiçbir ilan, tarih ya da link uydurmazsın.

İlan sunarken:
- Bir cümlelik doğal bir girişle başla (kaç ilan bulduğunu söyle).
- Her ilanı numaralı ayrı bir madde olarak yaz: kurum, şehir, eşleşen kadro(lar)ın kısa özeti ve
  varsa son başvuru tarihi. Her maddenin sonunda link'i, sana verildiği gibi, değiştirmeden ve
  markdown kullanmadan düz metin olarak ver (terminalde tıklanabilmesi için).
- Kalan ilan varsa sonunda "devam" yazarak sonrakileri görebileceğini nazikçe hatırlat.
- Toplam 180 kelimeyi geçme."""


def anahtarlari_yukle():
    """API anahtarlarını önce ortam değişkenlerinden, yoksa yerel api_keys.json'dan okur."""
    anahtarlar = {"gemini": "", "groq": ""}
    try:
        if os.path.exists(ANAHTAR_DOSYASI):
            with open(ANAHTAR_DOSYASI, encoding="utf-8") as f:
                anahtarlar.update({k: str(v).strip() for k, v in json.load(f).items() if k in anahtarlar})
    except Exception as e:
        LOG.warning("Anahtar dosyası okunamadı: %s", e)
    for ad, degisken in (("gemini", "GEMINI_API_KEY"), ("groq", "GROQ_API_KEY")):
        if os.environ.get(degisken, "").strip():
            anahtarlar[ad] = os.environ[degisken].strip()
    return anahtarlar


def anahtarlari_kaydet(anahtarlar):
    try:
        with open(ANAHTAR_DOSYASI, "w", encoding="utf-8") as f:
            json.dump(anahtarlar, f, ensure_ascii=False, indent=2)
        try:
            os.chmod(ANAHTAR_DOSYASI, 0o600)   # yalnızca dosya sahibi okuyabilsin (Linux/Mac)
        except Exception:
            pass
        return True
    except Exception as e:
        LOG.warning("Anahtar dosyası yazılamadı: %s", e)
        return False


class LLMIstemcisi:
    """İki ücretsiz sağlayıcılı LLM katmanı.
      1) Google Gemini (google-genai SDK, ücretsiz katman)
      2) Groq (OpenAI uyumlu REST uç noktası, ücretsiz katman) -> Gemini kotası dolarsa / bağlantı koparsa
    İkisi de yoksa veya ikisi de cevap vermezse None döner; çağıran taraf kural tabanlı yedeğe geçer.
    Böylece kullanıcı hiçbir koşulda hata mesajı görmez."""

    def __init__(self, anahtarlar=None):
        anahtarlar = anahtarlar or anahtarlari_yukle()
        self.gemini, self.types, self.gemini_model = None, None, None
        self.groq_anahtari, self.groq_model = anahtarlar.get("groq") or None, None
        self.son_saglayici = None
        if anahtarlar.get("gemini"):
            try:
                from google import genai
                from google.genai import types
                self.gemini, self.types = genai.Client(api_key=anahtarlar["gemini"]), types
            except Exception as e:
                LOG.warning("Gemini istemcisi kurulamadı: %s", e)

    @property
    def aktif(self):
        return self.gemini is not None or self.groq_anahtari is not None

    @property
    def saglayicilar(self):
        return [ad for ad, var in (("Gemini", self.gemini), ("Groq", self.groq_anahtari)) if var]

    def _gemini(self, sistem, icerik, json_mu, sicaklik):
        for ad in ([self.gemini_model] if self.gemini_model else GEMINI_MODELLERI):
            try:
                cfg = self.types.GenerateContentConfig(
                    system_instruction=sistem, temperature=sicaklik,
                    response_mime_type="application/json" if json_mu else "text/plain")
                metin = (self.gemini.models.generate_content(model=ad, contents=icerik, config=cfg).text or "").strip()
                if metin:
                    self.gemini_model = ad
                    return metin
            except Exception as e:
                LOG.warning("Gemini çağrısı başarısız (%s): %s", ad, e)
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    break           # kota doldu: diğer Gemini modellerini deneme, Groq'a geç
        return None

    def _groq(self, sistem, icerik, json_mu, sicaklik):
        """Groq'un OpenAI uyumlu uç noktası; ek kütüphane gerektirmez (standart urllib)."""
        import urllib.request
        for ad in ([self.groq_model] if self.groq_model else GROQ_MODELLERI):
            govde = {"model": ad, "temperature": sicaklik, "max_tokens": 900,
                     "messages": [{"role": "system", "content": sistem}, {"role": "user", "content": icerik}]}
            if json_mu:
                govde["response_format"] = {"type": "json_object"}
            try:
                istek = urllib.request.Request(
                    GROQ_URL, data=json.dumps(govde).encode("utf-8"), method="POST",
                    headers={"Authorization": f"Bearer {self.groq_anahtari}", "Content-Type": "application/json",
                             "User-Agent": "ilan-gov-tr-ai-assistant"})
                with urllib.request.urlopen(istek, timeout=30) as yanit:
                    veri = json.loads(yanit.read().decode("utf-8"))
                metin = (veri["choices"][0]["message"]["content"] or "").strip()
                if metin:
                    self.groq_model = ad
                    return metin
            except Exception as e:
                LOG.warning("Groq çağrısı başarısız (%s): %s", ad, e)
        return None

    def uret(self, sistem, icerik, json_mu=False, sicaklik=0.6):
        if self.gemini is not None:
            metin = self._gemini(sistem, icerik, json_mu, sicaklik)
            if metin:
                self.son_saglayici = "Gemini"
                return metin
        if self.groq_anahtari:
            metin = self._groq(sistem, icerik, json_mu, sicaklik)
            if metin:
                self.son_saglayici = "Groq"
                return metin
        return None


def anahtar_sihirbazi(zorla=False):
    """İlk açılışta (veya --anahtar ile) kullanıcıdan ÜCRETSİZ API anahtarlarını ister, dener ve kaydeder.
    Enter'a basarak geçilebilir; anahtarsız da asistan yerleşik anlama modülüyle çalışır."""
    anahtarlar = anahtarlari_yukle()
    if not zorla and (anahtarlar["gemini"] or anahtarlar["groq"] or os.path.exists(ANAHTAR_DOSYASI)):
        return anahtarlar
    print("\n  🔑 Daha doğal sohbet için ücretsiz bir yapay zekâ anahtarı ekleyebilirsin (isteğe bağlı).")
    print("     • Gemini: https://aistudio.google.com  →  'Get API key'  (faturalandırma açma, ücretsiz kalır)")
    print("     • Groq  : https://console.groq.com/keys  (kart bilgisi istemez)")
    print("     Anahtar yalnızca bu bilgisayarda, api_keys.json dosyasında saklanır. Geçmek için Enter.\n")
    for ad, etiket in (("gemini", "Gemini anahtarı"), ("groq", "Groq anahtarı")):
        try:
            girdi = input(f"  {etiket}: ").strip()
        except (EOFError, KeyboardInterrupt):
            girdi = ""
        if not girdi:
            continue
        deneme = LLMIstemcisi({ad: girdi})
        if deneme.uret("Yalnızca 'tamam' yaz.", "test", sicaklik=0.0):
            anahtarlar[ad] = girdi
            print("    ✔ Anahtar çalışıyor.")
        else:
            print("    Bu anahtarla bağlantı kuramadım; yine de kaydediyorum, sonra --anahtar ile değiştirebilirsin.")
            anahtarlar[ad] = girdi
    anahtarlari_kaydet(anahtarlar)
    return anahtarlar


BOS_FILTRE = {"sehir": None, "spesifik_kelime": None, "is_tipi": None, "alan": None}


def niyeti_kuralla_coz(mesaj, onceki):
    """LLM yokken çalışan yedek çözümleyici (anahtar kelime + il listesi)."""
    m = katla(re.sub(r"[’'`´]", "", mesaj))          # "İstanbul'daki" -> "istanbuldaki"
    kelimeler = re.findall(r"[a-z0-9]+", m)
    if re.fullmatch(r"\s*(q|cikis|cik|kapat|exit|quit|gorusuruz|hoscakal)\s*", m):
        return {"niyet": "cikis", **BOS_FILTRE}
    if re.search(r"\b(devam|daha fazla|sonraki|kalan|baska var)", m):
        return {"niyet": "devam", **BOS_FILTRE}
    if re.search(r"\b(tesekkur|sagol|eyvallah)", m):
        return {"niyet": "tesekkur", **BOS_FILTRE}
    if re.fullmatch(r"\s*(merhaba|selam|selamlar|hey|iyi gunler|gunaydin)\W*", m):
        return {"niyet": "selam", **BOS_FILTRE}
    if re.search(r"\b(yardim|ne yapabilirsin|nasil kullan)", m):
        return {"niyet": "yardim", **BOS_FILTRE}
    f = dict(BOS_FILTRE)
    takip = bool(re.match(r"\s*(peki|bir de|ya |sadece|ayrica)", m)) and onceki
    if takip:
        f.update(onceki)
    sehir_kelimeleri = set()
    for k in kelimeler:
        for ek in ("", "daki", "deki", "taki", "teki", "da", "de", "ta", "te", "dan", "den", "tan", "ten",
                   "nin", "nın", "in", "un", "a", "e", "ya", "ye", "ndaki", "ndeki", "nda", "nde"):
            kok_ = k[:-len(ek)] if ek and k.endswith(ek) else (k if not ek else None)
            if kok_ and kok_ in _SEHIR_KATLI:
                f["sehir"] = _SEHIR_KATLI[kok_]
                sehir_kelimeleri.add(k)
                break
    tipler = [("argor", r"arastirma gorevli|ars\.? ?gor|asistan"), ("ogrgor", r"ogretim gorevli|ogr\.? ?gor|okutman"),
              ("ogruye", r"ogretim uyes|profesor|docent|dr\.? ?ogr|hoca"),
              ("idari", r"memur|sozlesmeli|4/b|uzman yardimci|teknisyen|temizlik|hademe|guvenlik|sofor|idari|buro")]
    for etiket, desen in tipler:
        if re.search(desen, m):
            f["is_tipi"] = etiket
            break
    genis = {"muhendislik": r"\bmuhendislik\b", "sosyal_bilimler": r"sosyal bilim", "saglik": r"saglik alan|\bsaglik\b",
             "fen_bilimleri": r"fen bilim", "egitim": r"egitim bilim|ogretmenlik", "sanat_mimarlik": r"sanat|mimarlik"}
    for alan, desen in genis.items():
        if re.search(desen, m):
            f["alan"] = alan
    # Arama dışı / genel kelime kökleri: bunlarla başlayan kelimeler spesifik kelime olamaz
    dur_kokleri = ("ilan", "bul", "goster", "sirala", "bana", "beni", "icin", "var", "mi", "mu", "ne", "alim", "kadro",
                   "bolum", "muhendis", "arastirma", "gorevli", "ogretim", "uyes", "lazim", "isti", "ari", "acaba",
                   "olan", "tum", "hepsi", "yeni", "guncel", "personel", "peki", "bir", "sadece", "ayrica", "hangi",
                   "nere", "univers", "alan", "bilim", "sosyal", "fen", "saglik", "egitim", "sanat", "profesor",
                   "docent", "hoca", "asistan", "okutman", "memur", "sozlesmeli", "idari", "fakulte", "yuksekokul",
                   "myo", "meslek", "ve", "ile", "veya", "su", "bu", "o", "nasil", "nedir", "kac", "bugun", "yarin",
                   "hava", "lutfen", "misin", "musun", "eleman", "akademik", "kamu", "is", "dr", "ogr", "ars", "gor",
                   "doc", "prof", "uzman", "pozisyon", "basvuru", "listele")
    # Kısa kökler (ve, bu, ne, dr...) yalnızca TAM eşleşir; aksi halde 'veteriner' 've' ile elenirdi.
    adaylar = [k for k in kelimeler if len(k) > 2 and k not in sehir_kelimeleri
               and not any((k.startswith(d) if len(d) >= 4 else k == d) for d in dur_kokleri)]
    if "hademe" in m or "temizlik" in m:
        f["spesifik_kelime"] = "temizlik"
    elif adaylar and alan_sozlugunde_mi(adaylar[0]):
        f["spesifik_kelime"] = adaylar[0]
    elif not takip:
        f["spesifik_kelime"] = None
    # Hiçbir ilan kavramı yakalanamadıysa (ör. "bugün hava nasıl?", "futbol") soru alakasızdır.
    niyet = "arama" if any(f.values()) or re.search(r"\bilan", m) else "alakasiz"
    return {"niyet": niyet, **f}


_ALAN_SOZLUGU_KOKLERI = None


def alan_sozlugunde_mi(kelime):
    """Kural yedeğinde bir kelimenin gerçekten bir bölüm/meslek adı olup olmadığını denetler:
    alan anahtarlarında, iş tipi kurallarında veya canlı DB'deki kadro metinlerinde geçiyor mu?"""
    global _ALAN_SOZLUGU_KOKLERI
    if _ALAN_SOZLUGU_KOKLERI is None:
        _ALAN_SOZLUGU_KOKLERI = {kok(w) for _, a in _ALAN_DESENLERI for w in a.split() if len(w) > 2}
        _ALAN_SOZLUGU_KOKLERI |= {kok(w) for w in "hemsire tekniker teknisyen sofor guvenlik temizlik avukat "
                                  "programci sekreter veteriner ebe asci bahcivan itfaiye zabita".split()}
    kk = kok(kelime)
    if kk in _ALAN_SOZLUGU_KOKLERI:
        return True
    try:
        with baglan(CANLI_DB) as con:
            return con.execute("SELECT 1 FROM kadrolar WHERE metin_katli LIKE ? LIMIT 1",
                               (f"%{kk}%",)).fetchone() is not None
    except Exception:
        return False


def niyeti_anla(mesaj, onceki, llm):
    """Önce LLM ile (few-shot + JSON çıktı modu), olmazsa kural tabanlı yedekle niyet çıkarır."""
    if llm.aktif:
        icerik = f"Önceki filtre: {json.dumps(onceki or BOS_FILTRE, ensure_ascii=False)}\nKullanıcı: \"{mesaj}\""
        ham = llm.uret(NIYET_SISTEM_PROMPTU, icerik, json_mu=True, sicaklik=0.0)
        if ham:
            try:
                ham = re.sub(r"^```(?:json)?|```$", "", ham.strip()).strip()
                v = json.loads(ham)
                sonuc = {"niyet": v.get("niyet") or "arama"}
                for k in BOS_FILTRE:
                    deger = v.get(k)
                    sonuc[k] = tr_kucuk(deger).strip() if isinstance(deger, str) and deger.strip() else None
                if sonuc["is_tipi"] not in IS_TIPLERI:
                    sonuc["is_tipi"] = None
                if sonuc["alan"] not in ALANLAR:
                    sonuc["alan"] = None
                return sonuc
            except Exception as e:
                LOG.warning("LLM JSON çözülemedi: %s | %s", e, ham[:200])
    return niyeti_kuralla_coz(mesaj, onceki)


# ==============================================================================
# BÖLÜM 9: ARAMA MOTORU (7 GÜN KURALI + KATI ALT FİLTRE + 5'ERLİ SAYFALAMA)
# ==============================================================================
def ilan_ara(filtre):
    """Canlı veritabanında arar. Sıra:
       1) Sadece son 7 gün içinde yayımlanmış ilanlar (eski ilan asla gösterilmez)
       2) Şehir filtresi
       3) İş tipi filtresi (kadro düzeyinde)
       4) KATI ALT FİLTRE: spesifik_kelime kadro metninde (katlanmış, kök bazlı) GEÇMEK ZORUNDA.
          Böylece 'bilgisayar' arayan, aynı ilandaki 'inşaat' kadrosunu görmez; yalnızca
          bilgisayar kadrosu olan ilanlar listelenir.
       5) Spesifik kelime yoksa LSTM'in alan tahmini ile ana kategori filtresi.
    Dönen her öğe: ilan bilgisi + eşleşen kadroların kısa listesi."""
    sinir = (date.today() - timedelta(days=GOSTERIM_GUN_SINIRI)).isoformat()
    with baglan(CANLI_DB) as con:
        df = pd.read_sql_query(
            "SELECT i.id AS ilan_id, i.link, i.baslik, i.kurum, i.sehir, i.yayin_tarihi, i.son_basvuru, "
            "k.metin, k.metin_katli, k.unvan_ham, k.is_tipi, k.alan, k.alan_guven "
            "FROM ilanlar i JOIN kadrolar k ON k.ilan_id = i.id WHERE i.yayin_tarihi >= ?", con, params=(sinir,))
    if df.empty:
        return []
    if filtre.get("sehir"):
        df = df[df["sehir"].map(katla) == katla(filtre["sehir"])]
    if filtre.get("is_tipi"):
        df = df[df["is_tipi"] == filtre["is_tipi"]]
    if filtre.get("spesifik_kelime"):
        kokler = [kok(k) for k in re.findall(r"[a-zçğıöşüâîû]+", tr_kucuk(filtre["spesifik_kelime"])) if len(k) > 1]
        for kk in kokler:
            df = df[df["metin_katli"].str.contains(re.escape(kk), regex=True, na=False)]
    elif filtre.get("alan"):
        df = df[df["alan"] == filtre["alan"]]
    if df.empty:
        return []
    sonuc = []
    for ilan_id, grup in df.sort_values("yayin_tarihi", ascending=False).groupby("ilan_id", sort=False):
        ilk = grup.iloc[0]
        ozetler = []
        for _, r in grup.head(3).iterrows():
            unvan = r["unvan_ham"] or IS_TIPI_ACIKLAMA.get(r["is_tipi"], "")
            ozetler.append(f"{unvan}: {r['metin'][:110]}".strip(": "))
        # Sitedeki bazı adreslerde boşluk var; terminalde tıklanabilir olması için %20'ye çeviriyoruz.
        link = re.sub(r"\s", "%20", str(ilk["link"]).strip())
        sonuc.append({"baslik": ilk["baslik"], "kurum": ilk["kurum"], "sehir": ilk["sehir"],
                      "yayin_tarihi": ilk["yayin_tarihi"], "son_basvuru": ilk["son_basvuru"],
                      "eslesen_kadro_sayisi": len(grup), "kadrolar": ozetler, "link": link})
    sonuc.sort(key=lambda x: x["yayin_tarihi"] or "", reverse=True)
    return sonuc


def filtre_ozeti(f):
    parcalar = []
    if f.get("sehir"):
        s = f["sehir"]
        parcalar.append(("İ" if s[0] == "i" else "I" if s[0] == "ı" else s[0].upper()) + s[1:])
    if f.get("spesifik_kelime"):
        parcalar.append(f"'{f['spesifik_kelime']}'")
    elif f.get("alan"):
        parcalar.append(ALAN_ACIKLAMA[f["alan"]])
    if f.get("is_tipi"):
        parcalar.append(IS_TIPI_ACIKLAMA[f["is_tipi"]])
    return ", ".join(parcalar) or "tüm güncel personel ilanları"


# ==============================================================================
# BÖLÜM 10: SOHBET (OTURUM HAFIZASI + CEVAP ÜRETİMİ)
# ==============================================================================
class Oturum:
    def __init__(self):
        self.filtre = dict(BOS_FILTRE)
        self.sonuclar, self.gosterilen = [], 0


def sablon_sunum(parti, toplam, gosterilen, filtre, devam_mi):
    """LLM yokken kullanılan sıcak, şablon tabanlı sunum."""
    if devam_mi:
        giris = f"Tabii, sıradaki {len(parti)} ilanı getirdim:"
    else:
        giris = (f"Harika haber! {filtre_ozeti(filtre)} için son {GOSTERIM_GUN_SINIRI} günde {toplam} ilan buldum. "
                 f"İlk {len(parti)} tanesi şöyle:")
    satirlar = [giris, ""]
    for i, ilan in enumerate(parti, start=gosterilen - len(parti) + 1):
        son = f" | Son başvuru: {ilan['son_basvuru']}" if ilan.get("son_basvuru") else ""
        satirlar.append(f"{i}. {ilan['kurum'] or ilan['baslik']} ({(ilan['sehir'] or '').title()}){son}")
        for k in ilan["kadrolar"][:2]:
            satirlar.append(f"   • {k}")
        satirlar.append(f"   🔗 {ilan['link']}")
    kalan = toplam - gosterilen
    satirlar.append("")
    satirlar.append(f"Daha {kalan} ilan var; görmek istersen 'devam' yazman yeterli. 😊" if kalan > 0
                    else "Bu aramadaki tüm ilanları gösterdim. Başka bir şey aramak ister misin?")
    return "\n".join(satirlar)


def sunum_uret(mesaj, parti, toplam, gosterilen, filtre, devam_mi, llm):
    """Sonuçları LLM ile doğal dilde sunar. LLM bir linki atlarsa/bozarsa (halüsinasyon
    koruması) şablon sunuma geri döner; böylece kullanıcı her zaman doğru linkleri görür."""
    if llm.aktif:
        veri = {"kullanici_mesaji": mesaj, "filtre_ozeti": filtre_ozeti(filtre), "toplam_bulunan": toplam,
                "bu_partide_gosterilen": len(parti), "kalan": toplam - gosterilen, "devam_istegi": devam_mi,
                "ilanlar": [{k: v for k, v in i.items() if k != "yayin_tarihi"} for i in parti]}
        metin = llm.uret(SUNUM_SISTEM_PROMPTU, json.dumps(veri, ensure_ascii=False, default=str))
        if metin and all(i["link"] in metin for i in parti):
            return metin
    return sablon_sunum(parti, toplam, gosterilen, filtre, devam_mi)


def durum_cevabi(tur, mesaj, filtre, llm):
    """Bulunamayan ilan, alakasız soru, selam vb. durumlar için nazik cevap (LLM veya şablon)."""
    canli_bos = True
    try:
        with baglan(CANLI_DB) as con:
            canli_bos = con.execute("SELECT COUNT(*) FROM ilanlar").fetchone()[0] == 0
    except Exception:
        pass
    baglanti_sorunu = DURUM["toplam_tur"] > 0 and canli_bos
    talimatlar = {
        "bulunamadi": f"Kullanıcının araması ({filtre_ozeti(filtre)}) için son {GOSTERIM_GUN_SINIRI} günde uygun ilan "
                      "yok. Üzgün olduğunu nazikçe söyle, aramayı genişletmesi için 1-2 somut öneri ver (şehir "
                      "kısıtını kaldırmak, daha genel bir bölüm adı, farklı kadro türü). Listenin 10 dakikada bir "
                      "güncellendiğini, yeni ilan gelince tekrar sorabileceğini belirt.",
        "veri_yok": "Canlı ilan listesi şu an boş; arka plandaki tarayıcı ilk turunu tamamlıyor ya da siteye "
                    "erişilemiyor. Kullanıcıyı teknik detay vermeden nazikçe bilgilendir, birkaç dakika sonra "
                    "tekrar denemesini rica et.",
        "alakasiz": "Kullanıcı ilanlarla ilgisi olmayan bir şey sordu. Kibarca yalnızca ilan.gov.tr'deki akademik "
                    "ve kamu personel ilanlarında yardımcı olabileceğini söyle ve örnek bir arama cümlesi öner.",
        "selam": "Kullanıcı selam verdi. Sıcak bir karşılama yap, neler yapabildiğini bir cümleyle anlat, "
                 "örnek bir arama cümlesi öner.",
        "yardim": "Kullanıcı nasıl kullanacağını soruyor. Şehir, bölüm ve kadro türü söyleyerek arama "
                  "yapabileceğini, 'devam' ile sonraki 5 ilanı görebileceğini ve 'çıkış' ile ayrılabileceğini anlat.",
        "tesekkur": "Kullanıcı teşekkür etti. Sıcak bir dille rica et ve başarılar dile.",
        "devam_yok": "Kullanıcı 'devam' dedi ama gösterilecek başka ilan yok ya da henüz bir arama yapılmadı. "
                     "Nazikçe belirt ve yeni bir arama yapmasını öner.",
        "cikis": "Kullanıcı ayrılıyor. Kısa ve sıcak bir veda et, başvurularında başarılar dile.",
    }
    if tur == "bulunamadi" and canli_bos:
        tur = "veri_yok"
    if llm.aktif:
        ek = " (Not: siteye bağlantıda sorun olabilir.)" if baglanti_sorunu else ""
        metin = llm.uret(SUNUM_SISTEM_PROMPTU, f"Durum: {talimatlar[tur]}{ek}\nKullanıcı mesajı: \"{mesaj}\"\n"
                                               "En fazla 3 cümle yaz.")
        if metin:
            return metin
    sablonlar = {
        "bulunamadi": f"Üzgünüm, {filtre_ozeti(filtre)} için son {GOSTERIM_GUN_SINIRI} günde yayımlanmış bir ilan "
                      "bulamadım. 🙁 İstersen şehir kısıtını kaldırarak ya da daha genel bir bölüm adıyla tekrar "
                      "deneyelim. Liste 10 dakikada bir yenileniyor, yeni ilan düşerse hemen görebilirsin.",
        "veri_yok": "Güncel ilan listesini şu anda hazırlıyorum; arka plandaki taramam henüz tamamlanmadı ya da "
                    "siteye ulaşmakta küçük bir gecikme var. Birkaç dakika sonra tekrar sorar mısın? 🙏",
        "alakasiz": "Bu konuda pek yardımcı olamıyorum ama ilan.gov.tr'deki akademik ve kamu personel ilanlarında "
                    "elimden geleni yaparım! Örneğin: \"İzmir'de tarih bölümü öğretim üyesi ilanları\" diyebilirsin.",
        "selam": "Merhaba, hoş geldin! 👋 ilan.gov.tr'deki güncel akademik ve kamu personel ilanlarını senin için "
                 "tarayabilirim. Örneğin: \"Ankara'daki bilgisayar mühendisliği araştırma görevlisi ilanları\"",
        "yardim": "Bana şehir, bölüm ve kadro türünü söylemen yeterli; örneğin \"Konya'da hemşire alımı\" ya da "
                  "\"sosyal bilimlerde doçent kadroları\". Sonuçları 5'erli getiririm, 'devam' ile sonrakilere "
                  "geçebilir, 'çıkış' ile ayrılabilirsin.",
        "tesekkur": "Rica ederim, ne demek! Başvurularında bol şans dilerim. 🍀",
        "devam_yok": "Şu an gösterebileceğim başka ilan kalmadı. Yeni bir arama yapmak ister misin?",
        "cikis": "Görüşmek üzere! Başvurularında başarılar dilerim. 🌟",
    }
    return sablonlar[tur]


def mesaji_isle(mesaj, oturum, llm):
    """Tek bir kullanıcı mesajını uçtan uca işler ve kullanıcıya gösterilecek metni döndürür.
    Hiçbir koşulda istisna dışarı sızmaz."""
    try:
        niyet = niyeti_anla(mesaj, oturum.filtre, llm)
        tur = niyet.get("niyet")
        if tur == "devam":
            if oturum.gosterilen >= len(oturum.sonuclar):
                return durum_cevabi("devam_yok", mesaj, oturum.filtre, llm)
            parti = oturum.sonuclar[oturum.gosterilen:oturum.gosterilen + SAYFA_BOYUTU]
            oturum.gosterilen += len(parti)
            return sunum_uret(mesaj, parti, len(oturum.sonuclar), oturum.gosterilen, oturum.filtre, True, llm)
        if tur in ("selam", "yardim", "tesekkur", "alakasiz", "cikis"):
            return durum_cevabi(tur, mesaj, oturum.filtre, llm)
        filtre = {k: niyet.get(k) for k in BOS_FILTRE}
        oturum.filtre = filtre
        oturum.sonuclar = ilan_ara(filtre)
        oturum.gosterilen = 0
        if not oturum.sonuclar:
            return durum_cevabi("bulunamadi", mesaj, filtre, llm)
        parti = oturum.sonuclar[:SAYFA_BOYUTU]
        oturum.gosterilen = len(parti)
        return sunum_uret(mesaj, parti, len(oturum.sonuclar), oturum.gosterilen, filtre, False, llm)
    except Exception as e:
        LOG.exception("Mesaj işlenemedi: %s", e)
        return ("Kusura bakma, isteğini işlerken küçük bir aksaklık yaşadım. 🙏 Aynı soruyu biraz farklı "
                "kelimelerle tekrar sorar mısın?")


def sohbet_dongusu(llm):
    oturum = Oturum()
    while True:
        try:
            mesaj = input("\nSen: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAsistan: Görüşmek üzere! Başvurularında başarılar dilerim. 🌟")
            break
        if not mesaj:
            continue
        cevap = mesaji_isle(mesaj, oturum, llm)
        print(f"\nAsistan: {cevap}")
        if katla(mesaj) in ("q", "cikis", "cik", "exit", "quit", "kapat"):
            break


# ==============================================================================
# BÖLÜM 11: ANA PROGRAM
# ==============================================================================
def model_hazirla():
    """Kaydedilmiş model varsa yükler; yoksa eğitim verisini hazırlayıp eğitir."""
    if os.path.exists(os.path.join(MODEL_KLASORU, "ad_lstm.pt")):
        return model_yukle()
    print("  İlk çalıştırma: eğitim verisi hazırlanıyor ve LSTM modeli eğitiliyor (birkaç dakika sürebilir)...")
    return egitim_hattini_calistir(sessiz=True)


def main():
    ap = argparse.ArgumentParser(description="ilan.gov.tr doğal dil destekli otonom asistan")
    ap.add_argument("--veri-hazirla", action="store_true", help="training_data.csv dosyasını üret")
    ap.add_argument("--egit", action="store_true", help="modeli eğit, metrik ve grafikleri üret")
    ap.add_argument("--tek-tarama", action="store_true", help="tek bir kazıma turu yap ve çık")
    ap.add_argument("--anahtar", action="store_true", help="ücretsiz Gemini / Groq anahtarlarını gir veya değiştir")
    arg = ap.parse_args()

    if arg.anahtar:
        a = anahtar_sihirbazi(zorla=True)
        print("  Kaydedildi. Etkin sağlayıcılar: " + (", ".join(LLMIstemcisi(a).saglayicilar) or "yok"))
        return

    if arg.veri_hazirla:
        df = egitim_verisi_hazirla()
        print(f"training_data.csv hazır: {len(df)} örnek")
        print(df["etiket_is_tipi"].value_counts().to_string())
        return
    if arg.egit:
        egitim_verisi_hazirla()
        _, _, meta = egitim_hattini_calistir()
        print(f"\nModel kaydedildi. Test skoru: {json.dumps(meta['test_skoru'], ensure_ascii=False)}")
        print(f"Grafikler ve metrikler: {RAPOR_KLASORU}")
        return

    print("=" * 64)
    print("  🤖 ilan.gov.tr Doğal Dil Destekli Otonom Asistan başlatılıyor")
    print("=" * 64)
    veritabanlarini_kur()
    try:
        model, sozluk, meta = model_hazirla()
    except Exception as e:
        LOG.exception("Model hazırlanamadı: %s", e)
        print("  Asistan şu an başlatılamadı. Lütfen 'data/academic_ads_clean.json' dosyasının yerinde "
              "olduğundan emin olup tekrar dene. Ayrıntılar assistant.log dosyasında.")
        return
    yonetici = ModelYoneticisi(model, sozluk, meta)

    if arg.tek_tarama:
        n = kazima_turu(yonetici)
        bakim_yap()
        print(f"  Tarama tamamlandı: {n} yeni ilan eklendi.")
        return

    threading.Thread(target=kazima_dongusu, args=(yonetici,), name="kaziyici", daemon=True).start()
    threading.Thread(target=ogrenme_dongusu, args=(yonetici,), name="ogrenme", daemon=True).start()
    llm = LLMIstemcisi(anahtar_sihirbazi())
    print("  ✔ Arka plan tarayıcısı aktif (10 dakikada bir günceller)")
    print(f"  ✔ LSTM modeli hazır (sürüm {meta.get('surum', 1)})")
    print("  ✔ Doğal dil arayüzü: " + (" → ".join(llm.saglayicilar) + " (ücretsiz katman)" if llm.aktif
                                       else "yerleşik anlama modülü (anahtar eklemek için: python main.py --anahtar)"))
    print("-" * 64)
    print("  Örnek: \"İstanbul'daki bilgisayar mühendisliği araştırma görevlisi ilanlarını bul\"")
    print("  'devam' → sonraki 5 ilan  |  'çıkış' → kapat")
    print("-" * 64)
    sohbet_dongusu(llm)
    DURDUR.set()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        DURDUR.set()
        print("\nAsistan: Görüşmek üzere! Başvurularında başarılar dilerim. 🌟")
    except Exception as _e:   # son güvenlik ağı: kullanıcı asla traceback görmez
        LOG.exception("Beklenmeyen hata: %s", _e)
        print("\nAsistan: Kusura bakma, beklenmedik bir aksaklık oldu ve kapanmam gerekiyor. "
              "Tekrar başlatırsan kaldığımız yerden devam edebiliriz. 🙏")

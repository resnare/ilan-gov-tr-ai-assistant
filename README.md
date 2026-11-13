# ilan-gov-tr-ai-assistant

**İlan.gov.tr için Doğal Dil Destekli ve Otonom Yapay Zekâ Asistanı**
(Natural-language driven autonomous AI assistant for Turkish public & academic job ads)

Üç ayaklı hibrit sistem, tek dosya: [`main.py`](main.py)

| Ayak | Teknoloji | Görev |
| --- | --- | --- |
| Otonom kazıyıcı | Selenium + BeautifulSoup + SQLite, `threading` | 10 dakikada bir *Kamu-Akademik Personel* kategorisini tarar, ilan tablolarını kadro satırlarına böler, `guncel_ilanlar.db`'ye yazar |
| Derin öğrenme beyni | PyTorch, çoklu görevli çift yönlü LSTM | Kadro metninden **iş tipi** (argor / ogrgor / ogruye / idari / ilan_disi) ve **alan** (6 sınıf) tahmini; yeni linklerden sürekli ince ayar |
| Doğal dil arayüzü | Google Gemini API → yedek Groq API (ikisi de ücretsiz katman) → kural tabanlı yedek | Cümle → JSON filtre; sonuçları 5'erli, linkli ve sıcak bir dille sunar |

## Kurallar
- **Katı alt filtre:** `spesifik_kelime` kadro metninde birebir (Türkçe karakter katlanmış, kök bazlı) aranır — "bilgisayar" arayan "inşaat" kadrosu görmez.
- **Veri ayrımı:** `data/egitim_verisi.csv` + `ogrenme_havuzu.db` yalnızca modeli besler; kullanıcıya yalnızca `guncel_ilanlar.db` linkleri gider.
- **7 gün kuralı:** yayın tarihi 7 günden eski ilan gösterilmez; bakım iş parçacığı silmeden önce öğrenme havuzuna arşivler.
- **Sıfır print hatası:** tüm hatalar `asistan.log`'a; kullanıcı yalnızca nazik asistan mesajları görür.

## Kurulum ve çalıştırma
```bash
pip install -r requirements.txt
python main.py --anahtar           # ücretsiz Gemini ve/veya Groq anahtarı gir (isteğe bağlı)
python main.py --veri-hazirla      # data/egitim_verisi.csv
python main.py --egit              # 5 katlı CV + nihai model, grafikler report_outputs/ altına
python main.py --tek-tarama        # tek kazıma turu (test)
python main.py                     # tam sistem: kazıyıcı + sürekli öğrenme + sohbet
```
Google Chrome kurulu olmalıdır (Selenium 4.6+ sürücüyü kendisi indirir).

## Ücretsiz LLM anahtarları
Koda hiçbir API anahtarı gömülmez; her kullanıcı kendi **ücretsiz** anahtarını kullanır, kimseden ücret alınmaz.

| Sağlayıcı | Nereden | Not |
| --- | --- | --- |
| Google Gemini (birincil) | https://aistudio.google.com → *Get API key* | Faturalandırma **açılmazsa** ücretsiz katmanda kalır |
| Groq (yedek) | https://console.groq.com/keys | Kart bilgisi istemez; `openai/gpt-oss-120b` (yedek: `gpt-oss-20b`) |

Anahtarlar ilk açılışta sorulur (Enter ile geçilebilir) ve yalnızca yerel `api_anahtarlari.json` dosyasına yazılır (`.gitignore`'da). Ortam değişkeni olarak da verilebilir: `GEMINI_API_KEY`, `GROQ_API_KEY`. Gemini kotası dolarsa asistan otomatik olarak Groq'a, ikisi de yoksa yerleşik anlama modülüne geçer.

## Sonuçlar (5 katlı grup çapraz doğrulama, 942 örnek)
| Model | İş tipi makro-F1 | Alan makro-F1 |
| --- | --- | --- |
| Çoklu görevli BiLSTM (PyTorch) | 0,859 ± 0,085 | 0,747 ± 0,111 |
| TF-IDF + Lojistik Regresyon | 0,885 ± 0,026 | 0,880 ± 0,074 |

Sabit test kümesinde (135 örnek) BiLSTM: iş tipi makro-F1 **0,921**, alan makro-F1 **0,783** (TF-IDF + LR: 0,898 / 0,797). `experiments/numpy_lstm.py`, aynı mimarinin geliştirme sırasında doğrulama için kullanılan NumPy kopyasıdır (gradyan kontrolü dahil).

## Rapor
Proje raporu: [`Project_Report.pdf`](Project_Report.pdf) · Şekiller ve metrikler: [`report_outputs/`](report_outputs)

## Veri
- `data/akademik_ilanlar_temiz.json` — 73 akademik ilan (ilan.gov.tr, 30.09–14.10.2026), tablolarıyla
- `data/ilan_veritabani.db` — eski kazıyıcının topladığı 505 kayıt (çoğu ihale; negatif sınıf)
- Kaynak: [ilan.gov.tr — Personel Alımı (iş ilanları)](https://www.ilan.gov.tr/ilan/tum-ilanlar/personel-alimi?ats=5) · [Kamu-Akademik Personel](https://www.ilan.gov.tr/ilan/kategori/8/kamu-akademik-personel?ats=5)

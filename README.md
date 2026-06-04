# 🧠 SentiAI — Analisis Sentimen Bahasa Indonesia (BiLSTM Deep Learning)

Aplikasi analisis sentimen teks bahasa Indonesia menggunakan arsitektur **Bidirectional LSTM** (Deep Learning) sebagai model utama, dengan fallback **Logistic Regression** jika TensorFlow tidak tersedia.

---

## 📁 Struktur Proyek

```
sentiment_lstm/
├── app.py              # Flask Backend API (LSTM + sklearn fallback)
├── streamlit_app.py    # Streamlit Frontend (UI v2.0)
├── train_lstm.py       # Script training standalone BiLSTM
├── requirements.txt    # Dependensi Python
├── data/
│   └── dataset.csv     # Dataset bahasa Indonesia (200+ contoh)
├── model_cache/        # Model tersimpan setelah training
│   ├── lstm_model.h5
│   ├── tokenizer.pkl
│   └── training_stats.json
└── README.md
```

---

## 🤖 Arsitektur Model BiLSTM

```
Input Text
    ↓
Preprocessing (lowercase, hapus URL/mention, stopwords)
    ↓
Tokenization + Padding (max_len=50)
    ↓
Embedding Layer (10000 vocab, 64 dim)
    ↓
SpatialDropout1D (0.2)
    ↓
BiLSTM Layer 1 (64 units, return_sequences=True)
    ↓
BiLSTM Layer 2 (32 units)
    ↓
Dense (64, ReLU)
    ↓
Dropout (0.4)
    ↓
Dense (3, Softmax) → [positif, negatif, netral]
```

---

## 🚀 Cara Menjalankan (Lokal)

### 1. Clone & Install

```bash
git clone https://github.com/<username>/sentiment_lstm.git
cd sentiment_lstm
pip install -r requirements.txt
```

### 2. (Opsional) Training model sebelum deploy

```bash
python train_lstm.py
```

> Tanpa ini, model akan dilatih otomatis saat pertama kali dijalankan.
> Jika TensorFlow tidak tersedia, otomatis fallback ke Logistic Regression.

### 3. Jalankan Flask API

```bash
python app.py
# → http://localhost:5000
```

### 4. Jalankan Streamlit UI (terminal terpisah)

```bash
streamlit run streamlit_app.py
# → http://localhost:8501
```

---

## 🌐 Deploy ke Streamlit Cloud

1. Push semua file ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Klik **New App** → pilih repo ini
4. Set **Main file**: `streamlit_app.py`
5. Klik **Deploy**

> ⚠️ Streamlit Cloud RAM terbatas. Jika TensorFlow gagal install, app otomatis menggunakan model Logistic Regression.

---

## 🔌 Endpoint API

| Method | Endpoint         | Deskripsi                         |
|--------|------------------|-----------------------------------|
| GET    | `/`              | Info API & status                 |
| GET    | `/health`        | Health check                      |
| GET    | `/stats`         | Statistik training & akurasi      |
| POST   | `/predict`       | Prediksi satu teks                |
| POST   | `/predict/batch` | Prediksi banyak teks (maks 100)   |
| POST   | `/train`         | Latih ulang model (force retrain) |

### Contoh Request `/predict`

```json
POST /predict
{
  "text": "Produk ini sangat bagus dan berkualitas tinggi!"
}
```

### Contoh Response

```json
{
  "success": true,
  "result": {
    "text_original": "Produk ini sangat bagus dan berkualitas tinggi!",
    "text_preprocessed": "produk bagus berkualitas tinggi",
    "prediction": "positif",
    "prediction_label": "Positif 😊",
    "confidence": {
      "positif": 94.2,
      "negatif": 2.1,
      "netral": 3.7
    },
    "model_used": "lstm"
  }
}
```

---

## 📊 Dataset

200+ teks bahasa Indonesia dengan 3 kelas:

| Label     | Jumlah | Deskripsi                                    |
|-----------|--------|----------------------------------------------|
| Positif 😊 | ~78   | Review positif, pujian, kepuasan             |
| Negatif 😞 | ~69   | Keluhan, kekecewaan, kritik                  |
| Netral 😐  | ~54   | Fakta, deskripsi biasa, tanpa emosi kuat     |

---

## 🛠️ Tech Stack

- **Backend**: Flask + Flask-CORS
- **Deep Learning**: TensorFlow/Keras — BiLSTM
- **Fallback ML**: scikit-learn — Logistic Regression
- **Frontend**: Streamlit + Plotly
- **NLP**: Custom tokenizer + stopwords Bahasa Indonesia

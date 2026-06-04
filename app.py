from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import os
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# ── Try importing Keras/TF (LSTM) ──────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, GlobalMaxPooling1D
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    KERAS_AVAILABLE = True
    print("[INFO] TensorFlow/Keras tersedia — LSTM aktif")
except ImportError:
    KERAS_AVAILABLE = False
    print("[WARNING] TensorFlow tidak ditemukan — menggunakan fallback sklearn")

# ── sklearn (fallback / tambahan) ──────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

app = Flask(__name__)
CORS(app)

# ─── Config ───────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE_DIR, 'data', 'dataset.csv')
CACHE_DIR    = os.path.join(BASE_DIR, 'model_cache')
LSTM_PATH    = os.path.join(CACHE_DIR, 'lstm_model.h5')
TOKENIZER_PATH = os.path.join(CACHE_DIR, 'tokenizer.pkl')
SKLEARN_PATH = os.path.join(CACHE_DIR, 'sklearn_model.pkl')
STATS_PATH   = os.path.join(CACHE_DIR, 'training_stats.json')

MAX_VOCAB   = 10000
MAX_LEN     = 50
EMBED_DIM   = 64
LSTM_UNITS  = 64
EPOCHS      = 30
BATCH_SIZE  = 32

LABEL_MAP = {
    'positif': 'Positif 😊',
    'negatif': 'Negatif 😞',
    'netral':  'Netral 😐'
}
CLASSES = ['positif', 'negatif', 'netral']

# ─── Stopwords ────────────────────────────────────────────────────────────────
STOPWORDS_ID = {
    'yang','dan','di','ke','dari','ini','itu','dengan','untuk','adalah','ada','atau',
    'juga','pada','dalam','saya','kami','kita','anda','mereka','dia','ia','nya','akan',
    'sudah','telah','tidak','bukan','jangan','tak','lebih','sangat','sekali','paling',
    'bisa','dapat','seperti','saat','serta','oleh','agar','tetapi','namun','karena',
    'jika','kalau','maka','lalu','kemudian','pun','lah','kah','deh','dong','sih','kok',
    'ya','yah','oh','eh','ah','ih','uh','wah','hah','hem','hmm','bang','mas','kak'
}

# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split() if t not in STOPWORDS_ID and len(t) > 2]
    return ' '.join(tokens)

# ─── Model Manager ────────────────────────────────────────────────────────────
class SentimentModelManager:
    def __init__(self):
        self.mode = None           # 'lstm' | 'sklearn'
        self.lstm_model   = None
        self.tokenizer    = None
        self.sklearn_model = None
        self.sklearn_vec   = None
        self.is_trained    = False
        self.stats         = {}

    # ─── LOAD ────────────────────────────────────────────────────────────────
    def load_cached(self):
        """Muat model dari cache (lebih cepat saat startup)"""
        loaded = False

        # Coba muat LSTM
        if KERAS_AVAILABLE and os.path.exists(LSTM_PATH) and os.path.exists(TOKENIZER_PATH):
            try:
                self.lstm_model = load_model(LSTM_PATH)
                with open(TOKENIZER_PATH, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                self.mode = 'lstm'
                loaded = True
                print("[INFO] LSTM model dimuat dari cache")
            except Exception as e:
                print(f"[WARNING] Gagal muat LSTM: {e}")

        # Coba muat sklearn
        if not loaded and os.path.exists(SKLEARN_PATH):
            try:
                with open(SKLEARN_PATH, 'rb') as f:
                    payload = pickle.load(f)
                self.sklearn_model = payload['model']
                self.sklearn_vec   = payload['vectorizer']
                self.mode = 'sklearn'
                loaded = True
                print("[INFO] Sklearn model dimuat dari cache")
            except Exception as e:
                print(f"[WARNING] Gagal muat sklearn: {e}")

        # Muat stats
        if os.path.exists(STATS_PATH):
            with open(STATS_PATH) as f:
                self.stats = json.load(f)

        if loaded:
            self.is_trained = True
        return loaded

    # ─── TRAIN LSTM ──────────────────────────────────────────────────────────
    def train_lstm(self, df):
        label2idx = {'positif': 0, 'negatif': 1, 'netral': 2}
        X = df['text_clean'].tolist()
        y = np.array([label2idx[l] for l in df['label']])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Tokenize
        self.tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(X_train)
        X_tr = pad_sequences(self.tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding='post')
        X_te = pad_sequences(self.tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN, padding='post')

        # Build BiLSTM
        model = Sequential([
            Embedding(MAX_VOCAB, EMBED_DIM, input_length=MAX_LEN),
            Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
            Bidirectional(LSTM(32, dropout=0.3)),
            Dense(64, activation='relu'),
            Dropout(0.4),
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]

        history = model.fit(
            X_tr, y_train,
            validation_data=(X_te, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate
        y_pred_proba = model.predict(X_te)
        y_pred = np.argmax(y_pred_proba, axis=1)
        idx2label = {v: k for k, v in label2idx.items()}
        y_test_labels = [idx2label[i] for i in y_test]
        y_pred_labels = [idx2label[i] for i in y_pred]

        acc = accuracy_score(y_test_labels, y_pred_labels)
        report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
        cm = confusion_matrix(y_test_labels, y_pred_labels, labels=CLASSES)

        # Save
        os.makedirs(CACHE_DIR, exist_ok=True)
        model.save(LSTM_PATH)
        with open(TOKENIZER_PATH, 'wb') as f:
            pickle.dump(self.tokenizer, f)

        self.lstm_model = model
        self.mode = 'lstm'

        hist_dict = {
            'accuracy': [float(v) for v in history.history['accuracy']],
            'val_accuracy': [float(v) for v in history.history['val_accuracy']],
            'loss': [float(v) for v in history.history['loss']],
            'val_loss': [float(v) for v in history.history['val_loss']],
        }
        return acc, report, cm, hist_dict

    # ─── TRAIN SKLEARN ───────────────────────────────────────────────────────
    def train_sklearn(self, df):
        X = df['text_clean']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.sklearn_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        X_tr = self.sklearn_vec.fit_transform(X_train)
        X_te = self.sklearn_vec.transform(X_test)

        self.sklearn_model = LogisticRegression(max_iter=1000, C=1.0)
        self.sklearn_model.fit(X_tr, y_train)
        y_pred = self.sklearn_model.predict(X_te)

        acc    = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm     = confusion_matrix(y_test, y_pred, labels=CLASSES)

        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(SKLEARN_PATH, 'wb') as f:
            pickle.dump({'model': self.sklearn_model, 'vectorizer': self.sklearn_vec,
                         'classes': self.sklearn_model.classes_.tolist()}, f)

        self.mode = 'sklearn'
        return acc, report, cm, {}

    # ─── TRAIN ───────────────────────────────────────────────────────────────
    def train(self, data_path=DATA_PATH, force=False):
        if not force and self.load_cached():
            return True, f"Model dimuat dari cache (mode: {self.mode})"

        try:
            df = pd.read_csv(data_path).dropna()
            df['text_clean'] = df['text'].apply(preprocess_text)
            df = df[df['text_clean'].str.len() > 0]

            dist = df['label'].value_counts().to_dict()
            total = len(df)

            if KERAS_AVAILABLE:
                acc, report, cm, history = self.train_lstm(df)
                model_name = "BiLSTM (Deep Learning)"
            else:
                acc, report, cm, history = self.train_sklearn(df)
                model_name = "Logistic Regression (Fallback)"

            X_tr_size = int(total * 0.8)
            self.stats = {
                'model_type': self.mode,
                'model_name': model_name,
                'accuracy': round(acc * 100, 2),
                'report': report,
                'confusion_matrix': cm.tolist(),
                'labels': CLASSES,
                'dataset': {
                    'total': total,
                    'train_size': X_tr_size,
                    'test_size': total - X_tr_size,
                    'distribution': dist
                },
                'history': history
            }
            self.is_trained = True

            with open(STATS_PATH, 'w') as f:
                json.dump(self.stats, f, indent=2)

            return True, f"{model_name} berhasil dilatih! Akurasi: {acc*100:.2f}%"
        except Exception as e:
            import traceback
            return False, f"Error training: {str(e)}\n{traceback.format_exc()}"

    # ─── PREDICT ─────────────────────────────────────────────────────────────
    def predict(self, text):
        if not self.is_trained:
            return None, "Model belum dilatih"

        text_clean = preprocess_text(text)
        if not text_clean:
            return None, "Teks tidak valid setelah preprocessing"

        if self.mode == 'lstm' and self.lstm_model:
            seq = pad_sequences(
                self.tokenizer.texts_to_sequences([text_clean]),
                maxlen=MAX_LEN, padding='post'
            )
            proba = self.lstm_model.predict(seq, verbose=0)[0]
            pred_idx = int(np.argmax(proba))
            prediction = CLASSES[pred_idx]
            confidence = {CLASSES[i]: round(float(p) * 100, 2) for i, p in enumerate(proba)}
        else:
            vec = self.sklearn_vec.transform([text_clean])
            prediction = self.sklearn_model.predict(vec)[0]
            proba_raw = self.sklearn_model.predict_proba(vec)[0]
            classes   = self.sklearn_model.classes_
            confidence = {c: round(float(p) * 100, 2) for c, p in zip(classes, proba_raw)}

        return {
            'text_original':     text,
            'text_preprocessed': text_clean,
            'prediction':        prediction,
            'prediction_label':  LABEL_MAP.get(prediction, prediction),
            'confidence':        confidence,
            'model_used':        self.mode,
        }, None

    def predict_batch(self, texts):
        results = []
        for text in texts:
            result, err = self.predict(text)
            results.append(result if not err else {'text': text, 'error': err})
        return results


# ─── Init ─────────────────────────────────────────────────────────────────────
manager = SentimentModelManager()

@app.before_request
def initialize():
    if not manager.is_trained:
        ok, msg = manager.train(DATA_PATH)
        if not ok:
            print(f"[ERROR] Auto-train gagal: {msg}")

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return jsonify({
        'status': 'running',
        'message': 'Sentiment Analysis API — BiLSTM Deep Learning',
        'version': '2.0.0',
        'model_mode': manager.mode,
        'model_trained': manager.is_trained,
        'endpoints': {
            'POST /predict':       'Prediksi satu teks',
            'POST /predict/batch': 'Prediksi banyak teks',
            'GET  /stats':         'Statistik training model',
            'GET  /health':        'Health check',
            'POST /train':         'Latih ulang model (force)',
        }
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_trained': manager.is_trained, 'mode': manager.mode})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Field "text" dibutuhkan'}), 400
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'Teks tidak boleh kosong'}), 400
    result, error = manager.predict(text)
    if error:
        return jsonify({'error': error}), 500
    return jsonify({'success': True, 'result': result})

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    data = request.get_json()
    if not data or 'texts' not in data:
        return jsonify({'error': 'Field "texts" (array) dibutuhkan'}), 400
    texts = data.get('texts', [])
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({'error': 'texts harus array tidak kosong'}), 400
    if len(texts) > 100:
        return jsonify({'error': 'Maksimal 100 teks per request'}), 400
    results  = manager.predict_batch(texts)
    sentiments = [r.get('prediction') for r in results if 'prediction' in r]
    summary = {s: sentiments.count(s) for s in CLASSES}
    summary['total'] = len(results)
    return jsonify({'success': True, 'results': results, 'summary': summary})

@app.route('/stats')
def stats():
    if not manager.is_trained:
        return jsonify({'error': 'Model belum dilatih'}), 503
    return jsonify({'success': True, 'stats': manager.stats})

@app.route('/train', methods=['POST'])
def train():
    ok, msg = manager.train(DATA_PATH, force=True)
    if ok:
        return jsonify({'success': True, 'message': msg, 'stats': manager.stats})
    return jsonify({'success': False, 'error': msg}), 500

if __name__ == '__main__':
    print("🚀 Starting Sentiment Analysis API v2.0 (LSTM)...")
    ok, msg = manager.train(DATA_PATH)
    print(f"✅ {msg}" if ok else f"❌ {msg}")
    app.run(debug=True, host='0.0.0.0', port=5000)

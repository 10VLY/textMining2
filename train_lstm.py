"""
train_lstm.py — Script training standalone untuk model BiLSTM
Jalankan sekali sebelum deploy: python train_lstm.py
"""

import pandas as pd
import numpy as np
import pickle
import json
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ── TensorFlow / Keras ────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Embedding, LSTM, Bidirectional,
                                         Dense, Dropout, SpatialDropout1D)
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    print(f"✅ TensorFlow {tf.__version__} ditemukan")
except ImportError:
    print("❌ TensorFlow tidak ditemukan. Install: pip install tensorflow")
    exit(1)

# ─── Config ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'dataset.csv')
CACHE_DIR   = os.path.join(BASE_DIR, 'model_cache')

MAX_VOCAB   = 10000
MAX_LEN     = 50
EMBED_DIM   = 64
LSTM_UNITS  = 64
EPOCHS      = 50
BATCH_SIZE  = 32
TEST_SPLIT  = 0.2
CLASSES     = ['positif', 'negatif', 'netral']

os.makedirs(CACHE_DIR, exist_ok=True)

# ─── Stopwords ────────────────────────────────────────────────────────────────
STOPWORDS_ID = {
    'yang','dan','di','ke','dari','ini','itu','dengan','untuk','adalah','ada','atau',
    'juga','pada','dalam','saya','kami','kita','anda','mereka','dia','ia','nya','akan',
    'sudah','telah','tidak','bukan','jangan','tak','lebih','sangat','sekali','paling',
    'bisa','dapat','seperti','saat','serta','oleh','agar','tetapi','namun','karena',
    'jika','kalau','maka','lalu','kemudian','pun','lah','kah','deh','dong','sih','kok',
    'ya','yah','oh','eh','ah','ih','uh','wah','hah','hem','hmm','bang','mas','kak'
}

def preprocess(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join([t for t in text.split() if t not in STOPWORDS_ID and len(t) > 2])

# ─── Load & Preprocess ────────────────────────────────────────────────────────
print("\n📂 Memuat dataset...")
df = pd.read_csv(DATA_PATH).dropna()
df['text_clean'] = df['text'].apply(preprocess)
df = df[df['text_clean'].str.len() > 0]

print(f"   Total: {len(df)} sampel")
print(f"   Distribusi: {df['label'].value_counts().to_dict()}")

label2idx = {c: i for i, c in enumerate(CLASSES)}
idx2label = {i: c for c, i in label2idx.items()}

X = df['text_clean'].tolist()
y = np.array([label2idx[l] for l in df['label']])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=42, stratify=y
)
print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# ─── Tokenize ─────────────────────────────────────────────────────────────────
print("\n🔤 Tokenisasi teks...")
tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_tr = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding='post')
X_te = pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen=MAX_LEN, padding='post')
print(f"   Vocab size: {min(len(tokenizer.word_index)+1, MAX_VOCAB)}")

# ─── Build Model ──────────────────────────────────────────────────────────────
print("\n🏗️ Membangun arsitektur BiLSTM...")
model = Sequential([
    Embedding(MAX_VOCAB, EMBED_DIM, input_length=MAX_LEN),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)),
    Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.2)),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(3, activation='softmax')
], name='BiLSTM_Sentiment')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ─── Train ────────────────────────────────────────────────────────────────────
print("\n🚀 Mulai training...")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    ModelCheckpoint(os.path.join(CACHE_DIR, 'lstm_best.h5'),
                    monitor='val_accuracy', save_best_only=True, verbose=0),
]

history = model.fit(
    X_tr, y_train,
    validation_data=(X_te, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ─── Evaluate ─────────────────────────────────────────────────────────────────
print("\n📊 Evaluasi model...")
y_pred_proba = model.predict(X_te, verbose=0)
y_pred       = np.argmax(y_pred_proba, axis=1)
y_test_lbl   = [idx2label[i] for i in y_test]
y_pred_lbl   = [idx2label[i] for i in y_pred]

acc    = accuracy_score(y_test_lbl, y_pred_lbl)
report = classification_report(y_test_lbl, y_pred_lbl, output_dict=True)
cm     = confusion_matrix(y_test_lbl, y_pred_lbl, labels=CLASSES)

print(f"\n✅ Akurasi: {acc*100:.2f}%")
print(classification_report(y_test_lbl, y_pred_lbl))
print("Confusion Matrix:")
print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES))

# ─── Save ─────────────────────────────────────────────────────────────────────
print("\n💾 Menyimpan model...")
model.save(os.path.join(CACHE_DIR, 'lstm_model.h5'))
with open(os.path.join(CACHE_DIR, 'tokenizer.pkl'), 'wb') as f:
    pickle.dump(tokenizer, f)

hist_dict = {
    'accuracy':     [float(v) for v in history.history['accuracy']],
    'val_accuracy': [float(v) for v in history.history['val_accuracy']],
    'loss':         [float(v) for v in history.history['loss']],
    'val_loss':     [float(v) for v in history.history['val_loss']],
}
stats = {
    'model_type': 'lstm',
    'model_name': 'BiLSTM (Deep Learning)',
    'accuracy':   round(acc * 100, 2),
    'report':     report,
    'confusion_matrix': cm.tolist(),
    'labels':     CLASSES,
    'dataset': {
        'total':      len(df),
        'train_size': len(X_train),
        'test_size':  len(X_test),
        'distribution': df['label'].value_counts().to_dict()
    },
    'history': hist_dict,
    'architecture': {
        'max_vocab':  MAX_VOCAB,
        'max_len':    MAX_LEN,
        'embed_dim':  EMBED_DIM,
        'lstm_units': LSTM_UNITS,
        'epochs_run': len(history.history['accuracy']),
    }
}
with open(os.path.join(CACHE_DIR, 'training_stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)

print("\n🎉 Training selesai!")
print(f"   📁 Model: model_cache/lstm_model.h5")
print(f"   📁 Tokenizer: model_cache/tokenizer.pkl")
print(f"   📁 Stats: model_cache/training_stats.json")
print(f"   🎯 Akurasi final: {acc*100:.2f}%")

"""
Sentiment Analysis — BiLSTM Deep Learning
Streamlit UI v2.0
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import threading
import time
import os
import sys

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentiAI — Analisis Sentimen",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Cards */
.metric-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform .2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-card .metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #a78bfa;
    line-height: 1.1;
}
.metric-card .metric-label {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-top: 4px;
}

/* Result card */
.result-card {
    border-radius: 20px;
    padding: 28px 32px;
    margin: 16px 0;
    border: 1px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(12px);
}
.result-positif { background: linear-gradient(135deg,rgba(52,211,153,.15),rgba(16,185,129,.08)); border-color: rgba(52,211,153,.35); }
.result-negatif { background: linear-gradient(135deg,rgba(239,68,68,.15),rgba(220,38,38,.08)); border-color: rgba(239,68,68,.35); }
.result-netral  { background: linear-gradient(135deg,rgba(251,191,36,.15),rgba(245,158,11,.08)); border-color: rgba(251,191,36,.35); }

.result-emoji { font-size: 3.5rem; }
.result-label {
    font-size: 1.8rem;
    font-weight: 700;
    color: #fff;
    margin-top: 6px;
}
.result-confidence {
    font-size: 0.9rem;
    color: rgba(255,255,255,.6);
    margin-top: 4px;
}

/* Hero */
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    line-height: 1.15;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: rgba(255,255,255,.55);
    text-align: center;
    margin-top: 8px;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}
.badge-lstm  { background:rgba(167,139,250,.25); color:#c4b5fd; border:1px solid rgba(167,139,250,.4); }
.badge-info  { background:rgba(96,165,250,.2);   color:#93c5fd; border:1px solid rgba(96,165,250,.3); }
.badge-green { background:rgba(52,211,153,.2);   color:#6ee7b7; border:1px solid rgba(52,211,153,.3); }

/* Input */
.stTextArea textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-size: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: rgba(167,139,250,.6) !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,.15) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: all .2s !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(124,58,237,.4) !important;
}

/* Section header */
.section-header {
    font-size: 1.15rem;
    font-weight: 600;
    color: rgba(255,255,255,.85);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Divider */
hr { border-color: rgba(255,255,255,.1) !important; }

/* Tab */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,.05);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: rgba(255,255,255,.55) !important;
    border-radius: 8px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(167,139,250,.25) !important;
    color: #c4b5fd !important;
}

/* Plotly bg */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ─── Flask backend ────────────────────────────────────────────────────────────
API_URL = "http://localhost:5000"

def _run_flask():
    """Jalankan Flask di background thread"""
    import subprocess, sys
    subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(__file__), "app.py")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

def ensure_api():
    """Pastikan API berjalan, jalankan jika belum"""
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        t = threading.Thread(target=_run_flask, daemon=True)
        t.start()
        for _ in range(20):
            time.sleep(0.6)
            try:
                requests.get(f"{API_URL}/health", timeout=1)
                return True
            except Exception:
                pass
        return False

# ─── Session state ────────────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []
if 'stats' not in st.session_state:
    st.session_state.stats = None

# ─── Helpers ──────────────────────────────────────────────────────────────────
def fetch_stats():
    try:
        r = requests.get(f"{API_URL}/stats", timeout=5)
        if r.status_code == 200:
            return r.json().get('stats', {})
    except Exception:
        pass
    return {}

def call_predict(text):
    try:
        r = requests.post(f"{API_URL}/predict", json={'text': text}, timeout=10)
        if r.status_code == 200:
            return r.json().get('result'), None
        return None, r.json().get('error', 'Unknown error')
    except Exception as e:
        return None, str(e)

def call_batch(texts):
    try:
        r = requests.post(f"{API_URL}/predict/batch", json={'texts': texts}, timeout=15)
        if r.status_code == 200:
            return r.json(), None
        return None, r.json().get('error', 'Unknown error')
    except Exception as e:
        return None, str(e)

SENTIMENT_COLORS = {
    'positif': '#34d399',
    'negatif': '#f87171',
    'netral':  '#fbbf24'
}

def confidence_chart(confidence: dict):
    labels = list(confidence.keys())
    values = list(confidence.values())
    colors = [SENTIMENT_COLORS.get(l, '#a78bfa') for l in labels]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.1f}%" for v in values],
        textposition='outside',
        textfont=dict(color='white', size=13, family='Inter'),
    ))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Inter'),
        yaxis=dict(range=[0, 110], showgrid=False, zeroline=False, showticklabels=False),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=14)),
        margin=dict(l=0, r=0, t=20, b=0),
        height=200,
        showlegend=False,
    )
    return fig

def donut_chart(dist: dict):
    labels = list(dist.keys())
    values = list(dist.values())
    colors = [SENTIMENT_COLORS.get(l, '#a78bfa') for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.6,
        marker=dict(colors=colors, line=dict(width=0)),
        textinfo='percent',
        textfont=dict(size=12, color='white'),
        hovertemplate='%{label}: %{value} sampel (%{percent})<extra></extra>'
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Inter'),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, font=dict(color='white')),
        margin=dict(l=0, r=0, t=0, b=0),
        height=220,
        annotations=[dict(text=f"{sum(values)}<br>sampel", x=0.5, y=0.5,
                          font_size=15, showarrow=False, font_color='white')]
    )
    return fig

def confusion_heatmap(cm, labels):
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0,'rgba(30,20,60,.2)'],[1,'rgba(167,139,250,1)']],
        showscale=False,
        text=[[str(v) for v in row] for row in cm],
        texttemplate='%{text}',
        textfont=dict(size=18, color='white'),
        hovertemplate='Aktual %{y} → Prediksi %{x}: %{z}<extra></extra>'
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Inter', size=12),
        xaxis=dict(title='Prediksi', tickfont=dict(size=12)),
        yaxis=dict(title='Aktual', tickfont=dict(size=12)),
        margin=dict(l=0, r=0, t=10, b=0),
        height=260,
    )
    return fig

def training_history_chart(history):
    epochs = list(range(1, len(history['accuracy']) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history['accuracy'],   name='Train Acc',   line=dict(color='#a78bfa', width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=history['val_accuracy'], name='Val Acc', line=dict(color='#34d399', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=epochs, y=history['loss'],       name='Train Loss',  line=dict(color='#f87171', width=2), yaxis='y2'))
    fig.add_trace(go.Scatter(x=epochs, y=history['val_loss'],   name='Val Loss',    line=dict(color='#fbbf24', width=2, dash='dash'), yaxis='y2'))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Inter', size=11),
        xaxis=dict(title='Epoch', showgrid=False),
        yaxis=dict(title='Accuracy', showgrid=True, gridcolor='rgba(255,255,255,.07)'),
        yaxis2=dict(title='Loss', overlaying='y', side='right', showgrid=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(color='white', size=10)),
        margin=dict(l=0, r=0, t=30, b=0), height=260,
    )
    return fig

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 8px">
        <div style="font-size:2.5rem">🧠</div>
        <div style="font-size:1.2rem;font-weight:700;color:#c4b5fd">SentiAI</div>
        <div style="font-size:.75rem;color:rgba(255,255,255,.4);margin-top:4px">Deep Learning Sentiment v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # API Status
    api_ok = ensure_api()
    status_icon  = "🟢" if api_ok else "🔴"
    status_label = "API Online" if api_ok else "API Offline"
    st.markdown(f"**{status_icon} {status_label}**")

    if api_ok and st.session_state.stats is None:
        st.session_state.stats = fetch_stats()

    stats = st.session_state.stats or {}

    st.divider()

    # Model info
    mode = stats.get('model_type', 'N/A')
    acc  = stats.get('accuracy', 0)
    st.markdown(f"""
    <div class="metric-card" style="margin-bottom:12px">
        <div class="metric-value">{acc:.1f}%</div>
        <div class="metric-label">Akurasi Model</div>
    </div>
    """, unsafe_allow_html=True)

    badge_class = 'badge-lstm' if mode == 'lstm' else 'badge-info'
    mode_label  = 'BiLSTM' if mode == 'lstm' else 'Logistic Regression'
    st.markdown(f'<span class="badge {badge_class}">🔬 {mode_label}</span>', unsafe_allow_html=True)

    ds = stats.get('dataset', {})
    if ds:
        st.markdown(f"""
        <div style="margin-top:12px;font-size:.82rem;color:rgba(255,255,255,.5)">
        📊 Dataset: {ds.get('total',0)} sampel<br>
        🏋️ Train: {ds.get('train_size',0)} &nbsp;|&nbsp; 🧪 Test: {ds.get('test_size',0)}
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    if st.button("🔄 Latih Ulang Model"):
        with st.spinner("Training..."):
            try:
                r = requests.post(f"{API_URL}/train", timeout=120)
                if r.status_code == 200:
                    st.success("✅ Training selesai!")
                    st.session_state.stats = fetch_stats()
                    st.rerun()
                else:
                    st.error("Training gagal")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.history:
        st.divider()
        st.markdown("**📝 Riwayat Prediksi**")
        for i, h in enumerate(reversed(st.session_state.history[-5:])):
            emoji_map = {'positif': '😊', 'negatif': '😞', 'netral': '😐'}
            emoji = emoji_map.get(h.get('prediction',''), '❓')
            st.markdown(f"""
            <div style="background:rgba(255,255,255,.05);border-radius:8px;padding:8px 10px;margin-bottom:6px;font-size:.8rem;">
                <span style="font-size:1.1rem">{emoji}</span>
                <span style="color:rgba(255,255,255,.7);margin-left:6px">{h['text'][:40]}{'...' if len(h['text'])>40 else ''}</span>
            </div>
            """, unsafe_allow_html=True)

# ─── Main Content ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:32px 0 20px">
    <div class="hero-title">🧠 Analisis Sentimen</div>
    <div class="hero-subtitle">Bahasa Indonesia · BiLSTM Deep Learning · Real-time Prediction</div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["💬 Prediksi Teks", "📦 Analisis Batch", "📊 Dashboard Model"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Single Predict
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    col_main, col_info = st.columns([3, 2], gap="large")

    with col_main:
        st.markdown('<div class="section-header">✍️ Masukkan Teks</div>', unsafe_allow_html=True)
        text_input = st.text_area(
            label="",
            placeholder="Contoh: Produk ini sangat bagus dan kualitasnya sangat memuaskan saya...",
            height=160, key="single_input", label_visibility="collapsed"
        )

        # Quick examples
        st.markdown('<div style="font-size:.8rem;color:rgba(255,255,255,.45);margin-bottom:6px">💡 Contoh cepat:</div>', unsafe_allow_html=True)
        ex_cols = st.columns(3)
        examples = [
            ("😊 Positif", "Produk ini sangat bagus kualitas terbaik harga terjangkau!"),
            ("😞 Negatif", "Sangat kecewa barang rusak tidak sesuai deskripsi sama sekali."),
            ("😐 Netral",  "Barang sudah sampai kondisi normal sesuai estimasi pengiriman."),
        ]
        for i, (label, ex) in enumerate(examples):
            with ex_cols[i]:
                if st.button(label, key=f"ex_{i}"):
                    st.session_state.single_input = ex
                    st.rerun()

        predict_btn = st.button("🔍 Analisis Sentimen", key="predict_btn")

    with col_info:
        st.markdown('<div class="section-header">📌 Panduan Sentimen</div>', unsafe_allow_html=True)
        guide = [
            ("😊", "Positif", "#34d399", "Pujian, kepuasan, rekomendasi, review bagus"),
            ("😞", "Negatif", "#f87171", "Keluhan, kekecewaan, kritik, review buruk"),
            ("😐", "Netral",  "#fbbf24", "Fakta, deskripsi biasa, tanpa emosi kuat"),
        ]
        for emoji, label, color, desc in guide:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,.05);border-left:3px solid {color};
                        border-radius:0 10px 10px 0;padding:10px 14px;margin-bottom:8px">
                <div style="font-size:1.2rem;font-weight:600;color:{color}">{emoji} {label}</div>
                <div style="font-size:.8rem;color:rgba(255,255,255,.5);margin-top:3px">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Hasil prediksi
    if predict_btn and text_input.strip():
        with st.spinner("Menganalisis..."):
            result, err = call_predict(text_input.strip())

        if err:
            st.error(f"❌ Error: {err}")
        elif result:
            pred = result['prediction']
            label = result['prediction_label']
            conf  = result.get('confidence', {})
            top_conf = conf.get(pred, 0)

            card_class = f"result-{pred}"
            emoji_map  = {'positif': '😊', 'negatif': '😞', 'netral': '😐'}
            emoji       = emoji_map.get(pred, '❓')

            st.markdown(f"""
            <div class="result-card {card_class}">
                <div class="result-emoji">{emoji}</div>
                <div class="result-label">{label}</div>
                <div class="result-confidence">Keyakinan model: {top_conf:.1f}%</div>
                <div style="margin-top:12px;font-size:.85rem;color:rgba(255,255,255,.5)">
                    Teks: <em>"{result['text_original'][:80]}{'...' if len(result['text_original'])>80 else ''}"</em>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence chart
            if conf:
                st.markdown('<div class="section-header" style="margin-top:16px">📊 Distribusi Keyakinan</div>', unsafe_allow_html=True)
                st.plotly_chart(confidence_chart(conf), use_container_width=True, config={'displayModeBar': False})

            # Add to history
            st.session_state.history.append({
                'text': result['text_original'],
                'prediction': pred,
                'confidence': top_conf
            })

    elif predict_btn:
        st.warning("⚠️ Masukkan teks terlebih dahulu")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Batch Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="section-header">📦 Analisis Banyak Teks Sekaligus</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2], gap="large")

    with c1:
        batch_input = st.text_area(
            "Masukkan teks (satu per baris, maks 50):",
            height=200,
            placeholder="Produk bagus sekali!\nBarang rusak kecewa banget.\nSudah sampai kondisi normal."
        )
        upload_file = st.file_uploader("Atau upload CSV (kolom 'text')", type=['csv'])

        analyze_btn = st.button("⚡ Analisis Semua", key="batch_btn")

    with c2:
        st.markdown("""
        <div style="background:rgba(255,255,255,.05);border-radius:12px;padding:16px;font-size:.85rem;color:rgba(255,255,255,.6)">
        <b style="color:rgba(255,255,255,.8)">📋 Format Input</b><br><br>
        • Satu baris = satu teks<br>
        • Maksimal 50 teks sekaligus<br>
        • Atau upload file CSV dengan kolom <code>text</code><br><br>
        <b style="color:rgba(255,255,255,.8)">📤 Output</b><br><br>
        • Label sentimen per teks<br>
        • Ringkasan statistik<br>
        • Visualisasi distribusi<br>
        • Download hasil sebagai CSV
        </div>
        """, unsafe_allow_html=True)

    if analyze_btn:
        texts = []
        if upload_file:
            try:
                df_upload = pd.read_csv(upload_file)
                if 'text' in df_upload.columns:
                    texts = df_upload['text'].dropna().tolist()[:50]
                else:
                    st.error("CSV harus memiliki kolom 'text'")
            except Exception as e:
                st.error(f"Error membaca CSV: {e}")
        elif batch_input.strip():
            texts = [t.strip() for t in batch_input.strip().split('\n') if t.strip()][:50]

        if texts:
            with st.spinner(f"Menganalisis {len(texts)} teks..."):
                resp, err = call_batch(texts)

            if err:
                st.error(f"❌ {err}")
            elif resp:
                results  = resp['results']
                summary  = resp['summary']

                st.divider()

                # Summary metrics
                mc = st.columns(4)
                metrics = [
                    ("Total",    summary.get('total',    0), "📊"),
                    ("Positif",  summary.get('positif',  0), "😊"),
                    ("Negatif",  summary.get('negatif',  0), "😞"),
                    ("Netral",   summary.get('netral',   0), "😐"),
                ]
                for col, (label, val, icon) in zip(mc, metrics):
                    with col:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{icon} {val}</div>
                            <div class="metric-label">{label}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # Donut chart
                dist = {k: summary[k] for k in ['positif','negatif','netral'] if summary.get(k,0)>0}
                if dist:
                    st.plotly_chart(donut_chart(dist), use_container_width=True, config={'displayModeBar': False})

                # Table
                st.markdown('<div class="section-header" style="margin-top:8px">📋 Detail Hasil</div>', unsafe_allow_html=True)
                rows = []
                for r in results:
                    if 'error' in r:
                        rows.append({'Teks': r.get('text',''), 'Sentimen': '❓ Error', 'Keyakinan': '-'})
                    else:
                        emoji_map = {'positif':'😊','negatif':'😞','netral':'😐'}
                        pred = r.get('prediction','')
                        conf = r.get('confidence',{}).get(pred, 0)
                        rows.append({
                            'Teks':      r.get('text_original','')[:80],
                            'Sentimen':  f"{emoji_map.get(pred,'❓')} {r.get('prediction_label','')}",
                            'Keyakinan': f"{conf:.1f}%"
                        })

                df_result = pd.DataFrame(rows)
                st.dataframe(df_result, use_container_width=True, height=300)

                # Download
                csv_out = df_result.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Hasil CSV", csv_out, "hasil_sentimen.csv", "text/csv")
        else:
            st.warning("⚠️ Masukkan teks atau upload file CSV")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Dashboard
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    stats = st.session_state.stats
    if not stats:
        if api_ok:
            st.session_state.stats = fetch_stats()
            stats = st.session_state.stats

    if not stats:
        st.info("📊 Model belum dilatih. Klik 'Latih Ulang Model' di sidebar.")
    else:
        # Header metrics
        mc = st.columns(4)
        items = [
            ("Akurasi",     f"{stats.get('accuracy',0):.1f}%",                "🎯"),
            ("Total Data",  str(stats.get('dataset',{}).get('total',0)),       "📦"),
            ("Model",       "BiLSTM" if stats.get('model_type')=='lstm' else "LR", "🤖"),
            ("Kelas",       "3",                                                "🏷️"),
        ]
        for col, (label, val, icon) in zip(mc, items):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{icon} {val}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        row1 = st.columns(2, gap="large")

        # Dataset distribution
        with row1[0]:
            st.markdown('<div class="section-header">🗂️ Distribusi Dataset</div>', unsafe_allow_html=True)
            dist = stats.get('dataset', {}).get('distribution', {})
            if dist:
                st.plotly_chart(donut_chart(dist), use_container_width=True, config={'displayModeBar': False})

        # Confusion matrix
        with row1[1]:
            st.markdown('<div class="section-header">🔀 Confusion Matrix</div>', unsafe_allow_html=True)
            cm     = stats.get('confusion_matrix')
            labels = stats.get('labels', ['positif','negatif','netral'])
            if cm:
                st.plotly_chart(confusion_heatmap(cm, labels), use_container_width=True, config={'displayModeBar': False})

        # Training history (LSTM only)
        history = stats.get('history', {})
        if history and history.get('accuracy'):
            st.markdown('<div class="section-header">📈 Training History (LSTM)</div>', unsafe_allow_html=True)
            st.plotly_chart(training_history_chart(history), use_container_width=True, config={'displayModeBar': False})

        # Per-class metrics
        report = stats.get('report', {})
        if report:
            st.markdown('<div class="section-header">📋 Metrik per Kelas</div>', unsafe_allow_html=True)
            rows = []
            for cls in ['positif','negatif','netral']:
                if cls in report:
                    m = report[cls]
                    rows.append({
                        'Kelas':     cls.capitalize(),
                        'Precision': f"{m.get('precision',0)*100:.1f}%",
                        'Recall':    f"{m.get('recall',0)*100:.1f}%",
                        'F1-Score':  f"{m.get('f1-score',0)*100:.1f}%",
                        'Support':   int(m.get('support',0)),
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:rgba(255,255,255,.25);font-size:.78rem;padding:8px 0">
    SentiAI v2.0 · BiLSTM Deep Learning · Analisis Sentimen Bahasa Indonesia
</div>
""", unsafe_allow_html=True)

"""
app.py — TomatoCare: AI-Powered Tomato Leaf Disease Detection
Aplikasi Streamlit untuk klasifikasi penyakit daun tomat menggunakan
model ConvNeXt Transfer Learning (Keras 3 / TensorFlow).
"""

import logging
import datetime
import io
import base64
from typing import Optional
import streamlit.components.v1 as components

import streamlit as st
import numpy as np
from PIL import Image

# ──────────────────────────────────────────────────────────
# Konfigurasi logging (developer-level, tidak tampil ke user)
# ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Konfigurasi halaman Streamlit 
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TomatoCare — Deteksi Penyakit Daun Tomat",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "TomatoCare v1.0 — AI-assisted tomato leaf disease detection. For educational purposes."
    },
)

# ──────────────────────────────────────────────────────────
# Import internal (setelah set_page_config)
# ──────────────────────────────────────────────────────────
from utils.model_loader import load_model, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES
from utils.preprocessing import (
    validate_image_file,
    preprocess_image,
    load_pil_image,
    get_image_info,
    get_image_quality_warnings,
)
from utils.prediction import predict, format_prediction, get_confidence_color
from utils.disease_info import CLASS_NAMES, get_disease_info, get_display_name, DISEASE_INFO

# ──────────────────────────────────────────────────────────
# Custom CSS — Design System
# ──────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root Variables ── */
:root {
    --primary:       #1a4731;
    --primary-light: #2d6a4f;
    --secondary:     #40916c;
    --accent:        #f4a261;
    --accent-dark:   #e76f51;
    --bg-main:       #f8faf8;
    --bg-card:       #ffffff;
    --bg-sidebar:    #1a2e22;
    --text-primary:  #1c2b1e;
    --text-secondary:#4a5e4e;
    --text-muted:    #7a8f7e;
    --border:        #d8e8dc;
    --border-light:  #edf5ef;
    --success:       #2d6a4f;
    --warning:       #d4812e;
    --danger:        #b03a2e;
    --shadow-sm:     0 1px 3px rgba(0,0,0,0.08);
    --shadow-md:     0 4px 16px rgba(0,0,0,0.10);
    --shadow-lg:     0 8px 32px rgba(0,0,0,0.12);
    --radius-sm:     8px;
    --radius-md:     14px;
    --radius-lg:     20px;
    --radius-xl:     28px;
}

/* ── Base Reset ── */
html, body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}

/* ── Background with subtle dot pattern ── */
[data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #eef4ef;
    background-image:
        radial-gradient(circle, rgba(45,106,79,0.12) 1px, transparent 1px);
    background-size: 28px 28px;
    color: var(--text-primary);
}

/* ── Main block overlay for readability ── */
[data-testid="stMainBlockContainer"] {
    background: rgba(238,244,239,0.55);
}

/* ── Fix File Uploader text visibility ── */
[data-testid="stFileUploaderDropzone"] {
    background: var(--bg-card) !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stFileUploaderDropzone"] * {
    color: var(--text-primary) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] div,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: var(--text-secondary) !important;
}
/* Uploaded file row */
[data-testid="stFileUploaderFile"] {
    background: var(--bg-main) !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stFileUploaderFile"] * {
    color: var(--text-primary) !important;
}
/* Delete (X) button in uploader */
[data-testid="stFileUploaderDeleteBtn"] button {
    background: transparent !important;
    color: var(--text-muted) !important;
    border: none !important;
    box-shadow: none !important;
    transform: none !important;
}
[data-testid="stFileUploaderDeleteBtn"] button:hover {
    color: var(--danger) !important;
    transform: none !important;
}

/* ── Hide Streamlit Chrome (only branding, keep sidebar controls) ── */
#MainMenu, footer { display: none !important; }
/* Hide Streamlit toolbar but keep sidebar toggle intact */
[data-testid="stToolbar"] { display: none !important; }
/* Make header transparent so sidebar control still works */
[data-testid="stHeader"] {
    background: transparent !important;
    height: 2rem !important;
}
/* Only hide Streamlit branding inside header */
[data-testid="stHeader"] > div:first-child {
    opacity: 0 !important;
    pointer-events: none !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1a2e22 !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.9) !important; }
[data-testid="stSidebar"] .stRadio label { color: rgba(255,255,255,0.75) !important; }
[data-testid="stSidebar"] .stRadio [aria-checked="true"] + div { color: #ffffff !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }

/* ── Hide sidebar COLLAPSE button (so sidebar can't be closed accidentally) ── */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] * {
    display: none !important;
}

/* ── Reopen button styling (fallback if sidebar already collapsed) ── */
[data-testid="collapsedControl"],
button[aria-label="Open sidebar"] {
    background: #1a4731 !important;
    border-radius: 0 8px 8px 0 !important;
    border: none !important;
    box-shadow: 2px 0 8px rgba(0,0,0,0.3) !important;
    width: 1.8rem !important;
    height: 2.5rem !important;
    top: 4rem !important;
    opacity: 1 !important;
    visibility: visible !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
[data-testid="collapsedControl"] svg,
button[aria-label="Open sidebar"] svg {
    fill: #ffffff !important;
    color: #ffffff !important;
    opacity: 1 !important;
}

/* ── Main Content Padding ── */
[data-testid="stMainBlockContainer"] {
    padding: 2rem 3rem;
    max-width: 1200px;
    margin: 0 auto;
}
@media (max-width: 768px) {
    [data-testid="stMainBlockContainer"] { padding: 1rem 1.2rem; }
}

/* ── Spinner text ── */
[data-testid="stSpinner"] * { color: var(--text-secondary) !important; }


/* ── Hero Section ── */
.hero-section {
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 60%, #52b788 100%);
    border-radius: var(--radius-xl);
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-section::before {
    content: "";
    position: absolute;
    top: -50px; right: -60px;
    width: 280px; height: 280px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
}
.hero-section::after {
    content: "";
    position: absolute;
    bottom: -80px; left: -40px;
    width: 200px; height: 200px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.1;
    margin-bottom: 0.5rem;
    position: relative;
}
.hero-tagline {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.82);
    font-weight: 400;
    margin-bottom: 1.5rem;
    position: relative;
}
.hero-description {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.70);
    max-width: 520px;
    line-height: 1.6;
    position: relative;
}
@media (max-width: 768px) {
    .hero-title { font-size: 2rem; }
    .hero-section { padding: 2rem 1.5rem; }
}

/* ── Section Title ── */
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--border);
    margin-left: 0.5rem;
}

/* ── Upload Zone ── */
.upload-zone {
    border: 2px dashed var(--border);
    border-radius: var(--radius-lg);
    padding: 2.5rem;
    text-align: center;
    background: var(--bg-card);
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: var(--secondary); }
.upload-instructions {
    background: var(--bg-main);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-md);
    padding: 1rem 1.5rem;
    margin-top: 1rem;
    font-size: 0.88rem;
    color: var(--text-secondary);
    line-height: 1.7;
}

/* ── Card ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.5rem;
    box-shadow: var(--shadow-sm);
}
.card-elevated {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 2rem;
    box-shadow: var(--shadow-md);
}

/* ── Image Preview ── */
.image-meta {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 0.75rem;
}
.image-meta-item {
    background: var(--bg-main);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-sm);
    padding: 0.3rem 0.75rem;
    font-size: 0.8rem;
    color: var(--text-secondary);
    font-weight: 500;
}

/* ── Prediction Result ── */
.result-header {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 0.25rem;
}
.result-disease-name {
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1.2;
    margin-bottom: 0.5rem;
}
.result-confidence-num {
    font-size: 2.5rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.confidence-bar-container {
    height: 8px;
    background: var(--border-light);
    border-radius: 99px;
    overflow: hidden;
    margin: 0.5rem 0;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s ease;
}
.confidence-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 99px;
    font-size: 0.78rem;
    font-weight: 600;
}

/* ── Top-3 Predictions ── */
.top3-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid var(--border-light);
}
.top3-item:last-child { border-bottom: none; }
.top3-rank {
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--text-muted);
    width: 1.5rem;
    text-align: center;
}
.top3-name { flex: 1; font-size: 0.9rem; font-weight: 500; color: var(--text-primary); }
.top3-pct { font-size: 0.88rem; font-weight: 700; color: var(--text-secondary); min-width: 3.5rem; text-align: right; }
.top3-bar-bg { flex: 1; height: 5px; background: var(--border-light); border-radius: 99px; overflow: hidden; }
.top3-bar-fill { height: 100%; border-radius: 99px; background: var(--secondary); }

/* ── Disease Info ── */
.disease-info-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.75rem;
    box-shadow: var(--shadow-sm);
}
.info-section-title {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
}
.visual-sign-item {
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    padding: 0.35rem 0;
    font-size: 0.9rem;
    color: var(--text-secondary);
    line-height: 1.5;
}
.visual-sign-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--secondary);
    flex-shrink: 0;
    margin-top: 0.45rem;
}
.action-item {
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    padding: 0.35rem 0;
    font-size: 0.9rem;
    color: var(--text-secondary);
    line-height: 1.5;
}
.action-num {
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--secondary);
    background: rgba(64,145,108,0.1);
    border-radius: 50%;
    width: 1.4rem; height: 1.4rem;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.info-note {
    background: rgba(244,162,97,0.08);
    border-left: 3px solid var(--accent);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 0.75rem 1rem;
    font-size: 0.88rem;
    color: var(--warning);
    font-style: italic;
    margin-top: 1rem;
}

/* ── Warning Banner ── */
.low-confidence-banner {
    background: rgba(176,58,46,0.07);
    border: 1px solid rgba(176,58,46,0.25);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin-bottom: 1.25rem;
}
.quality-warning-banner {
    background: rgba(212,129,46,0.08);
    border: 1px solid rgba(212,129,46,0.25);
    border-radius: var(--radius-md);
    padding: 0.85rem 1.25rem;
    margin-bottom: 1rem;
    font-size: 0.88rem;
    color: var(--warning);
}

/* ── Healthy Result ── */
.healthy-banner {
    background: rgba(45,106,79,0.08);
    border: 1px solid rgba(45,106,79,0.25);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin-bottom: 1.25rem;
}

/* ── History Item ── */
.history-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0.75rem;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-light);
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
    background: var(--bg-main);
}
.history-name { font-weight: 600; color: var(--text-primary); }
.history-conf { font-weight: 700; color: var(--text-muted); }
.history-time { font-size: 0.78rem; color: var(--text-muted); }

/* ── How it works steps ── */
.step-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s, transform 0.2s;
}
.step-card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}
.step-number {
    width: 2.5rem; height: 2.5rem;
    background: linear-gradient(135deg, var(--primary-light), var(--secondary));
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; font-weight: 800; color: #fff;
    margin: 0 auto 0.75rem;
}
.step-title { font-size: 0.95rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.4rem; }
.step-desc { font-size: 0.85rem; color: var(--text-secondary); line-height: 1.5; }

/* ── Disease Explorer Card ── */
.explorer-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: box-shadow 0.15s;
    cursor: default;
}
.explorer-card:hover { box-shadow: var(--shadow-sm); }
.explorer-class-name { font-size: 1rem; font-weight: 700; color: var(--text-primary); }
.explorer-severity-badge {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-left: 0.5rem;
}

/* ── About Model ── */
.model-spec-row {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border-light);
    font-size: 0.9rem;
}
.model-spec-row:last-child { border-bottom: none; }
.spec-label { color: var(--text-muted); font-weight: 500; }
.spec-value { color: var(--text-primary); font-weight: 600; text-align: right; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1a4731 0%, #40916c 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12) !important;
    letter-spacing: 0.01em !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.14) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── File Uploader ── */
/* Outer wrapper */
[data-testid="stFileUploader"] > div {
    background: #ffffff !important;
    border-radius: 14px !important;
}
/* Dropzone area */
[data-testid="stFileUploaderDropzone"] {
    background: #ffffff !important;
    border: 2px dashed #d8e8dc !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #40916c !important;
}
/* All text inside dropzone */
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] * {
    color: #1c2b1e !important;
}
/* Instructions secondary text */
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #4a5e4e !important;
}
/* Browse files button inside uploader */
[data-testid="stFileUploaderDropzone"] button {
    background: #1a4731 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.4rem 1rem !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    transform: none !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: #2d6a4f !important;
    transform: none !important;
}
/* Uploaded file name row */
[data-testid="stFileUploaderFile"] {
    background: #f0f7f2 !important;
    border: 1px solid #d8e8dc !important;
    border-radius: 8px !important;
    margin-top: 0.5rem !important;
}
[data-testid="stFileUploaderFile"] * {
    color: #1c2b1e !important;
}
/* Delete X button */
[data-testid="stFileUploaderDeleteBtn"] > button {
    background: transparent !important;
    color: #7a8f7e !important;
    border: none !important;
    box-shadow: none !important;
    transform: none !important;
    padding: 0.2rem !important;
}
[data-testid="stFileUploaderDeleteBtn"] > button:hover {
    color: #b03a2e !important;
    background: rgba(176,58,46,0.08) !important;
    transform: none !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    background: var(--bg-card) !important;
}

/* ── Radio ── */
.stRadio > div { gap: 0.5rem; }

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    margin-top: 3rem;
    border-top: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 0.82rem;
    line-height: 1.7;
}
.footer-brand {
    font-size: 1rem;
    font-weight: 700;
    color: var(--primary);
    display: block;
    margin-bottom: 0.25rem;
}
.disclaimer {
    background: rgba(0,0,0,0.03);
    border-radius: var(--radius-sm);
    padding: 0.6rem 1rem;
    margin-top: 0.75rem;
    font-size: 0.8rem;
    color: var(--text-muted);
    font-style: italic;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Force sidebar open via JS (override browser localStorage state) ──
components.html("""
<script>
(function() {
    function openSidebar() {
        var doc = window.parent.document;
        // Try by aria-expanded attribute
        var sidebar = doc.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            var isCollapsed = sidebar.getAttribute('aria-expanded') === 'false'
                           || sidebar.style.transform
                           || getComputedStyle(sidebar).transform !== 'none';
        }
        // Click the collapse button if sidebar appears closed
        var collapseBtn = doc.querySelector('[data-testid="stSidebarCollapseButton"]');
        var openBtn = doc.querySelector('[data-testid="collapsedControl"]')
                   || doc.querySelector('button[aria-label="Open sidebar"]');
        if (openBtn) {
            openBtn.click();
        }
    }
    // Try at multiple intervals for reliability
    setTimeout(openSidebar, 50);
    setTimeout(openSidebar, 200);
    setTimeout(openSidebar, 600);
})();
</script>
""", height=0, scrolling=False)


# ──────────────────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────────────────
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "current_prediction" not in st.session_state:
    st.session_state.current_prediction = None
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# ──────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding:1.5rem 0 1rem;">
            <div style="font-size:1.8rem;font-weight:800;color:#fff;line-height:1;">
                🍅 TomatoCare
            </div>
            <div style="font-size:0.82rem;color:rgba(255,255,255,0.55);margin-top:0.3rem;font-weight:400;">
                AI Disease Detection
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "Navigasi",
        options=["Deteksi Penyakit", "Panduan Penyakit", "Tentang Model"],
        index=0,
        label_visibility="visible",
    )

    st.markdown("---")

    # Session History (preview)
    if st.session_state.analysis_history:
        st.markdown(
            "<div style='font-size:0.8rem;font-weight:600;color:rgba(255,255,255,0.5);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.6rem;'>Riwayat Sesi</div>",
            unsafe_allow_html=True,
        )
        for item in reversed(st.session_state.analysis_history[-5:]):
            conf_color = "#52b788" if item["confidence"] >= 0.75 else ("#e67e22" if item["confidence"] >= 0.50 else "#e74c3c")
            st.markdown(
                f"""
                <div style="padding:0.4rem 0;border-bottom:1px solid rgba(255,255,255,0.08);">
                    <div style="font-size:0.85rem;font-weight:600;color:rgba(255,255,255,0.9);">
                        {item['display_name']}
                    </div>
                    <div style="font-size:0.75rem;color:{conf_color};">
                        {item['confidence']*100:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.78rem;color:rgba(255,255,255,0.35);line-height:1.5;'>Model: ConvNeXtTiny<br>Keras 3 · TensorFlow<br>10 Kelas Penyakit</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────
# PAGE 1: DETEKSI PENYAKIT
# ──────────────────────────────────────────────────────────
if page == "Deteksi Penyakit":

    # ── Hero ──
    st.markdown(
        """
        <div class="hero-section">
            <div class="hero-title">🍅 TomatoCare</div>
            <div class="hero-tagline">AI-Powered Tomato Leaf Disease Detection</div>
            <div class="hero-description">
                Upload gambar daun tomat yang jelas dan biarkan model AI
                mengidentifikasi kondisi kesehatan daun secara otomatis.
                Dapatkan prediksi, confidence score, dan rekomendasi tindakan
                dalam hitungan detik.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load Model ──
    with st.spinner("Memuat model AI... Mohon tunggu sebentar."):
        try:
            model = load_model()
            model_loaded = True
        except FileNotFoundError as e:
            st.error(
                f"**File model tidak ditemukan.**\n\n"
                f"Pastikan `model.weights.h5` ada di folder `tomatocare/`.\n\n"
                f"Detail teknis (untuk developer): `{e}`"
            )
            st.stop()
        except Exception as e:
            logger.exception("Error memuat model")
            st.error(
                "**Gagal memuat model AI.**\n\n"
                "Terjadi kesalahan saat mempersiapkan model. "
                "Periksa log aplikasi untuk detail lebih lanjut."
            )
            st.stop()

    # ── How it works ──
    st.markdown('<div class="section-title">Cara Kerja</div>', unsafe_allow_html=True)
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    steps = [
        ("1", "Upload", "Upload foto daun tomat yang jelas dengan pencahayaan baik."),
        ("2", "Analisis", "Model AI ConvNeXt memproses dan menganalisis gambar."),
        ("3", "Prediksi", "Model membandingkan pola visual dari 10 kelas kondisi daun."),
        ("4", "Hasil", "Lihat prediksi, confidence score, dan informasi penyakit."),
    ]
    for col, (num, title, desc) in zip([col_s1, col_s2, col_s3, col_s4], steps):
        with col:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-number">{num}</div>
                    <div class="step-title">{title}</div>
                    <div class="step-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Upload Section ──
    st.markdown('<div class="section-title">Upload Gambar Daun</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="upload-instructions">
            <strong>Tips untuk hasil terbaik:</strong><br>
            &bull; Gunakan pencahayaan yang cukup dan merata<br>
            &bull; Pastikan daun terlihat jelas dan mengisi sebagian besar frame<br>
            &bull; Hindari gambar yang terlalu buram atau terlalu gelap<br>
            &bull; Hindari foto dengan banyak daun yang saling tumpang tindih<br>
            &bull; Format yang didukung: JPG, JPEG, PNG, WEBP &bull; Maks. 10 MB
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Pilih gambar daun tomat",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        help="Upload gambar daun tomat untuk dianalisis",
        key=f"file_uploader_{st.session_state.uploader_key}",
    )

    if uploaded_file is not None:
        # ── Validate file ──
        is_valid, error_msg = validate_image_file(uploaded_file)
        if not is_valid:
            st.error(
                f"**Gambar tidak dapat diproses.**\n\n{error_msg}\n\n"
                "Silakan upload ulang dengan gambar yang valid."
            )
        else:
            # ── Preview ──
            pil_image = load_pil_image(uploaded_file)
            img_info = get_image_info(uploaded_file)

            col_img, col_info = st.columns([2, 1])
            with col_img:
                st.markdown(
                    '<div class="section-title" style="margin-top:1.5rem;">Preview Gambar</div>',
                    unsafe_allow_html=True,
                )
                st.image(pil_image, use_container_width=True)
                st.markdown(
                    f"""
                    <div class="image-meta">
                        <span class="image-meta-item">📄 {img_info['filename']}</span>
                        <span class="image-meta-item">📐 {img_info['width']} × {img_info['height']} px</span>
                        <span class="image-meta-item">💾 {img_info['file_size_kb']:.1f} KB</span>
                        <span class="image-meta-item">🎨 {img_info['mode']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_info:
                st.markdown("<br><br>", unsafe_allow_html=True)

                # Quality warnings
                quality_warnings = get_image_quality_warnings(pil_image)
                if quality_warnings:
                    for warn in quality_warnings:
                        st.markdown(
                            f'<div class="quality-warning-banner">⚠️ {warn}</div>',
                            unsafe_allow_html=True,
                        )

                st.markdown("<br>", unsafe_allow_html=True)
                analyze_btn = st.button(
                    "🔍 Analisis Gambar",
                    key="analyze_btn",
                    use_container_width=True,
                    help="Klik untuk memulai analisis gambar menggunakan model AI",
                )

            # ── Analyze ──
            if analyze_btn:
                st.session_state.analyzed = False
                with st.spinner("Menganalisis gambar..."):
                    try:
                        preprocessed = preprocess_image(uploaded_file)
                        if preprocessed is None:
                            st.error(
                                "Terjadi kesalahan saat memproses gambar. "
                                "Silakan coba dengan gambar lain."
                            )
                        else:
                            probabilities = predict(model, preprocessed)
                            if probabilities is None:
                                st.error(
                                    "Terjadi kesalahan saat menjalankan analisis. "
                                    "Silakan coba lagi."
                                )
                            else:
                                result = format_prediction(probabilities)
                                st.session_state.current_prediction = result
                                st.session_state.analyzed = True

                                # Add to history
                                history_entry = {
                                    "display_name": result["display_name"],
                                    "predicted_class": result["predicted_class"],
                                    "confidence": result["confidence"],
                                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                                }
                                st.session_state.analysis_history.append(history_entry)
                                st.rerun()
                    except Exception:
                        logger.exception("Error saat analisis")
                        st.error(
                            "Terjadi kesalahan yang tidak terduga saat menganalisis gambar. "
                            "Silakan coba dengan gambar lain."
                        )

            # ── Show Results ──
            if st.session_state.analyzed and st.session_state.current_prediction:
                result = st.session_state.current_prediction
                disease_info = get_disease_info(result["predicted_class"])
                severity = disease_info.get("severity", "unknown")
                conf_color = get_confidence_color(result["confidence_level"])

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-title">Hasil Analisis AI</div>',
                    unsafe_allow_html=True,
                )

                # ── Low confidence warning ──
                if result["confidence_level"] == "low":
                    st.markdown(
                        f"""
                        <div class="low-confidence-banner">
                            <strong style="color:#b03a2e;">⚠️ Prediksi dengan Confidence Rendah</strong><br>
                            <span style="font-size:0.9rem;color:#7b241c;">
                            Model kurang yakin dengan gambar ini. Coba:<br>
                            &bull; Upload gambar dengan pencahayaan lebih baik<br>
                            &bull; Pastikan satu daun terlihat jelas di frame<br>
                            &bull; Hindari gambar yang terlalu buram<br>
                            &bull; Pastikan ini adalah gambar daun tomat
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # ── Healthy banner ──
                if disease_info.get("status") == "healthy":
                    st.markdown(
                        """
                        <div class="healthy-banner">
                            <strong style="color:#2d6a4f;">Daun tampak sehat!</strong>
                            <span style="font-size:0.9rem;color:#1a4731;"> Tidak terdeteksi tanda-tanda penyakit pada gambar ini.</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # ── Main result layout ──
                col_res_left, col_res_right = st.columns([1, 1])

                with col_res_left:
                    severity_colors = {"none": "#2d6a4f", "moderate": "#e67e22", "high": "#c0392b", "unknown": "#7f8c8d"}
                    sev_color = severity_colors.get(severity, "#7f8c8d")
                    severity_label = {"none": "Sehat", "moderate": "Perlu Perhatian", "high": "Segera Tangani", "unknown": "Tidak Diketahui"}.get(severity, "")

                    st.markdown(
                        f"""
                        <div class="card-elevated">
                            <div class="result-header">Prediksi Model AI</div>
                            <div class="result-disease-name">{result['display_name']}</div>
                            <span class="confidence-badge" style="background:{sev_color}18;color:{sev_color};border:1px solid {sev_color}30;">
                                {severity_label}
                            </span>
                            <hr style="border:none;border-top:1px solid #edf5ef;margin:1rem 0;">
                            <div class="result-header">Confidence Score</div>
                            <div class="result-confidence-num" style="color:{conf_color};">{result['confidence_pct']}</div>
                            <div class="confidence-bar-container">
                                <div class="confidence-bar-fill" style="width:{result['confidence']*100:.1f}%;background:{conf_color};"></div>
                            </div>
                            <span class="confidence-badge" style="background:{conf_color}18;color:{conf_color};border:1px solid {conf_color}30;margin-top:0.25rem;">
                                {result['confidence_label']}
                            </span>
                            <div style="font-size:0.82rem;color:#7a8f7e;margin-top:0.5rem;">{result['confidence_description']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col_res_right:
                    # Top-3 predictions
                    top3_html = ""
                    for i, (cls, disp, conf) in enumerate(result["top3"]):
                        bar_color = "#2d6a4f" if i == 0 else ("#40916c" if i == 1 else "#b7e4c7")
                        top3_html += f"""
                        <div class="top3-item">
                            <div class="top3-rank">#{i+1}</div>
                            <div class="top3-name">{disp}</div>
                            <div style="flex:1;margin:0 0.75rem;">
                                <div class="confidence-bar-container" style="height:5px;">
                                    <div class="confidence-bar-fill" style="width:{conf*100:.1f}%;background:{bar_color};"></div>
                                </div>
                            </div>
                            <div class="top3-pct">{conf*100:.1f}%</div>
                        </div>
                        """

                    st.markdown(
                        f"""
                        <div class="card-elevated" style="height:100%;">
                            <div class="result-header" style="margin-bottom:0.75rem;">Top Prediksi</div>
                            {top3_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # ── Disease Information ──
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-title">Informasi Kondisi</div>',
                    unsafe_allow_html=True,
                )

                col_dinfo_1, col_dinfo_2 = st.columns([1, 1])

                with col_dinfo_1:
                    signs_html = "".join(
                        f'<div class="visual-sign-item"><div class="visual-sign-dot"></div><div>{sign}</div></div>'
                        for sign in disease_info.get("visual_signs", [])
                    )
                    note_html = (
                        f'<div class="info-note">{disease_info["note"]}</div>'
                        if disease_info.get("note") else ""
                    )
                    st.markdown(
                        f"""
                        <div class="disease-info-card">
                            <div class="info-section-title">Deskripsi</div>
                            <p style="font-size:0.92rem;color:#4a5e4e;line-height:1.65;margin-bottom:1.25rem;">
                                {disease_info.get('description', '')}
                            </p>
                            <div class="info-section-title">Tanda Visual</div>
                            {signs_html}
                            {note_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col_dinfo_2:
                    actions = disease_info.get("general_actions", [])
                    actions_html = "".join(
                        f'<div class="action-item"><div class="action-num">{i+1}</div><div>{action}</div></div>'
                        for i, action in enumerate(actions)
                    )
                    st.markdown(
                        f"""
                        <div class="disease-info-card">
                            <div class="info-section-title">Langkah yang Disarankan</div>
                            {actions_html}
                            <div style="background:rgba(26,71,49,0.05);border-radius:8px;padding:0.75rem;margin-top:1.25rem;font-size:0.8rem;color:#7a8f7e;font-style:italic;">
                                Informasi ini bersifat umum dan edukatif. Untuk diagnosis dan penanganan yang tepat, selalu konsultasikan dengan ahli pertanian atau penyuluh pertanian setempat.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # ── Download Report ──
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("Unduh Laporan Prediksi"):
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    top3_text = "\n".join(
                        f"  {i+1}. {disp}: {conf*100:.2f}%"
                        for i, (cls, disp, conf) in enumerate(result["top3"])
                    )
                    report_text = f"""TomatoCare — Laporan Prediksi AI
{'='*50}
Waktu Analisis  : {timestamp}
File Gambar     : {img_info['filename']}
Dimensi Gambar  : {img_info['width']} x {img_info['height']} px

{'='*50}
HASIL PREDIKSI
{'='*50}
Kondisi Terdeteksi : {result['display_name']}
Confidence Score   : {result['confidence_pct']}
Level Confidence   : {result['confidence_label']}

{'='*50}
TOP 3 PREDIKSI
{'='*50}
{top3_text}

{'='*50}
INFORMASI KONDISI
{'='*50}
{disease_info.get('description', '')}

Tanda Visual:
{''.join(f"  - {s}" + chr(10) for s in disease_info.get('visual_signs', []))}
Langkah yang Disarankan:
{''.join(f"  {i+1}. {a}" + chr(10) for i, a in enumerate(disease_info.get('general_actions', [])))}
{'='*50}
DISCLAIMER
{'='*50}
Prediksi ini dibuat oleh model AI untuk tujuan edukatif dan
informasi umum. Prediksi model AI tidak boleh dijadikan sebagai
satu-satunya dasar diagnosis pertanian. Selalu konsultasikan
dengan ahli pertanian atau penyuluh pertanian yang berpengalaman.

TomatoCare v1.0 — AI-Powered Tomato Leaf Disease Detection
"""
                    st.download_button(
                        label="Unduh Laporan (.txt)",
                        data=report_text,
                        file_name=f"tomatocare_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                    )

                # ── Analisis Gambar Lain button ──
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    """
                    <div style="border-top:1px solid #d8e8dc;margin:0.5rem 0 1.5rem;padding-top:1.5rem;text-align:center;">
                        <div style="font-size:0.9rem;color:#7a8f7e;margin-bottom:0.75rem;">
                            Selesai menganalisis? Upload gambar daun lainnya untuk analisis berikutnya.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                col_rb1, col_rb2, col_rb3 = st.columns([1, 2, 1])
                with col_rb2:
                    if st.button(
                        "🔄 Analisis Gambar Lain",
                        key="reset_full_btn",
                        use_container_width=True,
                        help="Reset semua — upload gambar baru untuk dianalisis",
                    ):
                        st.session_state.analyzed = False
                        st.session_state.current_prediction = None
                        st.session_state.uploader_key += 1
                        st.rerun()


# ──────────────────────────────────────────────────────────
# PAGE 2: PANDUAN PENYAKIT
# ──────────────────────────────────────────────────────────
elif page == "Panduan Penyakit":
    st.markdown(
        """
        <div class="hero-section" style="padding:2rem 2.5rem;">
            <div class="hero-title" style="font-size:2rem;">Panduan Penyakit Daun Tomat</div>
            <div class="hero-description" style="margin-top:0.5rem;">
                Jelajahi seluruh 10 kondisi daun tomat yang dapat dideteksi oleh model AI ini.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    severity_labels = {
        "none": ("Sehat", "#2d6a4f"),
        "moderate": ("Perlu Perhatian", "#e67e22"),
        "high": ("Segera Tangani", "#c0392b"),
    }

    for class_name in CLASS_NAMES:
        info = get_disease_info(class_name)
        display = info["display_name"]
        severity = info.get("severity", "unknown")
        sev_label, sev_color = severity_labels.get(severity, ("Unknown", "#7f8c8d"))
        status = info.get("status", "disease")

        with st.expander(f"{display}  |  {sev_label}", expanded=False):
            col_exp1, col_exp2 = st.columns([1, 1])
            with col_exp1:
                st.markdown(
                    f"""
                    <div style="margin-bottom:1rem;">
                        <div style="font-size:1.1rem;font-weight:700;color:#1c2b1e;margin-bottom:0.25rem;">{display}</div>
                        <span class="confidence-badge" style="background:{sev_color}15;color:{sev_color};border:1px solid {sev_color}30;font-size:0.75rem;">
                            {sev_label}
                        </span>
                        <span class="confidence-badge" style="background:#40916c15;color:#40916c;border:1px solid #40916c30;font-size:0.75rem;margin-left:0.3rem;">
                            Indeks kelas: {CLASS_NAMES.index(class_name)}
                        </span>
                    </div>
                    <div style="font-size:0.92rem;color:#4a5e4e;line-height:1.65;margin-bottom:1rem;">
                        {info.get('description', '')}
                    </div>
                    <div class="info-section-title">Tanda Visual</div>
                    {''.join(f"<div class='visual-sign-item'><div class='visual-sign-dot'></div><div>{s}</div></div>" for s in info.get('visual_signs', []))}
                    """,
                    unsafe_allow_html=True,
                )
            with col_exp2:
                st.markdown(
                    f"""
                    <div class="info-section-title" style="margin-top:0.5rem;">Langkah yang Disarankan</div>
                    {''.join(f"<div class='action-item'><div class='action-num'>{i+1}</div><div>{a}</div></div>" for i, a in enumerate(info.get('general_actions', [])))}
                    {f'<div class="info-note">{info["note"]}</div>' if info.get('note') else ''}
                    """,
                    unsafe_allow_html=True,
                )


# ──────────────────────────────────────────────────────────
# PAGE 3: TENTANG MODEL
# ──────────────────────────────────────────────────────────
elif page == "Tentang Model":
    st.markdown(
        """
        <div class="hero-section" style="padding:2rem 2.5rem;">
            <div class="hero-title" style="font-size:2rem;">Tentang Model AI</div>
            <div class="hero-description" style="margin-top:0.5rem;">
                Informasi teknis mengenai model yang digunakan dalam aplikasi TomatoCare.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_about1, col_about2 = st.columns([1, 1])

    with col_about1:
        st.markdown('<div class="section-title">Spesifikasi Model</div>', unsafe_allow_html=True)
        specs = [
            ("Tipe Model", "ConvNeXtTiny Transfer Learning"),
            ("Framework", "TensorFlow / Keras 3"),
            ("Resolusi Input", f"{IMG_HEIGHT} × {IMG_WIDTH} pixel"),
            ("Jumlah Kelas", f"{NUM_CLASSES} kondisi"),
            ("Aktivasi Output", "Softmax"),
            ("Akurasi Validasi", "91.90%"),
            ("Macro F1-Score", "91.91%"),
        ]
        spec_rows = "".join(
            f'<div class="model-spec-row"><span class="spec-label">{k}</span><span class="spec-value">{v}</span></div>'
            for k, v in specs
        )
        st.markdown(f'<div class="card">{spec_rows}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Kelas yang Dapat Dideteksi</div>', unsafe_allow_html=True)
        for i, cls in enumerate(CLASS_NAMES):
            display = get_display_name(cls)
            info = get_disease_info(cls)
            sev = info.get("severity", "unknown")
            sev_colors = {"none": "#2d6a4f", "moderate": "#e67e22", "high": "#c0392b"}
            sev_color = sev_colors.get(sev, "#7f8c8d")
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:0.75rem;padding:0.5rem 0;border-bottom:1px solid #edf5ef;">
                    <span style="font-size:0.75rem;font-weight:700;color:#7a8f7e;width:1.5rem;text-align:center;">{i}</span>
                    <span style="font-size:0.9rem;font-weight:600;color:#1c2b1e;flex:1;">{display}</span>
                    <span style="font-size:0.72rem;font-weight:600;color:{sev_color};background:{sev_color}15;padding:0.15rem 0.5rem;border-radius:99px;">{sev.title()}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_about2:
        st.markdown('<div class="section-title">Arsitektur & Pipeline</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="card" style="line-height:1.8;">
                <div class="info-section-title">Backbone</div>
                <p style="font-size:0.9rem;color:#4a5e4e;margin-bottom:1rem;">
                    Model menggunakan <strong>ConvNeXtTiny</strong> yang di-pretrain pada dataset ImageNet
                    (~1.2 juta gambar, 1000 kelas) sebagai feature extractor. ConvNeXtTiny menggabungkan
                    desain modern transformer dengan efisiensi konvolusi.
                </p>
                <div class="info-section-title">Pipeline Inference</div>
                <div style="font-family:monospace;font-size:0.82rem;background:#f8faf8;border-radius:8px;padding:1rem;color:#2d3748;line-height:2;">
                    Input (224×224×3) [0-255]<br>
                    &darr; Data Augmentation (dinonaktifkan saat inference)<br>
                    &darr; Normalisasi ImageNet (layer internal model)<br>
                    &darr; ConvNeXtTiny Backbone (feature extraction)<br>
                    &darr; GlobalAveragePooling2D<br>
                    &darr; Dense(256) + L2 + BatchNorm + GELU + Dropout(0.4)<br>
                    &darr; Dense(10, softmax)<br>
                    Output: 10 probabilitas kelas
                </div>
                <br>
                <div class="info-section-title">Metode Training</div>
                <p style="font-size:0.9rem;color:#4a5e4e;">
                    Model dilatih dengan <strong>2 fase Transfer Learning</strong>:<br>
                    <strong>Fase 1</strong> — Feature Extraction: Backbone dibekukan, hanya classifier head yang dilatih.<br>
                    <strong>Fase 2</strong> — Fine-Tuning: 30 layer teratas backbone dibuka dan dilatih ulang
                    dengan learning rate sangat kecil (1e-5) untuk mencegah catastrophic forgetting.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Detail Teknis (untuk Developer)"):
            st.code(
                f"""
# Model Architecture Summary
Model Name   : ConvNeXt_Transfer_Learning
Input Shape  : (None, {IMG_HEIGHT}, {IMG_WIDTH}, 3)
Preprocessing: Raw float32 [0, 255]
               (Normalization inside model via
                convnext_tiny_prestem_normalization
                mean=[123.675, 116.28, 103.53])
Augmentation : RandomFlip, RandomRotation(0.15),
               RandomZoom(0.1), RandomTranslation(0.1)
               [Disabled during inference]

Output       : Dense({NUM_CLASSES}, activation='softmax')
Loss         : sparse_categorical_crossentropy
Optimizer    : AdamW

Model File   : model.weights.h5 (weights only)
Loading      : model.load_weights('model.weights.h5')
Keras Ver    : 3.13.2
                """.strip(),
                language="python",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Batasan Model</div>', unsafe_allow_html=True)
        limitations = [
            "Model hanya dilatih pada 10 kelas penyakit daun tomat — tidak dapat mengidentifikasi penyakit lain atau tanaman lain.",
            "Akurasi model (91.9%) berarti sekitar 8 dari 100 prediksi dapat berbeda dari kondisi sebenarnya.",
            "Kualitas gambar sangat mempengaruhi hasil — gambar buram, gelap, atau tidak jelas akan menghasilkan prediksi yang kurang akurat.",
            "Model tidak menggantikan diagnosa dari ahli pertanian yang berpengalaman.",
            "Dataset training menggunakan gambar laboratorium — kondisi lapangan yang berbeda dapat mempengaruhi akurasi.",
        ]
        for lim in limitations:
            st.markdown(
                f'<div class="visual-sign-item"><div class="visual-sign-dot" style="background:#e67e22;"></div><div style="font-size:0.88rem;color:#4a5e4e;">{lim}</div></div>',
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-footer">
        <span class="footer-brand">🍅 TomatoCare</span>
        AI-assisted tomato leaf disease classification
        <div class="disclaimer">
            Aplikasi ini dibuat untuk tujuan edukatif dan informasi umum.<br>
            Prediksi model AI tidak boleh dijadikan sebagai satu-satunya dasar diagnosis pertanian.<br>
            Selalu konsultasikan dengan ahli pertanian atau penyuluh pertanian yang berpengalaman.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

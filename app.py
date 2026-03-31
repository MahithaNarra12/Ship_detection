import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import base64, os

st.set_page_config(page_title="ShipVision — SAR Detection", page_icon="🛰️", layout="wide", initial_sidebar_state="collapsed")

# Load bg.jpg as base64
def get_bg():
    search_dirs = [
        os.path.dirname(os.path.abspath(__file__)),
        r"C:\Users\mahit\OneDrive\Desktop\ship\Ship_detection",
        r"C:\Users\mahit\Desktop\ship\Ship_detection",
        r"C:\Users\mahit\Ship_detection",
    ]
    for d in search_dirs:
        for name in ["bg.png", "bg.jpg", "bg.jpeg", "bg"]:
            p = os.path.join(d, name)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    data = base64.b64encode(f.read()).decode()
                    mime = "image/png" if name.endswith(".png") else "image/jpeg"
                    return data, mime
    return None, None

bg, mime = get_bg()
bg_url = f"data:{mime};base64,{bg}" if bg else ""

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body {{ font-family: 'Poppins', sans-serif; }}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="collapsedControl"] {{ display: none !important; }}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}

/* ── BACKGROUND IMAGE on the root container ── */
[data-testid="stAppViewContainer"] {{
    background-image: url("{bg_url}") !important;
    background-size: cover !important;
    background-position: top center !important;
    background-attachment: fixed !important;
    background-repeat: no-repeat !important;
    min-height: 100vh;
    position: relative;
}}

/* ── DARK OVERLAY ── */
[data-testid="stAppViewContainer"]::before {{
    content: '';
    position: fixed;
    inset: 0;
    background: rgba(1, 18, 38, 0.55);
    z-index: 0;
    pointer-events: none;
}}

/* ── MAKE ALL INNER DIVS TRANSPARENT ── */
[data-testid="stAppViewContainer"] > div,
[data-testid="stAppViewBlockContainer"],
.stApp,
section.main,
.main {{ background: transparent !important; }}

/* ── ALL CONTENT ABOVE OVERLAY ── */
[data-testid="stAppViewContainer"] > div {{ position: relative; z-index: 1; }}

/* ── NAV ── */
.nav {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 5vw;
    background: rgba(1, 15, 32, 0.35);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(0, 229, 204, 0.1);
    position: sticky; top: 0; z-index: 999;
    color: #fff;
}}
.nav-logo {{ font-size: 1.4rem; font-weight: 900; color: #fff; letter-spacing: -0.02em; }}
.nav-logo span {{ color: #00e5cc; text-shadow: 0 0 16px rgba(0,229,204,0.8); }}
.nav-links {{ display: flex; gap: 2rem; font-size: 0.85rem; font-weight: 500; color: rgba(255,255,255,0.6); }}
.nav-btn {{
    background: linear-gradient(135deg, #00b4d8, #00e5cc);
    color: #011226; font-size: 0.82rem; font-weight: 800;
    padding: 0.5rem 1.4rem; border-radius: 50px;
    box-shadow: 0 4px 18px rgba(0,229,204,0.4);
}}

/* ── HERO ── */
.hero {{
    display: grid; grid-template-columns: 1.1fr 0.9fr;
    gap: 2rem; align-items: center;
    padding: 4rem 5vw 3rem;
    min-height: calc(100vh - 60px);
    color: #fff;
}}
.hero-eyebrow {{
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: rgba(0,229,204,0.1); border: 1px solid rgba(0,229,204,0.25);
    color: #00e5cc; font-size: 0.72rem; font-weight: 700;
    padding: 0.35rem 1rem; border-radius: 50px;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 1.2rem;
}}
.hero-h1 {{
    font-size: clamp(2.4rem, 4.5vw, 3.8rem); font-weight: 900;
    line-height: 1.08; letter-spacing: -0.04em; margin-bottom: 1.2rem;
    text-shadow: 0 2px 30px rgba(0,0,0,0.6);
}}
.hero-h1 span {{ color: #00e5cc; text-shadow: 0 0 40px rgba(0,229,204,0.6); }}
.hero-p {{
    font-size: 1rem; color: rgba(255,255,255,0.7);
    line-height: 1.75; max-width: 460px; margin-bottom: 2rem;
}}
.hero-btns {{ display: flex; gap: 1rem; margin-bottom: 2.5rem; flex-wrap: wrap; }}
.btn-main {{
    background: linear-gradient(135deg, #00b4d8, #00e5cc);
    color: #011226; font-size: 0.88rem; font-weight: 800;
    padding: 0.8rem 2.2rem; border-radius: 50px;
    box-shadow: 0 6px 24px rgba(0,229,204,0.45); letter-spacing: 0.02em;
}}
.btn-ghost {{
    color: #fff; font-size: 0.88rem; font-weight: 600;
    padding: 0.8rem 1.8rem; border-radius: 50px;
    border: 1.5px solid rgba(0,229,204,0.35);
    background: rgba(0,229,204,0.07);
}}
.hero-stats {{
    display: flex; gap: 2.5rem;
    padding-top: 1.5rem; border-top: 1px solid rgba(0,229,204,0.2);
}}
.hs-val {{
    font-size: 1.7rem; font-weight: 900; color: #00e5cc;
    letter-spacing: -0.03em; line-height: 1;
    text-shadow: 0 0 20px rgba(0,229,204,0.5);
}}
.hs-lbl {{ font-size: 0.7rem; color: rgba(255,255,255,0.45); margin-top: 0.2rem; }}

/* ── GLASS CARD (right side) ── */
.glass-card {{
    background: rgba(0, 180, 216, 0.08);
    backdrop-filter: blur(24px);
    border: 1px solid rgba(0, 229, 204, 0.2);
    border-radius: 24px;
    padding: 2rem;
    box-shadow: 0 8px 40px rgba(0,180,216,0.15), inset 0 1px 0 rgba(0,229,204,0.1);
    color: #fff;
}}
.gc-title {{
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #00e5cc; margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: 0.5rem;
}}
.gc-title::before {{
    content: ''; width: 4px; height: 16px;
    background: linear-gradient(180deg, #00b4d8, #00e5cc);
    border-radius: 4px; display: inline-block;
}}

/* ── UPLOAD WIDGET ── */
[data-testid="stFileUploader"] {{
    background: rgba(0,20,50,0.7) !important;
    border: 2px dashed rgba(0,229,204,0.5) !important;
    border-radius: 14px !important;
}}
[data-testid="stFileUploader"]:hover {{ border-color: #00e5cc !important; }}
[data-testid="stFileUploader"] * {{ color: #fff !important; }}
[data-testid="stFileUploader"] small {{ color: rgba(255,255,255,0.6) !important; }}
[data-testid="stFileUploaderDropzone"] {{ border: none !important; background: transparent !important; }}
[data-testid="stFileUploaderDropzoneInstructions"] * {{ color: #fff !important; font-weight: 600 !important; }}
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploaderDropzoneInstructions"] span small,
section[data-testid="stFileUploaderDropzoneInstructions"] > div > small {{ display: none !important; visibility: hidden !important; }}
[data-testid="stBaseButton-secondary"] {{ background: linear-gradient(135deg,#00b4d8,#00e5cc) !important; color: #011226 !important; border: none !important; font-weight: 700 !important; border-radius: 8px !important; }}
[data-testid="stSlider"] > div > div > div {{ background: linear-gradient(90deg,#00b4d8,#00e5cc) !important; }}
[data-testid="stSlider"] label {{ color: #fff !important; font-size: 0.88rem !important; font-weight: 700 !important; text-shadow: 0 1px 4px rgba(0,0,0,0.8) !important; }}
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {{ color: #fff !important; font-weight: 700 !important; }}
[data-testid="stSlider"] div[data-baseweb="slider"] div {{ background: rgba(0,229,204,0.25) !important; }}
[data-testid="stSlider"] div[role="slider"] {{ background: #00e5cc !important; box-shadow: 0 0 10px rgba(0,229,204,0.8) !important; }}
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {{ color: #fff !important; font-weight: 600 !important; }}

/* ── RESULTS ── */
.results-wrap {{ padding: 0 5vw 3rem; color: #fff; background: transparent; }}
.r-title {{
    font-size: 1.4rem; font-weight: 800; color: #fff;
    letter-spacing: -0.02em; margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: 0.6rem;
}}
.r-title::before {{
    content: ''; width: 4px; height: 22px;
    background: linear-gradient(180deg,#00b4d8,#00e5cc);
    border-radius: 4px; display: inline-block;
}}
.img-panel {{
    background: rgba(0,180,216,0.07);
    border: 1px solid rgba(0,229,204,0.18);
    border-radius: 18px; overflow: hidden;
    box-shadow: 0 4px 30px rgba(0,180,216,0.12);
}}
.ip-bar {{
    padding: 0.8rem 1.2rem;
    border-bottom: 1px solid rgba(0,229,204,0.12);
    display: flex; align-items: center; justify-content: space-between;
    background: rgba(0,0,0,0.25);
}}
.ip-lbl {{ font-size: 0.7rem; font-weight: 700; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 0.1em; }}
.ip-chip {{ font-size: 0.65rem; font-weight: 700; padding: 0.2rem 0.7rem; border-radius: 50px; text-transform: uppercase; }}
.chip-c {{ background: rgba(0,229,204,0.15); color: #00e5cc; border: 1px solid rgba(0,229,204,0.25); }}
.chip-b {{ background: rgba(0,180,216,0.15); color: #00b4d8; border: 1px solid rgba(0,180,216,0.3); }}

/* ── METRICS ── */
.mrow {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin: 1.5rem 0; }}
.mc {{
    background: rgba(0,180,216,0.08);
    border: 1px solid rgba(0,229,204,0.15);
    border-radius: 16px; padding: 1.4rem 1.2rem; text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s, box-shadow 0.2s;
}}
.mc:hover {{ transform: translateY(-3px); box-shadow: 0 8px 28px rgba(0,229,204,0.18); }}
.mc-i {{ font-size: 1.5rem; margin-bottom: 0.5rem; }}
.mc-v {{
    font-size: 1.9rem; font-weight: 900; color: #00e5cc;
    letter-spacing: -0.04em; line-height: 1;
    text-shadow: 0 0 18px rgba(0,229,204,0.4);
}}
.mc-v .u {{ font-size: 0.95rem; color: rgba(255,255,255,0.3); font-weight: 600; }}
.mc-l {{ font-size: 0.65rem; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; margin-top: 0.3rem; }}

/* ── ALERT ── */
.al {{ border-radius: 12px; padding: 0.9rem 1.4rem; font-size: 0.86rem; font-weight: 600; margin: 1rem 0; display: flex; align-items: center; gap: 0.6rem; }}
.al-y {{ background: rgba(0,229,204,0.1); color: #00e5cc; border-left: 4px solid #00e5cc; }}
.al-n {{ background: rgba(255,255,255,0.04); color: rgba(255,255,255,0.4); border-left: 4px solid rgba(255,255,255,0.15); }}

/* ── EMPTY ── */
.empty {{
    background: rgba(0,20,50,0.6);
    border: 1.5px dashed rgba(0,229,204,0.35);
    border-radius: 18px; padding: 4rem 2rem; text-align: center; margin: 1rem 0;
}}
.empty-i {{ font-size: 2.8rem; margin-bottom: 0.8rem; }}
.empty-t {{ font-size: 1.1rem; font-weight: 800; color: #fff; }}
.empty-s {{ font-size: 0.88rem; color: rgba(255,255,255,0.7); margin-top: 0.4rem; }}

/* ── FOOTER ── */
.footer {{
    background: rgba(1,10,22,0.4); backdrop-filter: blur(10px);
    padding: 1.4rem 5vw;
    display: flex; justify-content: space-between; align-items: center;
    border-top: 1px solid rgba(0,229,204,0.1); color: #fff;
}}
.footer-logo {{ font-size: 1rem; font-weight: 900; }}
.footer-logo span {{ color: #00e5cc; text-shadow: 0 0 12px rgba(0,229,204,0.5); }}
.footer-copy {{ font-size: 0.72rem; color: rgba(255,255,255,0.3); }}

[data-testid="stImage"] img {{ border-radius: 0 !important; display: block; }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO('ship_model.pt')
model = load_model()

# ── NAV ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav">
    <div class="nav-logo">Ship<span>Vision</span></div>
    <div class="nav-btn">🛰️ &nbsp;Track 1 · SAR</div>
</div>
""", unsafe_allow_html=True)

# ── HERO + UPLOAD SIDE BY SIDE ────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div>
        <div class="hero-eyebrow">🌊 SAR Maritime Intelligence</div>
        <div class="hero-h1">Detect every ship.<br><span>In any condition.</span></div>
        <div class="hero-p">
            Upload a Sentinel-1 SAR satellite image and get instant vessel detection
            powered by YOLOv8-Nano — trained on SSDD, works through clouds, fog, and darkness.
        </div>
        <div class="hero-stats">            <div><div class="hs-val">98.6%</div><div class="hs-lbl">mAP Accuracy</div></div>
            <div><div class="hs-val">&lt;50ms</div><div class="hs-lbl">Inference Speed</div></div>
            <div><div class="hs-val">SSDD</div><div class="hs-lbl">Training Dataset</div></div>
        </div>
    </div>
    <div class="glass-card">
        <div class="gc-title">Upload SAR Image</div>
""", unsafe_allow_html=True)

# Streamlit widgets inside the glass card column
uploaded_file = st.file_uploader(
    "Drop Sentinel-1 SAR scene — JPG, PNG, TIF",
    type=["jpg", "png", "jpeg", "tif"],
    label_visibility="collapsed"
)
conf_threshold = st.slider("Confidence Threshold", 0.10, 1.0, 0.45, 0.05)

st.markdown("</div></div>", unsafe_allow_html=True)

# ── RESULTS ───────────────────────────────────────────────────────────────────
st.markdown('<div class="results-wrap">', unsafe_allow_html=True)

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    with st.spinner("Scanning for vessels..."):
        results = model.predict(img_array, conf=conf_threshold)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        ship_count = len(results[0].boxes)
        latency = results[0].speed['inference']
        boxes = results[0].boxes
        avg_conf = float(np.mean(boxes.conf.cpu().numpy())) if len(boxes) > 0 else 0.0

    st.markdown('<div class="r-title">Detection Results</div>', unsafe_allow_html=True)

    if ship_count > 0:
        st.markdown(f'<div class="al al-y">🚢 &nbsp; {ship_count} vessel{"s" if ship_count!=1 else ""} detected &nbsp;·&nbsp; avg confidence {avg_conf:.1%}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="al al-n">◌ &nbsp; No vessels detected — try lowering the confidence threshold</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="img-panel"><div class="ip-bar"><span class="ip-lbl">Original SAR Scene</span><span class="ip-chip chip-c">INPUT</span></div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="img-panel"><div class="ip-bar"><span class="ip-lbl">Detection Output</span><span class="ip-chip chip-b">{ship_count} VESSEL{"S" if ship_count!=1 else ""}</span></div>', unsafe_allow_html=True)
        st.image(res_rgb, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="mrow">
        <div class="mc"><div class="mc-i">🚢</div><div class="mc-v">{ship_count}</div><div class="mc-l">Vessels Found</div></div>
        <div class="mc"><div class="mc-i">🎯</div><div class="mc-v">{avg_conf:.2f}</div><div class="mc-l">Avg Confidence</div></div>
        <div class="mc"><div class="mc-i">⚡</div><div class="mc-v">{latency:.1f}<span class="u">ms</span></div><div class="mc-l">Inference Time</div></div>
        <div class="mc"><div class="mc-i">🔬</div><div class="mc-v">{int(conf_threshold*100)}<span class="u">%</span></div><div class="mc-l">Conf Threshold</div></div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="empty">
        <div class="empty-i">🛰️</div>
        <div class="empty-t">No image uploaded yet</div>
        <div class="empty-s">Upload a Sentinel-1 SAR scene above to begin vessel detection</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-logo">Ship<span>Vision</span></div>
    <div class="footer-copy">YOLOv8-Nano · SSDD · Sentinel-1 SAR · Track 1 · 98.6% mAP</div>
</div>
""", unsafe_allow_html=True)

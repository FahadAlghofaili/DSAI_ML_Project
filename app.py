import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import warnings
from pathlib import Path
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────
st.set_page_config(
    page_title="HydroGuard AI",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── PATHS ────────────────────────────────────────────────────────
# Expected layout next to this app.py:
#
#   your_project/
#   ├── app.py
#   ├── maintenance_needed_model.pkl
#   └── ACWA_with_target_FINAL.csv
#
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "maintenance_needed_model.pkl"
CSV_PATH   = BASE_DIR / "ACWA_with_target_FINAL.csv"

# ─── FEATURE NAMES ────────────────────────────────────────────────
# Exactly the 16 features the model was trained on (from model.feature_names_in_).
# Do NOT add or remove anything from this list.
FEATURE_NAMES = [
    "Room_temperature",
    "Current_1_stack",
    "Voltage_1_Stack",
    "DC_Power_Consumption_1_Stack",
    "H2_side_outlet_temp_1_stack",
    "O2_side_outlet_temp_1_stack",
    "Lye_Supply_to_Electrolyzer_Temp",
    "Lye_Concentration",
    "Lye_Flow_to_1_Stack",
    "H2_Separator_Level",
    "O2_Separator_Level",
    "LDI_H2_&_O2_Separator",
    "Pressure_O2_Separator",
    "H2_Flowrate_Purification_outlet",
    "DM_water_condctivity",
    "DM_water_flow_from_B.L.",
]

# ─── FEATURE UI CONFIG ────────────────────────────────────────────
# (default, min, max, unit, tooltip)
FEAT_CONFIG = {
    "Room_temperature":               (25.0,   24.0,   26.0,    "°C",    "Ambient room temperature"),
    "Current_1_stack":                (4200.0,  0.0,  8001.0,   "A",     "Electrolyzer stack current"),
    "Voltage_1_Stack":                (325.0,   0.0,   640.0,   "V",     "Stack voltage"),
    "DC_Power_Consumption_1_Stack":   (2400.0,  0.0,  5120.0,   "kW",    "DC power consumed by stack"),
    "H2_side_outlet_temp_1_stack":    (57.0,   23.0,   87.0,    "°C",    "H2 side outlet temperature"),
    "O2_side_outlet_temp_1_stack":    (60.0,   24.0,   92.0,    "°C",    "O2 side outlet temperature"),
    "Lye_Supply_to_Electrolyzer_Temp":(52.0,   24.0,   75.0,    "°C",    "Lye supply temperature"),
    "Lye_Concentration":              (17.0,    0.0,   32.0,    "%",     "KOH lye concentration"),
    "Lye_Flow_to_1_Stack":            (51.0,    0.0,  100.0,    "L/h",   "Lye flow rate to stack"),
    "H2_Separator_Level":             (20.0,    0.0,   52.0,    "%",     "H2 separator liquid level"),
    "O2_Separator_Level":             (20.0,    0.0,   52.0,    "%",     "O2 separator liquid level"),
    "LDI_H2_&_O2_Separator":         (0.5,     0.0,    3.0,    "",      "LDI differential level indicator"),
    "Pressure_O2_Separator":          (8.8,     0.0,   16.0,    "bar",   "O2 separator pressure"),
    "H2_Flowrate_Purification_outlet":(47.0,    0.0,  101.0,    "Nm³/h", "H2 flow at purification outlet"),
    "DM_water_condctivity":           (1.7,     0.0,    5.0,    "μS/cm", "Demineralised water conductivity"),
    "DM_water_flow_from_B.L.":        (473.0,   0.0, 1050.0,    "L/h",   "Demineralised water flow from BL"),
}

# ─── FEATURE GROUPS ───────────────────────────────────────────────
GROUPS = {
    " Electrical": [
        "Current_1_stack", "Voltage_1_Stack", "DC_Power_Consumption_1_Stack",
    ],
    " Thermal": [
        "Room_temperature", "H2_side_outlet_temp_1_stack",
        "O2_side_outlet_temp_1_stack", "Lye_Supply_to_Electrolyzer_Temp",
    ],
    " Lye & Chemistry": [
        "Lye_Concentration", "Lye_Flow_to_1_Stack",
    ],
    " Separator": [
        "H2_Separator_Level", "O2_Separator_Level", "LDI_H2_&_O2_Separator",
    ],
    " Flow & Pressure": [
        "Pressure_O2_Separator", "H2_Flowrate_Purification_outlet",
        "DM_water_condctivity", "DM_water_flow_from_B.L.",
    ],
}

# ─── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #FFFBEF 0%, #F0FDF9 40%, #E6FBF4 100%) !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }
section.main > div { padding-top: 0 !important; }
.block-container   { padding: 0 2rem 2rem 2rem !important; max-width: 1400px; }

h1,h2,h3,h4,h5,h6,p,span,label,div { color: #1b9986 !important; }
.stMarkdown p { color: #1b9986 !important; }

/* ── Splash ── */
.splash-wrap {
    display:flex; flex-direction:column; align-items:center;
    justify-content:center; min-height:100vh; gap:0;
    background: linear-gradient(135deg,#FFFBEF 0%,#F0FDF9 50%,#D4F5EE 100%);
}
.splash-logo {
    font-size:110px; line-height:1;
    animation: float 3s ease-in-out infinite;
    filter: drop-shadow(0 8px 32px rgba(27,153,134,0.35));
}
@keyframes float {
    0%,100% { transform:translateY(0px) rotate(-3deg); }
    50%      { transform:translateY(-18px) rotate(3deg); }
}
.splash-title {
    font-family:'Space Grotesk',sans-serif !important;
    font-size:3.4rem; font-weight:800; color:#1b9986 !important;
    letter-spacing:-1px; margin:0.6rem 0 0.1rem;
    text-shadow:0 4px 24px rgba(27,153,134,0.18);
}
.splash-subtitle { font-size:1.1rem; color:#3bb8a0 !important; margin-bottom:2rem; }
.splash-bar-track {
    width:420px; height:6px; background:rgba(27,153,134,0.15);
    border-radius:99px; overflow:hidden; margin-bottom:1rem;
}
.splash-bar-fill {
    height:100%; background:linear-gradient(90deg,#1b9986,#00c9a7,#1b9986);
    background-size:200%; border-radius:99px;
    animation:shimmer 1.5s linear infinite; transition:width 0.3s ease;
}
@keyframes shimmer { 0%{background-position:200% center} 100%{background-position:-200% center} }
.splash-status { font-size:0.85rem; color:#3bb8a0 !important; letter-spacing:0.08em; font-weight:500; }
.splash-line {
    width:320px; height:3px; margin-top:0.5rem;
    background:linear-gradient(90deg,transparent,#1b9986,#00c9a7,transparent);
    border-radius:2px; animation:glow-line 2s ease-in-out infinite alternate;
}
@keyframes glow-line { 0%{opacity:0.4;transform:scaleX(0.7)} 100%{opacity:1;transform:scaleX(1)} }

/* ── Nav ── */
.nav-bar {
    display:flex; align-items:center; justify-content:space-between;
    padding:0.9rem 1.8rem; background:rgba(255,255,255,0.85);
    backdrop-filter:blur(16px); border-bottom:1.5px solid rgba(27,153,134,0.18);
    border-radius:0 0 20px 20px; margin-bottom:1.8rem;
    box-shadow:0 4px 24px rgba(27,153,134,0.08);
    position:sticky; top:0; z-index:100;
}
.nav-logo { font-size:1.4rem; font-weight:800; color:#1b9986 !important; font-family:'Space Grotesk',sans-serif; }
.nav-right { font-size:0.82rem; color:#6ec8bc !important; }

/* ── Hero ── */
.hero {
    text-align:center; padding:1.6rem 0 1.2rem;
    background:linear-gradient(135deg,rgba(27,153,134,0.06),rgba(0,201,167,0.04));
    border-radius:24px; margin-bottom:1.8rem;
    border:1.5px solid rgba(27,153,134,0.12);
}
.hero-icon  { font-size:3rem; margin-bottom:0.3rem; }
.hero-title { font-family:'Space Grotesk',sans-serif !important; font-size:2rem; font-weight:800; color:#1b9986 !important; }
.hero-desc  { font-size:0.95rem; color:#5ab8ab !important; max-width:650px; margin:0.3rem auto 0; }

/* ── Cards ── */
.card {
    background:rgba(255,255,255,0.88); border:1.5px solid rgba(27,153,134,0.15);
    border-radius:20px; padding:1.4rem 1.6rem;
    box-shadow:0 4px 24px rgba(27,153,134,0.07); margin-bottom:1rem;
}
.card:hover { box-shadow:0 8px 40px rgba(27,153,134,0.14); }
.card-title { font-size:1rem; font-weight:700; color:#1b9986 !important; margin-bottom:0.8rem; }

/* ── Widgets ── */
div[data-testid="stNumberInput"] input {
    border:1.5px solid rgba(27,153,134,0.3) !important; border-radius:10px !important;
    color:#1b9986 !important; background:rgba(255,255,255,0.9) !important; font-size:0.88rem !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color:#1b9986 !important; box-shadow:0 0 0 3px rgba(27,153,134,0.15) !important;
}
label[data-testid="stWidgetLabel"] p { color:#1b9986 !important; font-weight:500 !important; }

/* ── Button ── */
div[data-testid="stButton"] > button {
    background:linear-gradient(135deg,#1b9986,#00c9a7) !important;
    color:white !important; border:none !important; border-radius:12px !important;
    font-weight:700 !important; font-size:1rem !important; padding:0.7rem 2.2rem !important;
    box-shadow:0 4px 18px rgba(27,153,134,0.3) !important; transition:all 0.22s !important;
}
div[data-testid="stButton"] > button:hover {
    transform:translateY(-2px) !important; box-shadow:0 8px 28px rgba(27,153,134,0.4) !important;
}

/* ── Results ── */
.result-safe {
    background:linear-gradient(135deg,#d4f5ee,#e6fbf4); border:2px solid #1b9986;
    border-radius:20px; padding:1.8rem; text-align:center; margin:1rem 0;
    box-shadow:0 8px 32px rgba(27,153,134,0.18); animation:fadeInUp 0.5s ease;
}
.result-warn {
    background:linear-gradient(135deg,#fff7e6,#fffbef); border:2px solid #f5a623;
    border-radius:20px; padding:1.8rem; text-align:center; margin:1rem 0;
    box-shadow:0 8px 32px rgba(245,166,35,0.18); animation:fadeInUp 0.5s ease;
}
@keyframes fadeInUp { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
.result-emoji { font-size:3.5rem; margin-bottom:0.5rem; }
.result-title { font-size:1.5rem; font-weight:800; margin-bottom:0.3rem; }
.result-safe .result-title { color:#1b9986 !important; }
.result-warn .result-title { color:#d4890a !important; }
.result-safe .result-desc  { color:#3bb8a0 !important; font-size:0.92rem; }
.result-warn .result-desc  { color:#b8860a !important; font-size:0.92rem; }

/* ── Confidence ── */
.conf-track { background:rgba(27,153,134,0.12); border-radius:99px; height:10px; overflow:hidden; margin:0.5rem 0; }
.conf-fill-safe { height:100%; background:linear-gradient(90deg,#1b9986,#00c9a7); border-radius:99px; transition:width 0.6s ease; }
.conf-fill-warn { height:100%; background:linear-gradient(90deg,#f5a623,#ffd166); border-radius:99px; transition:width 0.6s ease; }

.metric-pill {
    display:inline-block; background:linear-gradient(135deg,#1b9986,#00c9a7);
    color:white !important; font-size:0.78rem; font-weight:700;
    padding:4px 14px; border-radius:99px; margin:3px 4px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#F0FDF9; }
::-webkit-scrollbar-thumb { background:#b2e4da; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────
for key, val in [("loaded", False), ("model", None), ("scaler", None), ("errors", [])]:
    if key not in st.session_state:
        st.session_state[key] = val

# ─── MODEL + SCALER LOADER ────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    errors = []

    # Load model
    if not MODEL_PATH.exists():
        return None, None, [f"❌ maintenance_needed_model.pkl not found at: {MODEL_PATH}"]
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        return None, None, [f"❌ Could not load model: {e}"]

    # Build scaler from CSV — no pre-saved scaler.pkl needed
    if not CSV_PATH.exists():
        return None, None, [f"❌ ACWA_with_target_FINAL.csv not found at: {CSV_PATH}"]
    try:
        df     = pd.read_csv(CSV_PATH)
        X      = df[FEATURE_NAMES]
        scaler = RobustScaler()
        scaler.fit(X)
    except Exception as e:
        return None, None, [f"❌ Could not build scaler from CSV: {e}"]

    return model, scaler, errors


# ─── SPLASH ───────────────────────────────────────────────────────
def show_splash():
    steps = ["Initializing HydroGuard AI...", "Loading sensor intelligence...", "System ready ✓"]
    ph = st.empty()
    for i, step in enumerate(steps):
        pct = int((i + 1) / len(steps) * 100)
        ph.markdown(f"""
        <div class="splash-wrap">
            <div class="splash-logo">⚗️</div>
            <div class="splash-title">HydroGuard AI</div>
            <div class="splash-subtitle">Green Hydrogen Electrolyzer Failure Prediction</div>
            <div class="splash-bar-track"><div class="splash-bar-fill" style="width:{pct}%;"></div></div>
            <div class="splash-line"></div>
            <div class="splash-status">{step}</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(1.0)
    ph.empty()


# ─── MAIN ─────────────────────────────────────────────────────────
def main():
    # Splash once
    if not st.session_state.loaded:
        show_splash()
        model, scaler, errors = load_model_and_scaler()
        st.session_state.model   = model
        st.session_state.scaler  = scaler
        st.session_state.errors  = errors
        st.session_state.loaded  = True
        st.rerun()

    model  = st.session_state.model
    scaler = st.session_state.scaler

    # ── Nav
    st.markdown("""
    <div class="nav-bar">
        <div><span class="nav-logo">⚗️ HydroGuard System</span></div>
        <div class="nav-right">ACWA Power · Green Hydrogen · Failure Prediction System</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-icon">🔬</div>
        <div class="hero-title">Electrolyzer Maintenance Prediction</div>
        <div class="hero-desc">
            Enter real-time sensor values from ACWA Power electrolyzers.
            The ML model predicts whether maintenance is required
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Error banner
    for err in st.session_state.errors:
        st.error(err)


    #  Columns
    left_col, right_col = st.columns([3, 2], gap="large")


    with left_col:

        # Model info card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title"> Active Model</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
            <span class="metric-pill"> Logistic Regression</span>
            <span class="metric-pill">Accuracy 93.34%</span>

        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Feature input cards
        input_values = {}
        for group_name, group_feats in GROUPS.items():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="card-title">{group_name}</div>', unsafe_allow_html=True)
            valid_feats = [f for f in group_feats if f in FEAT_CONFIG]
            rows = [valid_feats[i:i+2] for i in range(0, len(valid_feats), 2)]
            for row in rows:
                cols = st.columns(len(row))
                for col, feat in zip(cols, row):
                    default, mn, mx, unit, tip = FEAT_CONFIG[feat]
                    label = feat.replace("_", " ").strip()
                    label = f"{label} ({unit})" if unit else label
                    with col:
                        input_values[feat] = st.number_input(
                            label,
                            min_value=float(mn), max_value=float(mx),
                            value=float(default),
                            step=round(float((mx - mn) / 200), 4) if mx > mn else 1.0,
                            help=tip, key=feat,
                        )
            st.markdown("</div>", unsafe_allow_html=True)


    with right_col:

        # Dataset stats
        st.markdown("""
        <div class="card">
            <div class="card-title">📊 Dataset Overview</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
                <div style="background:linear-gradient(135deg,rgba(27,153,134,0.08),rgba(0,201,167,0.05));
                    border:1.5px solid rgba(27,153,134,0.18);border-radius:14px;padding:0.9rem;text-align:center;">
                    <div style="font-size:1.6rem;font-weight:800;color:#1b9986;">9,985</div>
                    <div style="font-size:0.75rem;color:#5ab8ab;">Sensor Readings</div>
                </div>
                <div style="background:linear-gradient(135deg,rgba(27,153,134,0.08),rgba(0,201,167,0.05));
                    border:1.5px solid rgba(27,153,134,0.18);border-radius:14px;padding:0.9rem;text-align:center;">
                    <div style="font-size:1.6rem;font-weight:800;color:#1b9986;">16</div>
                    <div style="font-size:0.75rem;color:#5ab8ab;">Model Features</div>
                </div>
                <div style="background:linear-gradient(135deg,rgba(27,153,134,0.08),rgba(0,201,167,0.05));
                    border:1.5px solid rgba(27,153,134,0.18);border-radius:14px;padding:0.9rem;text-align:center;">
                    <div style="font-size:1.6rem;font-weight:800;color:#1b9986;">1</div>
                    <div style="font-size:0.75rem;color:#5ab8ab;">ML Model</div>
                </div>
                <div style="background:linear-gradient(135deg,rgba(27,153,134,0.08),rgba(0,201,167,0.05));
                    border:1.5px solid rgba(27,153,134,0.18);border-radius:14px;padding:0.9rem;text-align:center;">
                    <div style="font-size:1.6rem;font-weight:800;color:#1b9986;">86.8%</div>
                    <div style="font-size:0.75rem;color:#5ab8ab;">Accuracy</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Prediction card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title"> Prediction</div>', unsafe_allow_html=True)

        predict_btn = st.button("Prediction", use_container_width=True)

        if predict_btn:
            if model is None or scaler is None:
                st.error("❌ Model not loaded. Check the debug panel above.")
            else:
                with st.spinner("Analysing sensor data..."):
                    time.sleep(0.4)

                # Build input vector in exact FEATURE_NAMES order
                X_input  = np.array([[input_values[f] for f in FEATURE_NAMES]])
                X_scaled = scaler.transform(X_input)

                pred  = model.predict(X_scaled)[0]
                proba = model.predict_proba(X_scaled)[0]
                conf  = float(max(proba)) * 100

                if pred == 0:
                    fill_cls = "conf-fill-safe"
                    st.markdown("""
                    <div class="result-safe">
                        <div class="result-emoji">✅</div>
                        <div class="result-title">No Maintenance Needed</div>
                        <div class="result-desc">All sensor readings are within safe operating
                        parameters. The electrolyzer is performing normally.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    fill_cls = "conf-fill-warn"
                    st.markdown("""
                    <div class="result-warn">
                        <div class="result-emoji">⚠️</div>
                        <div class="result-title">Maintenance Required</div>
                        <div class="result-desc">One or more sensor readings indicate a potential
                        failure condition. Immediate inspection is recommended.</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="margin-top:1rem;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="font-size:0.82rem;color:#3bb8a0;font-weight:600;">Model Confidence</span>
                        <span style="font-size:0.82rem;color:#1b9986;font-weight:700;">{conf:.1f}%</span>
                    </div>
                    <div class="conf-track">
                        <div class="{fill_cls}" style="width:{conf:.1f}%;"></div>
                    </div>
                </div>
                <div style="display:flex;gap:10px;margin-top:10px;">
                    <div style="flex:1;background:rgba(27,153,134,0.08);border:1.5px solid rgba(27,153,134,0.2);
                        border-radius:12px;padding:0.7rem;text-align:center;">
                        <div style="font-size:1.25rem;font-weight:800;color:#1b9986;">{proba[0]*100:.1f}%</div>
                        <div style="font-size:0.72rem;color:#5ab8ab;">No Maintenance</div>
                    </div>
                    <div style="flex:1;background:rgba(245,166,35,0.08);border:1.5px solid rgba(245,166,35,0.25);
                        border-radius:12px;padding:0.7rem;text-align:center;">
                        <div style="font-size:1.25rem;font-weight:800;color:#d4890a;">{proba[1]*100:.1f}%</div>
                        <div style="font-size:0.72rem;color:#b8860a;">Maintenance Needed</div>
                    </div>
                </div>
                <div style="margin-top:10px;font-size:0.78rem;color:#6ec8bc;text-align:center;">
                    Prediction by <strong style="color:#1b9986;"> Logistic Regression</strong>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # About
        st.markdown("""
        <div class="card" style="margin-top:0.5rem;">
            <div class="card-title">ℹ️ About This System</div>
            <div style="font-size:0.83rem;color:#5ab8ab;line-height:1.7;">
                <strong style="color:#1b9986;">HydroGuard AI</strong> was built for
                <strong style="color:#1b9986;">ACWA Power</strong>'s green hydrogen programme.
                It uses sensor data from alkaline water electrolyzers (AWE) to predict
                maintenance needs before failures occur.<br><br>
                <span style="color:#3bb8a0;"> Tuwaiq Academy · 2026</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Footer
    st.markdown("""
    <div style="text-align:center;padding:2rem 0 1rem;margin-top:1rem;
        border-top:1.5px solid rgba(27,153,134,0.12);">
        <span style="font-size:0.82rem;color:#6ec8bc;">
             HydroGuard System · ACWA Power Green Hydrogen · Tuwaiq Academy 2026 ·
            Developed by Ramah Alharbi · Bayan Alfarsi · Fahad Alghofaili · Ahmed Alshimshir
        </span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

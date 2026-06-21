# app_cardio.py
# Ejecutar con: streamlit run app_cardio.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioRisk · Detección temprana",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Estilos ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Fraunces:ital,wght@0,300;1,300&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .block-container { padding: 2rem 3rem; max-width: 1100px; }

    h1, h2, h3 { font-family: 'Fraunces', serif; font-weight: 300; }

    /* Header */
    .hero { display:flex; align-items:baseline; gap:.75rem; margin-bottom:.25rem; }
    .hero-title { font-family:'Fraunces',serif; font-size:2.6rem; font-weight:300;
                  color:#0f172a; letter-spacing:-.02em; line-height:1; }
    .hero-sub   { font-size:.95rem; color:#64748b; margin-top:.25rem; margin-bottom:2rem; }
    .pulse-dot  { display:inline-block; width:10px; height:10px; border-radius:50%;
                  background:#e11d48; margin-right:6px;
                  animation: pulse 1.6s ease-in-out infinite; }
    @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.7)} }

    /* Sección de inputs */
    .section-label { font-size:.7rem; font-weight:600; letter-spacing:.1em;
                     text-transform:uppercase; color:#94a3b8; margin-bottom:.5rem; }

    /* Resultado */
    .result-card { border-radius:12px; padding:1.5rem 2rem; margin-top:1rem;
                   border-left:5px solid; }
    .result-alto { background:#fff1f2; border-color:#e11d48; }
    .result-bajo { background:#f0fdf4; border-color:#16a34a; }
    .result-title { font-family:'Fraunces',serif; font-size:1.8rem; font-weight:300; }
    .result-prob  { font-size:1rem; color:#475569; margin-top:.25rem; }

    /* Gauge */
    .stProgress > div > div { background: linear-gradient(90deg,#16a34a,#facc15,#e11d48); }

    /* Botón */
    div.stButton > button {
        background:#0f172a; color:white; border:none;
        padding:.6rem 2rem; border-radius:8px; font-weight:600;
        font-size:1rem; width:100%; cursor:pointer;
        transition: background .2s;
    }
    div.stButton > button:hover { background:#1e293b; }

    /* Divider */
    hr { border:none; border-top:1px solid #e2e8f0; margin:2rem 0; }

    /* Footer */
    .footer { font-size:.75rem; color:#94a3b8; text-align:center; margin-top:3rem; }
</style>
""", unsafe_allow_html=True)

# ── Carga del modelo ──────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    modelo   = joblib.load("modelo_cardio_rf.pkl")
    features = joblib.load("features_cardio.pkl")
    return modelo, features

try:
    modelo, features = cargar_modelo()
    modelo_ok = True
except FileNotFoundError:
    modelo_ok = False

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="pulse-dot"></span>
    <span class="hero-title">CardioRisk</span>
</div>
<p class="hero-sub">Detección temprana de enfermedades cardiovasculares · Modelo Random Forest</p>
""", unsafe_allow_html=True)

if not modelo_ok:
    st.error("⚠️ No se encontraron los archivos del modelo (`modelo_cardio_rf.pkl` y `features_cardio.pkl`). "
             "Ejecutá primero `cardio_model_optimizado.py` para entrenar y guardar el modelo.")
    st.stop()

# ── Formulario ────────────────────────────────────────────────────────────────
st.markdown("### Datos del paciente")
st.markdown("Completá los campos y presioná **Evaluar riesgo** para obtener la predicción.")
st.markdown("<hr>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<p class="section-label">Datos personales</p>', unsafe_allow_html=True)
    edad      = st.number_input("Edad (años)", min_value=1, max_value=120, value=45)
    genero    = st.selectbox("Género", ["Femenino", "Masculino"])
    altura    = st.number_input("Altura (cm)", min_value=100, max_value=250, value=168)
    peso      = st.number_input("Peso (kg)", min_value=30, max_value=250, value=75)

with col2:
    st.markdown('<p class="section-label">Signos vitales</p>', unsafe_allow_html=True)
    ap_hi = st.number_input("Presión sistólica (ap_hi)", min_value=60, max_value=300, value=120)
    ap_lo = st.number_input("Presión diastólica (ap_lo)", min_value=40, max_value=200, value=80)
    st.markdown("")
    st.markdown("")
    colesterol = st.selectbox("Colesterol", ["Normal (1)", "Por encima de lo normal (2)", "Muy por encima (3)"])
    glucosa    = st.selectbox("Glucosa", ["Normal (1)", "Por encima de lo normal (2)", "Muy por encima (3)"])

with col3:
    st.markdown('<p class="section-label">Hábitos</p>', unsafe_allow_html=True)
    fuma     = st.radio("¿Fuma?",     ["No", "Sí"], horizontal=True)
    alcohol  = st.radio("¿Consume alcohol?", ["No", "Sí"], horizontal=True)
    actividad = st.radio("¿Hace actividad física?", ["No", "Sí"], horizontal=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Predicción ────────────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    evaluar = st.button("🫀 Evaluar riesgo cardiovascular")

if evaluar:
    # Construir input
    datos = {
        'id':          0,
        'age':         edad * 365,
        'gender':      2 if genero == "Masculino" else 1,
        'height':      altura,
        'weight':      float(peso),
        'ap_hi':       ap_hi,
        'ap_lo':       ap_lo,
        'cholesterol': {"Normal (1)": 1, "Por encima de lo normal (2)": 2, "Muy por encima (3)": 3}[colesterol],
        'gluc':        {"Normal (1)": 1, "Por encima de lo normal (2)": 2, "Muy por encima (3)": 3}[glucosa],
        'smoke':       1 if fuma == "Sí" else 0,
        'alco':        1 if alcohol == "Sí" else 0,
        'active':      1 if actividad == "Sí" else 0,
    }

    df_pac = pd.DataFrame([datos])
    df_pac['bmi']            = df_pac['weight'] / ((df_pac['height'] / 100) ** 2)
    df_pac['pulse_pressure'] = df_pac['ap_hi'] - df_pac['ap_lo']
    df_pac['age_years']      = (df_pac['age'] / 365).astype(int)
    df_pac = df_pac[features]

    pred = modelo.predict(df_pac)[0]
    prob = modelo.predict_proba(df_pac)[0][1]

    # ── Resultado principal ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Resultado")

    res_col1, res_col2 = st.columns([1.2, 1])

    with res_col1:
        if pred == 1:
            st.markdown(f"""
            <div class="result-card result-alto">
                <div class="result-title">🔴 Riesgo ALTO</div>
                <div class="result-prob">El modelo detecta señales de riesgo cardiovascular elevado.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-bajo">
                <div class="result-title">🟢 Riesgo BAJO</div>
                <div class="result-prob">No se detectaron señales significativas de riesgo cardiovascular.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"**Probabilidad de enfermedad cardiovascular: {prob:.1%}**")
        st.progress(float(prob))
        st.caption("0% = sin riesgo · 100% = riesgo máximo")

    # ── Métricas derivadas ──
    with res_col2:
        bmi = peso / ((altura / 100) ** 2)
        pp  = ap_hi - ap_lo

        st.markdown('<p class="section-label">Indicadores calculados</p>', unsafe_allow_html=True)

        imc_cat = ("Bajo peso" if bmi < 18.5 else
                   "Normal" if bmi < 25 else
                   "Sobrepeso" if bmi < 30 else "Obesidad")
        st.metric("IMC", f"{bmi:.1f}", imc_cat)

        pp_cat = "Normal" if 30 <= pp <= 50 else "Elevada" if pp > 50 else "Baja"
        st.metric("Presión de pulso", f"{pp} mmHg", pp_cat)

        st.metric("Edad", f"{edad} años")

    # ── Feature importance ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### ¿Qué factores influyen más en el modelo?")

    rf         = modelo.named_steps['clf']
    importances = rf.feature_importances_
    feat_imp   = pd.Series(importances, index=features).sort_values()

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['#e11d48' if f in ['age', 'age_years', 'bmi', 'ap_hi', 'pulse_pressure']
              else '#94a3b8' for f in feat_imp.index]
    feat_imp.plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel("Importancia relativa", fontsize=9)
    ax.set_title("Importancia de variables", fontsize=11, fontweight='bold')
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(labelsize=9)

    destacado = mpatches.Patch(color='#e11d48', label='Variables clave')
    resto     = mpatches.Patch(color='#94a3b8', label='Resto de variables')
    ax.legend(handles=[destacado, resto], fontsize=8)

    fig.tight_layout()
    st.pyplot(fig)

    # ── Advertencia médica ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption(
        "⚕️ **Aviso:** Esta herramienta es un prototipo académico basado en un modelo de Machine Learning. "
        "No reemplaza el diagnóstico médico profesional. Consultá siempre a un especialista."
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Trabajo de Laboratorio 1 · Data Science & Machine Learning · Random Forest · 2025
</div>
""", unsafe_allow_html=True)
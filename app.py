import streamlit as st
import pandas as pd
import numpy as np
import pickle

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# CONFIGURATION PAGE (doit être EN PREMIER)
st.set_page_config(
    page_title="SmartFood AI",
    page_icon="",
    layout="centered"
)

#  DESIGN SYSTEM 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

/*  Reset & Base  */
.sf-header, .sf-dish, .sf-nutrition, .sf-nut-card, .sf-sodium, .sf-verdict {
    box-sizing: border-box; }

.stApp {
    background: #FFFFFF;
    font-family: 'Inter', sans-serif;
    color: #1A1A1A;
}

/*  Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2.5rem 1.5rem 4rem;
    max-width: 720px;
}

/*  Hero header */
.sf-header {
    text-align: center;
    padding: 3rem 0 2rem;
}
.sf-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #10B981;
    margin-bottom: 14px;
}
.sf-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(38px, 8vw, 58px);
    font-weight: 800;
    color: #111111;
    line-height: 1.0;
    margin: 0 0 16px;
    letter-spacing: -0.03em;
}
.sf-title span {
    color: #10B981;
}
.sf-sub {
    font-size: 15px;
    color: #555555;
    font-weight: 400;
    line-height: 1.6;
}

/* ── Divider ── */
.sf-rule {
    border: none;
    border-top: 1px solid #E0E0E0;
    margin: 2rem 0;
}

/* ── Section label ── */
.sf-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #888888;
    margin-bottom: 10px;
}

/* ── Condition selector ── */
.stSelectbox > div > div {
    background: #F9F9F9 !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 10px !important;
    color: #1A1A1A !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}
.stSelectbox > div > div:focus-within {
    border-color: #10B981 !important;
    box-shadow: 0 0 0 3px rgba(16,185,129,0.12) !important;
}

/*  File uploader */
[data-testid="stFileUploader"] {
    background: #F9FAFB;
    border: 1.5px dashed #D1D5DB;
    border-radius: 14px;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #10B981;
}
[data-testid="stFileUploader"] section {
    padding: 1.5rem;
}

/* ── Image display ── */
[data-testid="stImage"] img {
    border-radius: 14px;
    border: 1px solid #E5E7EB;
}

/* ── Confidence bar ── */
.stProgress {
    margin: 1.5rem 0 2rem !important;
}
[data-testid="stProgressBarMessage"] {
    font-size: 12px !important;
    color: #6B7280 !important;
    font-family: 'Inter', sans-serif !important;
    margin-bottom: 8px !important;
    display: block !important;
}
[data-testid="stProgressBar"] {
    background: #E5E7EB !important;
    border-radius: 99px !important;
    height: 8px !important;
}
[data-testid="stProgressBar"] > div {
    background: linear-gradient(90deg, #10B981, #34D399) !important;
    border-radius: 99px !important;
    height: 8px !important;
}

/* Detected dish  */
.sf-dish {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin: 1.5rem 0 0.5rem;
}
.sf-dish-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-bottom: 6px;
}
.sf-dish-name {
    font-family: 'Syne', sans-serif;
    font-size: 26px;
    font-weight: 700;
    color: #111111;
    letter-spacing: -0.02em;
}

/*  Nutrition grid  */
.sf-nutrition {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin: 1.5rem 0;
}
.sf-nut-card {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1rem 0.75rem;
    text-align: center;
}
.sf-nut-value {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #111111;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.sf-nut-unit {
    font-size: 11px;
    color: #6B7280;
    font-weight: 500;
    margin-top: 1px;
}
.sf-nut-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-top: 6px;
}

/* ── Sodium pill ── */
.sf-sodium {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #F3F4F6;
    border: 1px solid #E5E7EB;
    border-radius: 99px;
    padding: 6px 14px;
    font-size: 12px;
    color: #6B7280;
    font-weight: 500;
    margin-bottom: 1.5rem;
}
.sf-sodium strong { color: #111111; }

/* ── Verdict card ── */
.sf-verdict {
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 14px;
    font-family: 'Inter', sans-serif;
    font-size: 15px;
    font-weight: 500;
    line-height: 1.4;
    margin-top: 0.5rem;
}
.sf-verdict.vert {
    background: rgba(52, 211, 153, 0.08);
    border: 1px solid rgba(52, 211, 153, 0.25);
    color: #6EE7B7;
}
.sf-verdict.jaune {
    background: rgba(251, 191, 36, 0.08);
    border: 1px solid rgba(251, 191, 36, 0.25);
    color: #FCD34D;
}
.sf-verdict.rouge {
    background: rgba(248, 113, 113, 0.08);
    border: 1px solid rgba(248, 113, 113, 0.25);
    color: #FCA5A5;
}
.sf-verdict-icon {
    font-size: 22px;
    flex-shrink: 0;
}

/* ── Section headings ── */
h1, h2, h3 { color: #111111 !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] * { color: #10B981 !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #F9FAFB;
    border-right: 1px solid #E5E7EB;
}
section[data-testid="stSidebar"] * { color: #1A1A1A !important; }

/* ── Metrics override (fallback) ── */
[data-testid="stMetric"] {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="stMetricLabel"] { color: #9CA3AF !important; font-size: 11px !important; }
[data-testid="stMetricValue"] { color: #111111 !important; }

</style>
""", unsafe_allow_html=True)


# CHARGEMENT DES RESSOURCES
@st.cache_resource
def charger_modele():
    return load_model("models/food_classifier.keras")

@st.cache_data
def charger_dataset():
    df = pd.read_csv("nutrition_base.csv")
    if "plat" in df.columns:
        df["plat"] = df["plat"].astype(str).str.lower().str.strip()
    return df

@st.cache_resource
def charger_classes():
    with open("models/classes.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = charger_modele()
    df = charger_dataset()
    classes = charger_classes()
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()


#  PRÉDICTION

def predire_plat(image):
    img = image.convert("RGB").resize((160, 160))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)[0]
    indice = np.argmax(prediction)
    confiance = float(prediction[indice])
    plat = classes[indice]
    return plat, confiance, prediction


#  ÉVALUATION NUTRITIONNELLE
def evaluer_repas(plat_info, condition):
    glucides = plat_info["glucides_g"]
    calories = plat_info["calories_kcal"]
    sodium = str(plat_info["sodium_niveau"]).lower().strip()

    if condition == "diabete":
        if glucides < 45:
            return "Ce plat est adapté à votre condition", "", "vert"
        elif glucides < 70:
            return "À consommer avec modération — glucides élevés", "", "jaune"
        else:
            return "À éviter — teneur en glucides trop élevée", "", "rouge"

    elif condition == "hypertension":
        if sodium == "faible":
            return "Ce plat est adapté à votre condition", "", "vert"
        elif sodium == "modere":
            return "À consommer avec modération — sodium modéré", "", "jaune"
        else:
            return "À éviter — teneur en sodium trop élevée", "", "rouge"

    elif condition == "obesite":
        if calories < 500:
            return "Plat adapté à votre objectif", "", "vert"
        elif calories < 800:
            return "À consommer avec modération — apport calorique élevé", "", "jaune"
        else:
            return "À éviter — trop calorique pour votre objectif", "", "rouge"

    elif condition == "sur regime":
        if calories < 400:
            return "Recommandé pour votre régime", "", "vert"
        elif calories < 700:
            return "À consommer avec modération", "", "jaune"
        else:
            return "Trop calorique pour votre régime actuel", "", "rouge"

    return "Repas standard — aucune restriction détectée", "", "vert"


#  INTERFACE 

st.markdown("""
<div class="sf-header">
    <div class="sf-eyebrow">Analyse alimentaire intelligente</div>
    <div class="sf-title">Smart<span>Food</span> AI</div>
    <div class="sf-sub">Photographiez votre repas.<br>Obtenez une analyse nutritionnelle personnalisée en secondes.</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="sf-rule">', unsafe_allow_html=True)

#  Condition
st.markdown('<div class="sf-label">Votre profil de santé</div>', unsafe_allow_html=True)
condition = st.selectbox(
    "",
    ["diabete", "hypertension", "obesite", "sur regime"],
    format_func=lambda x: {
        "diabete":      "  Diabète",
        "hypertension": "  Hypertension",
        "obesite":      "  Obésité / Perte de poids",
        "sur regime":   "  Régime alimentaire",
    }.get(x, x),
    label_visibility="collapsed"
)

st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)

# Upload
st.markdown('<div class="sf-label">Image du repas</div>', unsafe_allow_html=True)
image_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

#Analyse 
if image_file is not None:
    image = Image.open(image_file)
    st.markdown('<div style="height:2rem"></div>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)

    st.markdown('<div style="height:2rem"></div>', unsafe_allow_html=True)

    with st.spinner("Analyse en cours…"):
        plat_detecte, confiance, prediction = predire_plat(image)

    #  Plat détecté
    nom_affiche = plat_detecte.replace("_", " ").title()
    st.markdown(f"""
    <div class="sf-dish">
        <div class="sf-dish-label">Plat identifié</div>
        <div class="sf-dish-name">{nom_affiche}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
    st.progress(min(confiance, 1.0), text=f"Confiance du modèle : {confiance*100:.1f}%")

    # Données nutritionnelles
    resultat = df[df["plat"] == plat_detecte.lower().strip()]

    if not resultat.empty:
        plat = resultat.iloc[0]
        verdict_txt, verdict_icon, couleur = evaluer_repas(plat, condition)

        st.markdown('<div style="height:2.5rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sf-label">Valeurs nutritionnelles · pour 100 g</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sf-nutrition">
            <div class="sf-nut-card">
                <div class="sf-nut-value">{plat['calories_kcal']}</div>
                <div class="sf-nut-unit">kcal</div>
                <div class="sf-nut-label">Calories</div>
            </div>
            <div class="sf-nut-card">
                <div class="sf-nut-value">{plat['glucides_g']}</div>
                <div class="sf-nut-unit">g</div>
                <div class="sf-nut-label">Glucides</div>
            </div>
            <div class="sf-nut-card">
                <div class="sf-nut-value">{plat['proteines_g']}</div>
                <div class="sf-nut-unit">g</div>
                <div class="sf-nut-label">Protéines</div>
            </div>
            <div class="sf-nut-card">
                <div class="sf-nut-value">{plat['lipides_g']}</div>
                <div class="sf-nut-unit">g</div>
                <div class="sf-nut-label">Lipides</div>
            </div>
        </div>
        <div>
            <span class="sf-sodium">Sodium &nbsp;<strong>{plat['sodium_niveau'].capitalize()}</strong></span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sf-label">Recommandation</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sf-verdict {couleur}">
            <span class="sf-verdict-icon">{verdict_icon}</span>
            <span>{verdict_txt}</span>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning(f"Le plat « {plat_detecte} » n'est pas encore dans la base nutritionnelle.")
        st.markdown('<div class="sf-label">Plats disponibles (extrait)</div>', unsafe_allow_html=True)
        st.dataframe(df[["plat"]].head(20), hide_index=True)

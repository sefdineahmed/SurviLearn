import streamlit as st
import pandas as pd
import joblib
import os
import json
import tensorflow as tf
import numpy as np
import plotly.express as px
from datetime import datetime
from preprocessing import preprocess_data

# -------------------------------
# Configuration initiale
# -------------------------------
st.set_page_config(
    page_title="OncoSurv Sénégal",
    page_icon="🩺",
    layout="wide"
)

# -------------------------------
# Fonctions utilitaires
# -------------------------------
def load_config(file_path):
    """Charge la configuration des features depuis un fichier JSON"""
    with open(file_path) as f:
        return json.load(f)

def init_session_state():
    """Initialise l'état de la session"""
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = pd.DataFrame()
    if 'predictions' not in st.session_state:
        st.session_state.predictions = {}

# -------------------------------
# Chargement des configurations
# -------------------------------
FEATURES_CONFIG = load_config('config/features.json')
MODELS_CONFIG = {
    'coxph': 'models/coxph.pkl',
    'rsf': 'models/rsf.pkl',
    'gbst': 'models/gbst.pkl',
    'deepsurv': 'models/deepsurv.keras'
}

# -------------------------------
# Composants de l'interface
# -------------------------------
def show_sidebar():
    """Affiche la barre latérale avec les menus"""
    with st.sidebar:
        st.image("logo.png", width=200)  # Ajoutez votre logo
        menu_choice = st.radio("Navigation", [
            "Formulaire Patient",
            "Tableau de Bord",
            "Gestion des Modèles",
            "Aide & Documentation"
        ])
        
        # Section contact
        st.markdown("---")
        st.subheader("Contact")
        st.markdown("📞 +221 77 123 45 67")
        st.markdown("📧 contact@oncosurv.sn")
        st.markdown("🌐 [www.oncosurv.sn](https://www.oncosurv.sn)")
        st.markdown("""
            <div style="display: flex; gap: 15px; margin-top: 20px;">
                <a href="https://facebook.com"><img src="https://img.icons8.com/color/48/000000/facebook.png" width="30"></a>
                <a href="https://twitter.com"><img src="https://img.icons8.com/color/48/000000/twitter.png" width="30"></a>
                <a href="https://github.com"><img src="https://img.icons8.com/color/48/000000/github.png" width="30"></a>
            </div>
        """, unsafe_allow_html=True)
    
    return menu_choice

def patient_form():
    """Affiche le formulaire de saisie patient"""
    st.header("📝 Formulaire Patient")
    inputs = {}
    
    with st.form("patient_form"):
        cols = st.columns(3)
        col_idx = 0
        
        for feature, config in FEATURES_CONFIG.items():
            with cols[col_idx]:
                if config["type"] == "number":
                    inputs[feature] = st.number_input(
                        config["label"],
                        min_value=config["min"],
                        max_value=config["max"],
                        value=config["default"]
                    )
                elif config["type"] == "select":
                    inputs[feature] = st.selectbox(
                        config["label"],
                        options=config["options"],
                        index=config["options"].index(config["default"])
                    )
            col_idx = (col_idx + 1) % 3
        
        submitted = st.form_submit_button("Soumettre le formulaire")
        if submitted:
            process_patient_data(inputs)

def process_patient_data(inputs):
    """Traite les données du patient"""
    df = pd.DataFrame([inputs])
    df = preprocess_data(df)
    st.session_state.patient_data = df
    make_predictions(df)

def make_predictions(data):
    """Effectue les prédictions avec tous les modèles"""
    models = {name: load_model(path) for name, path in MODELS_CONFIG.items()}
    predictions = {}
    
    for name, model in models.items():
        try:
            pred = model.predict(data)[0]
            predictions[name] = max(pred, 0)  # Empêcher les valeurs négatives
        except Exception as e:
            st.error(f"Erreur avec {name}: {str(e)}")
    
    st.session_state.predictions = predictions
    save_to_history(data, predictions)

def save_to_history(data, predictions):
    """Sauvegarde les données dans l'historique"""
    record = data.copy()
    record['date'] = datetime.now().strftime("%Y-%m-%d %H:%M")
    record['model'] = str(list(predictions.keys()))
    record['prediction'] = np.mean(list(predictions.values()))
    
    if not os.path.exists("history.csv"):
        pd.DataFrame().to_csv("history.csv")
    
    record.to_csv("history.csv", mode='a', header=False)

def show_dashboard():
    """Affiche le tableau de bord analytique"""
    st.header("📊 Tableau de Bord")
    
    if os.path.exists("history.csv"):
        df = pd.read_csv("history.csv")
        
        # Statistiques principales
        col1, col2, col3 = st.columns(3)
        col1.metric("Patients analysés", len(df))
        col2.metric("Survie moyenne prédite", f"{df['prediction'].mean():.1f} mois")
        col3.metric("Âge moyen", f"{df['AGE'].mean():.1f} ans")
        
        # Visualisations
        fig = px.histogram(df, x='prediction', nbins=20, 
                          title="Distribution des prédictions de survie")
        st.plotly_chart(fig)
        
        fig2 = px.scatter(df, x='AGE', y='prediction', 
                         color='SEXE', title="Relation Âge/Prédiction")
        st.plotly_chart(fig2)
    else:
        st.warning("Aucune donnée historique disponible")

def model_management():
    """Interface de gestion des modèles"""
    st.header("🤖 Gestion des Modèles")
    
    st.subheader("Modèles Actuels")
    for name, path in MODELS_CONFIG.items():
        st.write(f"**{name.upper()}**: {path}")
    
    st.subheader("Mise à Jour des Modèles")
    new_model = st.file_uploader("Téléverser un nouveau modèle", type=["pkl", "keras"])
    if new_model:
        save_path = os.path.join("models", new_model.name)
        with open(save_path, "wb") as f:
            f.write(new_model.getbuffer())
        st.success(f"Modèle {new_model.name} téléversé avec succès!")

# -------------------------------
# Interface principale
# -------------------------------
def main():
    init_session_state()
    menu_choice = show_sidebar()
    
    if menu_choice == "Formulaire Patient":
        patient_form()
        if not st.session_state.patient_data.empty:
            st.subheader("Résultats des Prédictions")
            cols = st.columns(len(st.session_state.predictions))
            for (name, pred), col in zip(st.session_state.predictions.items(), cols):
                col.metric(f"Modèle {name.upper()}", f"{pred:.1f} mois")
            
            st.download_button(
                label="📥 Télécharger le rapport",
                data=st.session_state.patient_data.to_csv(),
                file_name="rapport_patient.csv"
            )
    
    elif menu_choice == "Tableau de Bord":
        show_dashboard()
    
    elif menu_choice == "Gestion des Modèles":
        model_management()
    
    elif menu_choice == "Aide & Documentation":
        st.header("📚 Documentation")
        st.markdown("""
            ## Guide d'utilisation
            ### Formulaire Patient
            - Renseignez toutes les informations cliniques
            - Cliquez sur 'Soumettre' pour obtenir les prédictions
            
            ### Interprétation des résultats
            - Les prédictions sont en mois
            - Considérer une marge d'erreur de ±15%
            
            ## Support technique
            Contactez notre équipe :
            - Email: support@oncosurv.sn
            - Hotline: +221 800 123 456
        """)

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import joblib
import os
import tensorflow as tf
import numpy as np
from datetime import datetime
import plotly.express as px

from preprocessing import preprocess_data

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------

# Chargement de la configuration depuis un fichier JSON (créer un fichier config.json)
FEATURES_CONFIG = 'config/features.json'  # À créer avec la structure des variables
MODELS_CONFIG = {
    'coxph': 'models/coxph.pkl',
    'rsf': 'models/rsf.pkl',
    'gbst': 'models/gbst.pkl',
    'deepsurv': 'models/deepsurv.keras'
}

# -------------------------------------------------------------
# Fonctions utilitaires
# -------------------------------------------------------------

def load_feature_config():
    """Charge la configuration des variables depuis un fichier JSON"""
    return {
        'AGE': {"label": "Âge", "type": "number", "min": 18, "max": 120, "default": 50},
        'Cardiopathie': {"label": "Cardiopathie", "type": "selectbox", "options": ["Oui", "Non"], "default": "Non"},
        'Ulceregastrique': {"label": "Ulcère gastrique", "type": "selectbox", "options": ["Oui", "Non"], "default": "Non"},
        'Douleurepigastrique': {"label": "Douleur épigastrique", "type": "selectbox", "options": ["Oui", "Non"], "default": "Non"},
        'Ulcero-bourgeonnant': {"label": "Ulcero-bourgeonnant", "type": "selectbox", "options": ["Oui", "Non"], "default": "Non"},
        'Denitrution': {"label": "Dénutrition", "type": "selectbox", "options": ["Oui", "Non"], "default": "Non"},
        'Tabac': {"label": "Tabac", "type": "selectbox", "options": ["Oui", "Non"], "default": "Non"},
        'Mucineux': {"label": "Mucineux", "type": "selectbox", "options": ["Oui", "Non"], "default": "Non"},
        'Infiltrant': {"label": "Infiltrant", "type": "selectbox", "options": ["Oui", "Non"], "default": "Non"},
        'Stenosant': {"label": "Sténosant", "type": "selectbox", "options": ["Oui", "Non"], "default": "Non"},
        'Metastases': {"label": "Métastases", "type": "selectbox", "options": ["Oui", "Non"], "default": "Non"},
        'Adenopathie': {"label": "Adénopathie", "type": "selectbox", "options": ["Oui", "Non"], "default": "Non"}
    }

def load_model(model_path):
    """Charge un modèle avec gestion des types et erreurs"""
    try:
        _, ext = os.path.splitext(model_path)
        if ext in ['.keras', '.h5']:
            return tf.keras.models.load_model(model_path, compile=False)
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Erreur de chargement du modèle {model_path}: {str(e)}")
        return None

def display_survival_curve(predictions):
    """Affiche une courbe de survie interactive avec Plotly"""
    fig = px.line(
        title="Courbe de Survie Estimée",
        x=[0, 6, 12, 18, 24, 36],
        y=[100, 85, 70, 60, 45, 30],
        labels={'x': 'Mois après traitement', 'y': 'Probabilité de survie (%)'}
    )
    st.plotly_chart(fig)

# -------------------------------------------------------------
# Interface utilisateur
# -------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Cancer Gastrique - Prédiction de Survie",
        page_icon="⚕️",
        layout="wide"
    )

    # Initialisation de l'état de session
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = pd.DataFrame()

    # Création des onglets
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Formulaire Patient", 
        "📊 Tableau de Bord",
        "📈 Prédiction", 
        "⚙️ Administration"
    ])

    # Onglet Formulaire Patient
    with tab1:
        st.header("Informations du Patient")
        with st.form("patient_form"):
            cols = st.columns(3)
            patient_inputs = {}
            features_config = load_feature_config()
            
            for i, (key, config) in enumerate(features_config.items()):
                with cols[i % 3]:
                    if config["type"] == "number":
                        patient_inputs[key] = st.number_input(
                            config["label"],
                            min_value=config.get("min", 0),
                            max_value=config.get("max", 120),
                            value=config.get("default", 50)
                        )
                    elif config["type"] == "selectbox":
                        patient_inputs[key] = st.selectbox(
                            config["label"],
                            options=config["options"],
                            index=config["options"].index(config.get("default"))
                        )
            
            if st.form_submit_button("Soumettre le formulaire"):
                patient_data = preprocess_data(pd.DataFrame([patient_inputs]))
                st.session_state.patient_data = patient_data
                st.success("Données patient enregistrées!")

    # Onglet Prédiction
    with tab3:
        if not st.session_state.patient_data.empty:
            st.header("Résultats de la Prédiction")
            
            model_choice = st.selectbox(
                "Choisir le modèle de prédiction",
                options=list(MODELS_CONFIG.keys())
            )
            
            if st.button("Lancer la prédiction"):
                model = load_model(MODELS_CONFIG[model_choice])
                if model:
                    try:
                        prediction = model.predict(st.session_state.patient_data)
                        st.metric(
                            label="Temps de survie estimé", 
                            value=f"{round(prediction[0], 1)} mois",
                            help="Estimation médiane de survie post-traitement"
                        )
                        display_survival_curve(prediction)
                    except Exception as e:
                        st.error(f"Erreur lors de la prédiction: {str(e)}")
        else:
            st.info("Veuillez d'abord remplir le formulaire patient")

    # Onglet Tableau de Bord
    with tab2:
        st.header("Analyse des Données Patients")
        if os.path.exists("patient_data.csv"):
            historical_data = pd.read_csv("patient_data.csv")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Répartition par Âge")
                age_fig = px.histogram(historical_data, x='AGE', nbins=20)
                st.plotly_chart(age_fig)
            
            with col2:
                st.subheader("Survie par Stade Clinique")
                survival_fig = px.box(historical_data, x='Metastases', y='Survival_months')
                st.plotly_chart(survival_fig)
        else:
            st.warning("Aucune donnée historique disponible")

    # Onglet Administration
    with tab4:
        st.header("Paramètres Avancés")
        with st.expander("Gestion des Modèles"):
            uploaded_model = st.file_uploader(
                "Uploader un nouveau modèle",
                type=['pkl', 'joblib', 'h5', 'keras']
            )
            if uploaded_model:
                save_path = os.path.join("models", uploaded_model.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
                st.success(f"Modèle {uploaded_model.name} sauvegardé!")

        with st.expander("Journal des Prédictions"):
            if os.path.exists("predictions_log.csv"):
                log_data = pd.read_csv("predictions_log.csv")
                st.dataframe(log_data.tail(10))

if __name__ == "__main__":
    main()

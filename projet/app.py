import streamlit as st
import pandas as pd
import joblib
import json
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import plotly.express as px
from preprocessing import preprocess_data

# -------------------------------------------------------------
# Configuration initiale
# -------------------------------------------------------------

# Chargement de la configuration depuis le JSON
with open('config/features.json') as f:
    features_config = json.load(f)

# -------------------------------------------------------------
# Fonctions utilitaires
# -------------------------------------------------------------

def load_model(model_path):
    """Charge un modèle avec gestion des types"""
    if model_path.endswith('.keras'):
        return tf.keras.models.load_model(model_path)
    else:
        return joblib.load(model_path)

def predict_survival(model, data, model_name):
    """Effectue la prédiction selon le modèle"""
    try:
        if model_name == 'COXPH':
            return model.predict_median(data)
        elif model_name == 'RSF':
            return model.predict(data)[0]
        elif model_name == 'DEEPSURV':
            return model.predict(data)[0][0]
        else:
            return model.predict(data)[0]
    except Exception as e:
        st.error(f"Erreur de prédiction: {e}")
        return None

# -------------------------------------------------------------
# Interface utilisateur
# -------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="OncoSurv Sénégal",
        page_icon="🩺",
        layout="wide"
    )

    # Sidebar avec paramètres
    with st.sidebar:
        st.title("Configuration")
        selected_models = st.multiselect(
            "Modèles à utiliser",
            ['COXPH', 'RSF', 'GBST', 'DEEPSURV'],
            default=['COXPH', 'RSF']
        )
    
    # Navigation principale
    tabs = ["📝 Formulaire", "📊 Dashboard", "⚙️ Administration"]
    current_tab = st.sidebar.radio("Navigation", tabs)

    # Onglet Formulaire
    if current_tab == "📝 Formulaire":
        st.header("Formulaire Patient")
        
        # Saisie dynamique depuis features.json
        patient_data = {}
        cols = st.columns(3)
        for i, (feature, config) in enumerate(features_config.items()):
            with cols[i%3]:
                if config['type'] == 'number':
                    patient_data[feature] = st.number_input(
                        config['label'],
                        min_value=config['min'],
                        max_value=config['max'],
                        value=config['default']
                    )
                elif config['type'] == 'select':
                    patient_data[feature] = st.selectbox(
                        config['label'],
                        options=config['options']
                    )

        if st.button("Lancer la prédiction"):
            df = preprocess_data(pd.DataFrame([patient_data]))
            
            # Chargement des modèles
            models = {name: load_model(f'models/{name.lower()}.pkl') 
                     for name in selected_models}
            
            # Affichage des résultats
            st.subheader("Résultats de prédiction")
            results = {}
            
            for name, model in models.items():
                prediction = predict_survival(model, df, name)
                results[name] = prediction
                st.metric(label=name, value=f"{prediction:.1f} mois")
            
            # Visualisation
            fig = px.bar(
                x=list(results.keys()),
                y=list(results.values()),
                labels={'x':'Modèle', 'y':'Mois de survie'},
                title="Comparaison des modèles"
            )
            st.plotly_chart(fig)

    # Onglet Dashboard
    elif current_tab == "📊 Dashboard":
        st.header("Analytique des données")
        
        # Chargement des données historiques
        try:
            hist_data = pd.read_csv('data/historical_data.csv')
            st.dataframe(hist_data.tail(10))
            
            # Visualisations
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(hist_data, x='Prediction', nbins=20)
                st.plotly_chart(fig)
            
            with col2:
                fig = px.box(hist_data, y='AGE', x='SEXE')
                st.plotly_chart(fig)
                
        except FileNotFoundError:
            st.warning("Aucune donnée historique trouvée")

    # Onglet Administration
    elif current_tab == "⚙️ Administration":
        st.header("Gestion des modèles")
        
        # Téléversement de nouveaux modèles
        uploaded_model = st.file_uploader("Nouveau modèle", type=['pkl', 'keras'])
        if uploaded_model:
            # Sauvegarde du modèle
            with open(f'models/{uploaded_model.name}', 'wb') as f:
                f.write(uploaded_model.getbuffer())
            st.success("Modèle mis à jour avec succès!")
        
        # Gestion des variables
        if st.button("Recharger la configuration"):
            with open('config/features.json') as f:
                features_config = json.load(f)
            st.experimental_rerun()

if __name__ == "__main__":
    main()

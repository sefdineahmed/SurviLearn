import streamlit as st
import pandas as pd
import joblib
import os
import json
import numpy as np
import plotly.express as px
import tensorflow as tf
from datetime import datetime
from preprocessing import preprocess_data

# -------------------------------------------------------------
# Configuration initiale
# -------------------------------------------------------------
st.set_page_config(
    page_title="Survie Cancer Gastrique",
    page_icon="🩺",
    layout="wide"
)

# -------------------------------------------------------------
# Fonctions utilitaires
# -------------------------------------------------------------
def load_model(model_path):
    _, ext = os.path.splitext(model_path)
    if ext in ['.keras', '.h5']:
        return tf.keras.models.load_model(model_path, compile=False)
    else:
        return joblib.load(model_path)

def predict_survival(model, data):
    if hasattr(model, "predict_median"):
        return model.predict_median(data)
    elif hasattr(model, "predict"):
        pred = model.predict(data)
        return pred[0] if isinstance(pred, np.ndarray) else pred
    else:
        raise ValueError("Modèle non supporté")

def save_prediction(patient_data, predictions):
    record = {
        **patient_data,
        **predictions,
        'timestamp': datetime.now().isoformat()
    }
    df = pd.DataFrame([record])
    if not os.path.exists('data/patient_records.csv'):
        df.to_csv('data/patient_records.csv', index=False)
    else:
        df.to_csv('data/patient_records.csv', mode='a', header=False, index=False)

# -------------------------------------------------------------
# Chargement des configurations
# -------------------------------------------------------------
with open('config/features.json') as f:
    FEATURES = json.load(f)

MODELS = {
    'CoxPH': 'models/coxph.pkl',
    'RSF': 'models/rsf.pkl',
    'GBST': 'models/gbst.pkl',
    'DeepSurv': 'models/deepsurv.keras'
}

# -------------------------------------------------------------
# Interface utilisateur
# -------------------------------------------------------------
def main():
    # Initialisation de l'état de session
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None

    # Création des onglets
    tabs = st.tabs(["📝 Formulaire Patient", "📊 Résultats", "📈 Dashboard", "ℹ️ Aide"])
    
    # Onglet 1: Formulaire Patient
    with tabs[0]:
        st.header("Informations du Patient")
        patient_data = {}
        
        cols = st.columns(3)
        for i, feature in enumerate(FEATURES):
            with cols[i % 3]:
                config = FEATURES[feature]
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
                        options=config['options'],
                        index=config['options'].index(config['default'])
                    )

        if st.button("Prédire la survie", type="primary"):
            try:
                # Prétraitement
                df = pd.DataFrame([patient_data])
                processed_data = preprocess_data(df)
                
                # Chargement des modèles
                models = {name: load_model(path) for name, path in MODELS.items()}
                
                # Prédictions
                predictions = {}
                for name, model in models.items():
                    pred = predict_survival(model, processed_data)
                    predictions[name] = max(round(float(pred), 1), 0)
                
                st.session_state.predictions = predictions
                save_prediction(patient_data, predictions)
                
            except Exception as e:
                st.error(f"Erreur lors de la prédiction: {str(e)}")

    # Onglet 2: Résultats
    with tabs[1]:
        st.header("Résultats des Prédictions")
        
        if st.session_state.predictions:
            cols = st.columns(len(MODELS))
            for (name, pred), col in zip(st.session_state.predictions.items(), cols):
                col.metric(
                    label=name,
                    value=f"{pred} mois",
                    help="Temps de survie médian prédit"
                )
            
            st.subheader("Interprétation")
            st.progress(min(st.session_state.predictions['CoxPH']/36, 1.0))
            st.write("""
            **Guide d'interprétation:**
            - <12 mois: Survie courte - Prise en charge palliative recommandée
            - 12-36 mois: Survie intermédiaire - Surveillance rapprochée
            - >36 mois: Survie longue - Suivi standard
            """)
        else:
            st.info("Soumettre le formulaire patient pour voir les prédictions")

    # Onglet 3: Dashboard
    with tabs[2]:
        st.header("Analyse des Données Historiques")
        
        if os.path.exists('data/patient_records.csv'):
            df = pd.read_csv('data/patient_records.csv')
            
            # Filtres
            age_range = st.slider("Filtrer par âge", 
                min_value=int(df['AGE'].min()), 
                max_value=int(df['AGE'].max()),
                value=(30, 80))
            
            filtered_df = df[(df['AGE'] >= age_range[0]) & (df['AGE'] <= age_range[1])]
            
            # Visualisations
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(filtered_df, x='CoxPH', nbins=20, 
                                 title="Distribution des prédictions CoxPH")
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                fig = px.scatter(filtered_df, x='AGE', y='CoxPH', color='SEXE',
                               title="Relation Âge/Survie par Sexe")
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Données Brutes")
            st.dataframe(filtered_df.sort_values('timestamp', ascending=False), 
                        height=300)
        else:
            st.warning("Aucune donnée historique disponible")

    # Onglet 4: Aide
    with tabs[3]:
        st.header("Guide d'Utilisation")
        
        st.markdown("""
        ### Documentation de l'Application
        
        **1. Formulaire Patient**
        - Renseigner toutes les informations cliniques disponibles
        - Les champs marqués d'un * sont obligatoires
        - Cliquer sur 'Prédire la survie' pour lancer l'analyse
        
        **2. Résultats**
        - Comparaison des prédictions entre différents modèles
        - Interprétation clinique des résultats
        - Recommandations thérapeutiques basées sur le consensus
        
        **3. Dashboard**
       - Analyse populationnelle des données historiques
        - Filtres interactifs pour l'exploration des données
        - Export des données au format CSV
        """)

if __name__ == "__main__":
    main()

# preprocessing.py
from sklearn.preprocessing import StandardScaler

def preprocess_data(raw_data):
    # Conversion des variables catégorielles
    data = raw_data.copy()
    #data['SEXE'] = data['SEXE'].map({'Homme': 0, 'Femme': 1})
    
    # Normalisation des variables continues
    scaler = StandardScaler()
    numerical_features = ['AGE']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    
    return data

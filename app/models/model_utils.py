from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import skfuzzy as fuzz
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score
from datetime import datetime

# Obtener la ruta al directorio actual del script
current_dir = Path(__file__).parent
# Subir dos niveles (porque este archivo está en app/models/) y luego entrar a data/
dataset_path = current_dir.parent.parent / 'data' / 'FGR_dataset (1).xlsx'
MODELS_DIR = current_dir  # Ahora usamos la carpeta models/ donde está este archivo

def load_dataset():
    """Carga y prepara el dataset"""
    if not dataset_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {dataset_path}")
    
    df = pd.read_excel(dataset_path)
    X = df.iloc[:, :-1]  # Todas las columnas excepto la última
    y = df.iloc[:, -1]   # Última columna (C31)
    return X, y

def get_model_path(model_name):
    """Devuelve la ruta completa para un modelo dado"""
    return MODELS_DIR / f"{model_name}_model.pkl"

def retrain_model(model_type, train_size=0.8, random_state=42):
    """
    Reentrena un modelo específico y lo guarda
    
    Args:
        model_type (str): Tipo de modelo ('logistic', 'neural', 'svm', 'fuzzy')
        train_size (float): Porcentaje de datos para entrenamiento (0-1)
        random_state (int): Semilla aleatoria para reproducibilidad
    
    Returns:
        tuple: (success: bool, message: str, metrics: dict)
    """
    try:
        # 1. Cargar datos
        X, y = load_dataset()
        
        # 2. Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            train_size=train_size, 
            random_state=random_state
        )
        
        # 3. Estandarizar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 4. Cargar o crear modelo
        model = None
        model_path = get_model_path(model_type)
        
        # Diccionario de modelos disponibles
        models = {
            'logistic': {
                'constructor': LogisticRegression(max_iter=1000),
                'name': 'Regresión Logística'
            },
            'neural': {
                'constructor': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42),
                'name': 'Red Neuronal'
            },
            'svm': {
                'constructor': SVC(probability=True, random_state=42),
                'name': 'Máquina de Soporte Vectorial'
            },
            'fuzzy': {
                'name': 'Mapa Difuso'
            }
        }
        
        if model_type not in models:
            return False, f"Tipo de modelo no soportado: {model_type}", {}
        
        # Caso especial para modelo difuso
        if model_type == 'fuzzy':
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                X_train_scaled.T, 2, 2, error=0.005, maxiter=1000, init=None)
            
            fuzzy_model = {
                'cntr': cntr,
                'u': u,
                'fpc': fpc,
                'last_retrained': datetime.now().isoformat()
            }
            joblib.dump(fuzzy_model, model_path)
            return True, f"Modelo difuso reentrenado exitosamente", {'fpc': fpc}
        
        # Para otros modelos
        if model_path.exists():
            model = joblib.load(model_path)
        else:
            model = models[model_type]['constructor']
        
        # 5. Reentrenar
        model.fit(X_train_scaled, y_train)
        
        # 6. Calcular métricas
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 7. Guardar modelo y scaler
        model.last_retrained = datetime.now().isoformat()
        joblib.dump(model, model_path)
        joblib.dump(scaler, MODELS_DIR / 'scaler.save')
        
        metrics = {
            'accuracy': accuracy,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'last_retrained': model.last_retrained
        }
        
        return True, f"{models[model_type]['name']} reentrenado exitosamente. Precisión: {accuracy:.2%}", metrics
    
    except Exception as e:
        return False, f"Error al reentrenar {model_type}: {str(e)}", {}

def train_and_save_models():
    """Función original para entrenar todos los modelos inicialmente"""
    X, y = load_dataset()
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Estandarizar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Guardar el scaler
    joblib.dump(scaler, MODELS_DIR / 'scaler.save')
    
    # Entrenar y guardar todos los modelos
    retrain_model('logistic', train_size=0.8, random_state=42)
    retrain_model('neural', train_size=0.8, random_state=42)
    retrain_model('svm', train_size=0.8, random_state=42)
    retrain_model('fuzzy', train_size=0.8, random_state=42)
    
    print("Todos los modelos han sido entrenados y guardados exitosamente!")

if __name__ == '__main__':
    train_and_save_models()
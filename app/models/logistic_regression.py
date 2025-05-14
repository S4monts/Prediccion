import joblib
import numpy as np

class LogisticRegressionModel:
    def __init__(self):
        self.model = joblib.load('app/models/logistic_regression_model.pkl')
        self.scaler = joblib.load('app/models/scaler.save')
    
    def predict(self, input_data):
        # Escalar los datos de entrada
        scaled_data = self.scaler.transform([input_data])
        # Hacer la predicción
        prediction = self.model.predict(scaled_data)
        probabilities = self.model.predict_proba(scaled_data)
        return prediction[0], probabilities[0]
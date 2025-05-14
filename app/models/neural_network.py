import joblib
import numpy as np

class NeuralNetworkModel:
    def __init__(self):
        self.model = joblib.load('app/models/neural_network_model.pkl')
        self.scaler = joblib.load('app/models/scaler.save')
    
    def predict(self, input_data):
        scaled_data = self.scaler.transform([input_data])
        prediction = self.model.predict(scaled_data)
        probabilities = self.model.predict_proba(scaled_data)
        return prediction[0], probabilities[0]
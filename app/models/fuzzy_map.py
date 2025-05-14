import joblib
import numpy as np
import skfuzzy as fuzz 

class FuzzyMapModel:
    def __init__(self):
        fuzzy_data = joblib.load('app/models/fuzzy_model.pkl')
        self.cntr = fuzzy_data['cntr']
        self.u = fuzzy_data['u']
        self.fpc = fuzzy_data['fpc']
        self.scaler = joblib.load('app/models/scaler.save')
    
    def predict(self, input_data):
        scaled_data = self.scaler.transform([input_data])
        
        # Calcular la pertenencia a los clusters
        u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
            scaled_data.T, self.cntr, 2, error=0.005, maxiter=1000)
        
        # Asumimos que el cluster con mayor pertenencia es la predicción
        prediction = np.argmax(u, axis=0)[0]
        probabilities = u[:, 0]
        return prediction, probabilities
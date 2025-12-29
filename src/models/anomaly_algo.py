from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

class BearingAnomalyDetector:
    def __init__(self, contamination=0.05):
        """
        args:
            contamination (float): expected proportion of outliers in the dataset.
        """
        self.scaler = StandardScaler()
        self.model = IsolationForest(contamination=contamination, random_state=42)
        
    def train(self, feature_matrix):
        """
        trains the isolation forest on the feature matrix.
        """
        print("scaling data...")
        scaled_data = self.scaler.fit_transform(feature_matrix)
        
        print("fitting isolation forest...")
        self.model.fit(scaled_data)
        print("training complete.")
        
    def predict(self, feature_matrix):
        """
        returns -1 for anomaly, 1 for normal.
        """
        scaled_data = self.scaler.transform(feature_matrix)
        return self.model.predict(scaled_data)
    
    def get_anomaly_score(self, feature_matrix):
        """
        lower scores = more abnormal.
        """
        scaled_data = self.scaler.transform(feature_matrix)
        return self.model.decision_function(scaled_data)

    def save_model(self, path='saved_models/iso_forest.pkl'):
        joblib.dump(self.model, path)
        joblib.dump(self.scaler, path.replace('.pkl', '_scaler.pkl'))

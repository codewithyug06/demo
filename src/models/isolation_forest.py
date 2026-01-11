from sklearn.ensemble import IsolationForest
import pandas as pd

class AnomalyDetector:
    def __init__(self, contamination=0.01):
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def detect_spikes(self, df):
        """
        Expects a DataFrame with 'total_updates' and 'velocity'.
        Returns the DF with an 'is_anomaly' flag.
        """
        # Feature set for detection
        features = df[['total_updates', 'velocity']].fillna(0)
        
        # -1 is anomaly, 1 is normal
        df['anomaly_score'] = self.model.fit_predict(features)
        df['is_anomaly'] = df['anomaly_score'].apply(lambda x: True if x == -1 else False)
        
        return df
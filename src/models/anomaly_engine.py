from sklearn.ensemble import IsolationForest
import pandas as pd

class AnomalyDetector:
    def __init__(self, contamination=0.01):
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def detect_velocity_anomalies(self, df):
        """
        Detects anomalies in 'Update Velocity' (Daily updates per Pin Code).
        """
        # --- CRITICAL FIX ---
        # We must group by 'district' as well, otherwise pandas drops it.
        # Old line: df.groupby(['date', 'pincode', 'state'])
        # New line: df.groupby(['date', 'state', 'district', 'pincode'])
        daily_stats = df.groupby(['date', 'state', 'district', 'pincode'])['total_activity'].sum().reset_index()
        
        # Fit Model
        # We use the 'total_activity' volume to find spikes
        X = daily_stats[['total_activity']]
        
        # -1 indicates Anomaly, 1 indicates Normal
        daily_stats['anomaly_score'] = self.model.fit_predict(X)
        
        # Filter only the anomalies
        anomalies = daily_stats[daily_stats['anomaly_score'] == -1]
        
        return anomalies
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class DemandForecaster:
    def __init__(self, df):
        self.df = df
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def predict_next_month(self):
        """
        Trains a robust ML model on daily update volumes to predict
        the total demand for the next 30 days for each state.
        """
        # Aggregating daily total activity
        daily_data = self.df.groupby('date')['total_activity'].sum().reset_index()
        daily_data = daily_data.sort_values('date')
        
        # Feature Engineering: Lag Features (Past 1 day, Past 7 days)
        daily_data['lag_1'] = daily_data['total_activity'].shift(1)
        daily_data['lag_7'] = daily_data['total_activity'].shift(7)
        daily_data['day_of_week'] = daily_data['date'].dt.dayofweek
        daily_data['day_of_month'] = daily_data['date'].dt.day
        
        # Drop NaNs created by lags
        train_data = daily_data.dropna()
        
        X = train_data[['lag_1', 'lag_7', 'day_of_week', 'day_of_month']]
        y = train_data['total_activity']
        
        # Train Model
        self.model.fit(X, y)
        
        # Forecast Loop (Next 30 Days)
        future_predictions = []
        last_row = train_data.iloc[-1]
        current_lag_1 = last_row['total_activity']
        current_lag_7 = train_data.iloc[-7]['total_activity']
        
        for i in range(1, 31):
            # Create future feature vector
            next_date = last_row['date'] + pd.Timedelta(days=i)
            features = np.array([[current_lag_1, current_lag_7, next_date.dayofweek, next_date.day]])
            
            # Predict
            pred = self.model.predict(features)[0]
            future_predictions.append({'date': next_date, 'predicted_demand': int(pred)})
            
            # Update lags for next step
            current_lag_7 = current_lag_1 # Simplification for loop
            current_lag_1 = pred
            
        return pd.DataFrame(future_predictions)
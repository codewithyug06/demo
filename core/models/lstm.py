import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DeepTemporalNet(nn.Module):
    """ EXISTING CLASS PRESERVED """
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

class ForecastEngine:
    """ UPDATED ENGINE """
    def __init__(self, df):
        self.df = df
        self.model = DeepTemporalNet()
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def generate_forecast(self, days=30):
        """ EXISTING FUNCTION PRESERVED """
        if 'date' not in self.df.columns: return pd.DataFrame()
        daily = self.df.groupby('date')['total_activity'].sum().reset_index().sort_values('date')
        if len(daily) < 15: return pd.DataFrame()
        
        data = daily['total_activity'].values.reshape(-1, 1)
        normalized = self.scaler.fit_transform(data)
        X_train = torch.FloatTensor(normalized[:-1]).view(-1, 1, 1)
        
        self.model.eval()
        predictions = []
        last_tensor = X_train[-1].view(1, 1, 1)
        
        for _ in range(days):
            with torch.no_grad():
                pred = self.model(last_tensor)
                predictions.append(pred.item())
                last_tensor = pred.view(1, 1, 1)
        
        true_preds = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        future_dates = [daily['date'].iloc[-1] + pd.Timedelta(days=i) for i in range(1, days+1)]
        return pd.DataFrame({'Date': future_dates, 'Predicted_Load': true_preds.flatten()})

    def generate_forecast_with_confidence(self, days=30, volatility_factor=0.1):
        """ EXISTING FUNCTION PRESERVED """
        base_forecast = self.generate_forecast(days)
        if base_forecast.empty: return pd.DataFrame()
        
        time_scale = np.linspace(1, 2, days)
        uncertainty = base_forecast['Predicted_Load'] * volatility_factor * time_scale
        base_forecast['Upper_Bound'] = base_forecast['Predicted_Load'] + uncertainty
        base_forecast['Lower_Bound'] = base_forecast['Predicted_Load'] - uncertainty
        return base_forecast

    def calculate_resource_demand(self, days=30):
        """
        NEW FUNCTION: AI RESOURCE DEMAND FORECASTING
        Translates raw transaction volume into infrastructure requirements.
        Assumption: 1 Server Unit handles 500 transactions/day. 1 Operator handles 50.
        """
        forecast = self.generate_forecast_with_confidence(days)
        if forecast.empty: return pd.DataFrame()
        
        # Resource Logic
        forecast['Required_Server_Units'] = np.ceil(forecast['Upper_Bound'] / 500)
        forecast['Required_Manpower'] = np.ceil(forecast['Upper_Bound'] / 50)
        
        return forecast
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DeepTemporalNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

class ForecastEngine:
    def __init__(self, df):
        self.df = df
        self.model = DeepTemporalNet()
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def generate_forecast(self, days=30):
        if 'date' not in self.df.columns: return pd.DataFrame()
        
        daily = self.df.groupby('date')['total_activity'].sum().reset_index().sort_values('date')
        if len(daily) < 15: return pd.DataFrame()
        
        data = daily['total_activity'].values.reshape(-1, 1)
        normalized = self.scaler.fit_transform(data)
        
        # Tensor Prep
        X_train = torch.FloatTensor(normalized[:-1]).view(-1, 1, 1)
        
        # Inference Loop (Simulated for real-time UI response)
        self.model.eval()
        predictions = []
        last_tensor = X_train[-1].view(1, 1, 1)
        
        for _ in range(days):
            with torch.no_grad():
                pred = self.model(last_tensor)
                predictions.append(pred.item())
                # In a real autoregressive loop, pred becomes next input
                last_tensor = pred.view(1, 1, 1)
        
        # Post-process
        true_preds = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        future_dates = [daily['date'].iloc[-1] + pd.Timedelta(days=i) for i in range(1, days+1)]
        
        return pd.DataFrame({'Date': future_dates, 'Predicted_Load': true_preds.flatten()})
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class AadhaarLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

class PyTorchForecaster:
    def __init__(self, df):
        self.df = df
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.model = AadhaarLSTM()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train_and_forecast(self, days_to_predict=30):
        # 1. Prepare Data
        daily = self.df.groupby('date')['total_activity'].sum().reset_index().sort_values('date')
        data = daily['total_activity'].values.astype(float)
        normalized_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        # Convert to Tensors
        train_data_normalized = torch.FloatTensor(normalized_data).view(-1)
        
        # 2. Create Sequences
        train_window = 7
        def create_inout_sequences(input_data, tw):
            inout_seq = []
            L = len(input_data)
            for i in range(L-tw):
                train_seq = input_data[i:i+tw]
                train_label = input_data[i+tw:i+tw+1]
                inout_seq.append((train_seq ,train_label))
            return inout_seq

        train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

        # 3. Train Loop
        epochs = 5 # Keep low for hackathon speed (Use 50 for production)
        for i in range(epochs):
            for seq, labels in train_inout_seq:
                self.optimizer.zero_grad()
                self.model.hidden_cell = (torch.zeros(1, 1, self.model.hidden_layer_size),
                                          torch.zeros(1, 1, self.model.hidden_layer_size))
                y_pred = self.model(seq)
                single_loss = self.criterion(y_pred, labels)
                single_loss.backward()
                self.optimizer.step()

        # 4. Forecast
        fut_pred = 30
        test_inputs = train_data_normalized[-train_window:].tolist()

        self.model.eval()
        for i in range(fut_pred):
            seq = torch.FloatTensor(test_inputs[-train_window:])
            with torch.no_grad():
                self.model.hidden_cell = (torch.zeros(1, 1, self.model.hidden_layer_size),
                                          torch.zeros(1, 1, self.model.hidden_layer_size))
                test_inputs.append(self.model(seq).item())

        # Inverse Transform
        actual_predictions = self.scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))
        
        # Dates
        last_date = daily['date'].iloc[-1]
        future_dates = [last_date + pd.Timedelta(days=x) for x in range(1, fut_pred+1)]
        
        return pd.DataFrame({'date': future_dates, 'predicted_demand': actual_predictions.flatten()})
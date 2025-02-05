import torch
import torch.nn as nn
from .base_forecaster import BaseForecaster

class TransformerForecaster(BaseForecaster):
    def __init__(self, input_dim, model_dim, num_layers, nhead, forecast_horizon):
        super(TransformerForecaster, self).__init__()
        self.encoder = nn.Linear(input_dim, model_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim, nhead=nhead, dim_feedforward=model_dim*4, dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.positional_encoding = nn.Parameter(self._generate_positional_encoding(forecast_horizon, model_dim), requires_grad=False)
        self.fc_out = nn.Linear(model_dim, input_dim)
        self.forecast_horizon = forecast_horizon

    def _generate_positional_encoding(self, forecast_horizon, model_dim):
        pe = torch.zeros(forecast_horizon, model_dim)
        position = torch.arange(0, forecast_horizon, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        batch_size, T_in, input_dim = x.size()
        memory = self.encoder(x).transpose(0, 1)
        tgt = self.positional_encoding.unsqueeze(1).repeat(1, batch_size, 1)
        out = self.decoder(tgt, memory)
        preds = self.fc_out(out).transpose(0, 1)
        return preds

    def train_model(self, train_loader, num_epochs, learning_rate, device):
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device).squeeze(1)
                y_batch = y_batch.to(device).transpose(1,2)
                optimizer.zero_grad()
                preds = self(x_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * x_batch.size(0)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader.dataset):.4f}")
        return

    def predict(self, new_x, forecast_horizon, device):
        self.eval()
        new_x = new_x.to(device).squeeze(1)
        # If new_x is 2D, add the feature dimension.
        if new_x.dim() == 2:
            new_x = new_x.unsqueeze(-1)  # Now shape is (batch, T_in, 1)
        with torch.no_grad():
            preds = self(new_x)
        return preds, None


    def evaluate(self, test_loader, device):
        self.eval()
        mse_total, mae_total, count = 0.0, 0.0, 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device).squeeze(1)
                y_batch = y_batch.to(device).squeeze(1) 
                preds = self(x_batch)
                mse = ((preds.squeeze(-1) - y_batch) ** 2).mean().item()
                mae = (torch.abs(preds.squeeze(-1) - y_batch)).mean().item()
                mse_total += mse * x_batch.size(0)
                mae_total += mae * x_batch.size(0)
                count += x_batch.size(0)
        mse_avg = mse_total / count
        mae_avg = mae_total / count
        print(f"Evaluation -> MSE: {mse_avg:.4f}, MAE: {mae_avg:.4f}")
        return mse_avg, mae_avg

import torch
import torch.nn as nn


class TimeAwareTransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, forecast_horizon, num_layers=1, nhead=4):
        super(TimeAwareTransformerDecoder, self).__init__()
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim
        self.positional_encoding = self._generate_positional_encoding(
            forecast_horizon, hidden_dim
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4, dropout=0.1
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.fc_logvar = nn.Linear(hidden_dim, output_dim)

    def _generate_positional_encoding(self, forecast_horizon, hidden_dim):
        pe = torch.zeros(forecast_horizon, hidden_dim)
        position = torch.arange(0, forecast_horizon, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, hidden):
        batch_size = hidden.size(1)
        tgt = self.positional_encoding.unsqueeze(1).repeat(1, batch_size, 1)
        out = self.transformer_decoder(tgt, hidden)
        preds = self.fc_out(out)
        pred_logvars = self.fc_logvar(out)
        return preds.transpose(0, 1), pred_logvars.transpose(0, 1)

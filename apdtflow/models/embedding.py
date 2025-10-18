import torch
import torch.nn as nn

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.1, context_dim=None):
        super(GatedResidualNetwork, self).__init__()
        self.output_dim = output_dim or input_dim
        self.context_fc = nn.Linear(context_dim, hidden_dim) if context_dim is not None else None
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(self.output_dim, self.output_dim)
        if self.output_dim != input_dim:
            self.skip = nn.Linear(input_dim, self.output_dim)
        else:
            self.skip = None

    def forward(self, x, context=None):
        residual = self.skip(x) if self.skip is not None else x
        x_proj = self.fc1(x)
        if self.context_fc is not None and context is not None:
            x_proj = x_proj + self.context_fc(context)
        x_proj = self.elu(x_proj)
        x_proj = self.dropout(x_proj)
        x_proj = self.fc2(x_proj)
        gating = torch.sigmoid(self.gate(x_proj))
        return gating * x_proj + (1 - gating) * residual

class TimeSeriesEmbedding(nn.Module):
    def __init__(self, embed_dim, calendar_dim=None, dropout=0.1):
        """
        A learnable time series embedding module that processes:
          - a raw time index (normalized)
          - a periodic component
          - optionally, calendar features.
        """
        super(TimeSeriesEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.raw_grn = GatedResidualNetwork(1, embed_dim, embed_dim, dropout=dropout)
        self.periodic_grn = GatedResidualNetwork(1, embed_dim, embed_dim, dropout=dropout)
        if calendar_dim is not None:
            self.calendar_grn = GatedResidualNetwork(calendar_dim, embed_dim, embed_dim, dropout=dropout)
            self.fc = nn.Linear(embed_dim * 3, embed_dim)
        else:
            self.calendar_grn = None
            self.fc = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, time_input, periodic_input, calendar_features=None):
        raw_emb = self.raw_grn(time_input)
        periodic_emb = self.periodic_grn(periodic_input)
        if self.calendar_grn is not None and calendar_features is not None:
            cal_emb = self.calendar_grn(calendar_features)
            combined = torch.cat([raw_emb, periodic_emb, cal_emb], dim=-1)
        else:
            combined = torch.cat([raw_emb, periodic_emb], dim=-1)
        embedding = self.fc(combined)
        return embedding
import torch
from apdtflow.models.embedding import TimeSeriesEmbedding

def test_time_series_embedding_shape():
    batch_size = 4
    T = 20
    embed_dim = 16
    time_input = torch.linspace(0, 1, steps=T).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1)
    periodic_input = time_input.clone()
    embedding_module = TimeSeriesEmbedding(embed_dim=embed_dim, calendar_dim=None, dropout=0.1)
    output = embedding_module(time_input, periodic_input)
    assert output.shape == (batch_size, T, embed_dim)

def test_time_series_embedding_with_calendar():
    batch_size = 4
    T = 20
    embed_dim = 16
    calendar_dim = 4
    time_input = torch.linspace(0, 1, steps=T).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1)
    periodic_input = time_input.clone()
    calendar_features = torch.randn(batch_size, T, calendar_dim)
    embedding_module = TimeSeriesEmbedding(embed_dim=embed_dim, calendar_dim=calendar_dim, dropout=0.1)
    output = embedding_module(time_input, periodic_input, calendar_features=calendar_features)
    assert output.shape == (batch_size, T, embed_dim)

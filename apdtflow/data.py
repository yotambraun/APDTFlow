import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class TimeSeriesWindowDataset(Dataset):
    """
    A dataset for a univariate time series stored in a CSV file.
    Creates samples using a sliding window.
    """
    def __init__(self, csv_file, date_col, value_col, T_in, T_out, transform=None):
        self.df = pd.read_csv(csv_file)
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df.sort_values(date_col, inplace=True)
        self.series = self.df[value_col].values.astype(np.float32)
        self.T_in = T_in
        self.T_out = T_out
        self.transform = transform
        self.samples = []
        total_length = T_in + T_out
        for i in range(len(self.series) - total_length + 1):
            self.samples.append(self.series[i:i+total_length])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = torch.tensor(sample[:self.T_in], dtype=torch.float32).unsqueeze(0) 
        y = torch.tensor(sample[self.T_in:], dtype=torch.float32).unsqueeze(0)
        if self.transform:
            x = self.transform(x)
        return x, y

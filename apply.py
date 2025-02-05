import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
from torchdiffeq import odeint_adjoint

############################################
# 1. DATA PROCESSING AND AUGMENTATION FUNCTIONS
############################################

def jitter(x, sigma=0.03):
    """Add random Gaussian noise."""
    return x + sigma * torch.randn_like(x)

def scaling(x, sigma=0.1):
    """Multiply by a random scaling factor."""
    factor = torch.randn(x.size(0), 1, 1, device=x.device) * sigma + 1.0
    return x * factor

def time_warp(x, max_warp=0.2):
    """Simple time warping by re–interpolating the series."""
    batch_size, channels, length = x.size()
    warp = torch.linspace(0, 1, steps=length, device=x.device) + (torch.rand(length, device=x.device) - 0.5) * max_warp
    warp, _ = torch.sort(warp)
    orig_idx = torch.linspace(0, 1, steps=length, device=x.device)
    x_warped = torch.zeros_like(x)
    for b in range(batch_size):
        for c in range(channels):
            x_np = x[b, c, :].cpu().numpy()
            warp_np = warp.cpu().numpy()
            orig_np = orig_idx.cpu().numpy()
            interp_np = np.interp(orig_np, warp_np, x_np)
            x_warped[b, c, :] = torch.tensor(interp_np, device=x.device)
    return x_warped

############################################
# 2. CUSTOM DATASET FOR A SINGLE TIME SERIES (SLIDING WINDOW)
############################################

class TimeSeriesWindowDataset(Dataset):
    """
    A dataset for a single univariate time series stored in a CSV file.
    It creates samples by sliding a window of length (T_in + T_out) along the series.
    
    Args:
        csv_file (str): Path to the CSV file.
        date_col (str): Name of the date column.
        value_col (str): Name of the value column.
        T_in (int): Number of time steps used as input.
        T_out (int): Number of time steps to forecast.
        transform (callable, optional): Optional transform (e.g., augmentation) for the input.
    """
    def __init__(self, csv_file, date_col, value_col, T_in, T_out, transform=None):
        self.df = pd.read_csv(csv_file)
        # Parse dates and sort (if not already sorted)
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df.sort_values(date_col, inplace=True)
        self.series = self.df[value_col].values.astype(np.float32)
        self.T_in = T_in
        self.T_out = T_out
        self.transform = transform
        
        self.samples = []
        total_length = T_in + T_out
        # Create sliding windows
        for i in range(len(self.series) - total_length + 1):
            self.samples.append(self.series[i:i+total_length])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = sample[:self.T_in]
        y = sample[self.T_in:]
        # Convert to torch tensors and add a channel dimension (shape: [1, T])
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        if self.transform:
            x = self.transform(x)
        return x, y

############################################
# 3. MODEL COMPONENTS (APDTFlow and submodules)
############################################

# 3.1 Dynamic Dilation Convolution (for multi-scale encoder)
class DynamicDilationConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DynamicDilationConv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.log_dilation = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        dilation = int(torch.clamp(torch.exp(self.log_dilation), min=1.0).round().item())
        padding = (self.kernel_size - 1) * dilation // 2
        return F.conv1d(x, self.conv.weight, self.conv.bias, padding=padding, dilation=dilation)

# 3.2 Multi-scale Residual Decomposer
class ResidualMultiScaleDecomposer(nn.Module):
    def __init__(self, num_scales, input_channels, filter_size):
        super(ResidualMultiScaleDecomposer, self).__init__()
        self.num_scales = num_scales
        self.paths = nn.ModuleList()
        for _ in range(num_scales):
            self.paths.append(nn.Sequential(
                DynamicDilationConv(input_channels, input_channels, filter_size),
                nn.BatchNorm1d(input_channels),
                nn.ReLU(),
                nn.Conv1d(input_channels, 1, filter_size, padding=filter_size//2),
                nn.BatchNorm1d(1),
                nn.ReLU()
            ))
    
    def forward(self, x):
        return [path(x) for path in self.paths]

# 3.3 Advanced Probabilistic Neural Dynamics Module (Variational update)
class ProbabilisticNeuralDynamics(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(ProbabilisticNeuralDynamics, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + input_dim + 1, 64),  # hidden_dim + input_dim + time_feature
            nn.ReLU(),
            nn.Linear(64, hidden_dim * 2)
        )
    
    def forward(self, t, state):
        h, x_scale = state  # h: (batch_size, hidden_dim), x_scale: (batch_size, input_dim)
        t_scalar = t if t.dim() == 0 else t[0]  # Ensure t is scalar
        time_feature = t_scalar.unsqueeze(0).expand(h.size(0), 1)
        inp = torch.cat([h, x_scale, time_feature], dim=-1)
        update = self.net(inp)
        delta_mu, delta_logvar = torch.chunk(update, 2, dim=-1)
        return delta_mu, delta_logvar

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# 3.4 Adaptive ODE Integration with Variational Updates
def adaptive_prob_ode_solver(dynamics, h0, logvar0, t_span, x_scale_sequence):
    class ODEFunc(nn.Module):
        def __init__(self, dynamics, x_sequence, t_span):
            super(ODEFunc, self).__init__()
            self.dynamics = dynamics
            self.x_sequence = x_sequence  # (batch_size, T_in, input_dim)
            self.t_span = t_span
            
        def forward(self, t, state):
            h, logvar = state
            idx = (torch.abs(self.t_span - t)).argmin()
            x_t = self.x_sequence[:, idx, :]  # (batch_size, input_dim)
            delta_mu, delta_logvar = self.dynamics(t, (h, x_t))
            return (delta_mu, delta_logvar)
    
    ode_func = ODEFunc(dynamics, x_scale_sequence, t_span)
    state0 = (h0, logvar0)
    sol = odeint_adjoint(ode_func, state0, t_span, rtol=1e-3, atol=1e-3)
    h_sol = sol[0].transpose(0, 1)       # (batch_size, T_in, hidden_dim)
    logvar_sol = sol[1].transpose(0, 1)    # (batch_size, T_in, hidden_dim)
    return h_sol, logvar_sol

# 3.5 Fusion Module (Probabilistic Attention)
class ProbScaleFusion(nn.Module):
    def __init__(self, hidden_dim, num_scales):
        super(ProbScaleFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.num_scales = num_scales

    def forward(self, latent_means, latent_logvars):
        batch_size, hidden_dim = latent_means[0].size()
        means_stack = torch.stack(latent_means, dim=1)
        scores = self.attention(means_stack.view(-1, hidden_dim))
        scores = scores.view(batch_size, self.num_scales, 1)
        uncert_stack = torch.stack(latent_logvars, dim=1).exp().mean(dim=-1, keepdim=True)
        epsilon = 1e-6
        weights = torch.softmax(scores / (uncert_stack + epsilon), dim=1)
        fused = torch.sum(weights * means_stack, dim=1)
        return fused

# 3.6 Sequence Decoder for Multi-step Forecasting
class SequenceDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, forecast_horizon, num_layers=1):
        super(SequenceDecoder, self).__init__()
        self.forecast_horizon = forecast_horizon
        self.gru = nn.GRU(output_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.logvar_fc = nn.Linear(hidden_dim, output_dim)
        self.init_token = nn.Parameter(torch.zeros(1, output_dim))
    
    def forward(self, hidden):
        batch_size = hidden.size(1)
        input_token = self.init_token.expand(1, batch_size, -1)
        outputs = []
        logvars = []
        for _ in range(self.forecast_horizon):
            out, hidden = self.gru(input_token, hidden)
            pred = self.fc(out.squeeze(0))
            pred_logvar = self.logvar_fc(out.squeeze(0))
            outputs.append(pred)
            logvars.append(pred_logvar)
            input_token = pred.unsqueeze(0)
        outputs = torch.stack(outputs, dim=1)
        logvars = torch.stack(logvars, dim=1)
        return outputs, logvars

# 3.7 End-to-End APDTFlow Model
class APDTFlow(nn.Module):
    def __init__(self, num_scales, input_channels, filter_size, hidden_dim, output_dim, forecast_horizon):
        super(APDTFlow, self).__init__()
        self.decomposer = ResidualMultiScaleDecomposer(num_scales, input_channels, filter_size)
        self.num_scales = num_scales
        self.dynamics_modules = nn.ModuleList([
            ProbabilisticNeuralDynamics(hidden_dim=hidden_dim, input_dim=1)
            for _ in range(num_scales)
        ])
        self.fusion = ProbScaleFusion(hidden_dim, num_scales)
        self.decoder = SequenceDecoder(hidden_dim, output_dim, forecast_horizon)
    
    def forward(self, x, t_span):
        scale_components = self.decomposer(x)  # List of tensors: each (batch_size, 1, T_in)
        latent_means = []
        latent_logvars = []
        batch_size, _, T_in = scale_components[0].size()
        for i in range(self.num_scales):
            x_scale_seq = scale_components[i].squeeze(1).unsqueeze(-1)  # (batch_size, T_in, 1)
            h0 = torch.zeros(batch_size, self.dynamics_modules[i].hidden_dim, device=x.device)
            logvar0 = torch.zeros(batch_size, self.dynamics_modules[i].hidden_dim, device=x.device)
            h_sol, logvar_sol = adaptive_prob_ode_solver(self.dynamics_modules[i], h0, logvar0, t_span, x_scale_seq)
            latent_means.append(h_sol[:, -1, :])
            latent_logvars.append(logvar_sol[:, -1, :])
        fused_state = self.fusion(latent_means, latent_logvars)
        hidden = fused_state.unsqueeze(0)
        outputs, out_logvars = self.decoder(hidden)
        return outputs, out_logvars

############################################
# 4. TRAINING AND INFERENCE FUNCTIONS
############################################

def train_apdtflow(model, train_loader, num_epochs, learning_rate, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # Check if x_batch has an extra dimension and squeeze if necessary.
            if x_batch.dim() == 4 and x_batch.size(1) == 1:
                x_batch = x_batch.squeeze(1)
            # Expected shape: (batch_size, 1, T_in)
            batch_size, _, T_in = x_batch.size()
            t_span = torch.linspace(0, 1, steps=T_in, device=device)
            optimizer.zero_grad()
            preds, pred_logvars = model(x_batch, t_span)
            # Adjust y_batch from (batch_size, 1, T_out) to (batch_size, T_out, 1)
            mse = (preds - y_batch.transpose(1, 2)) ** 2
            loss = torch.mean(0.5 * (mse / (pred_logvars.exp() + 1e-6)) + 0.5 * pred_logvars)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_size
        avg_loss = epoch_loss / len(train_loader.dataset)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    return loss_history

def infer_multi_step(model, new_x, forecast_horizon, device):
    model.eval()
    new_x = new_x.to(device)
    # Ensure new_x has shape (batch_size, 1, T_in)
    if new_x.dim() == 4 and new_x.size(1) == 1:
        new_x = new_x.squeeze(1)
    batch_size, _, T_in = new_x.size()
    t_span = torch.linspace(0, 1, steps=T_in, device=device)
    with torch.no_grad():
        preds, pred_logvars = model(new_x, t_span)
    return preds.cpu(), pred_logvars.cpu()

############################################
# 5. MAIN EXECUTION: TRAINING AND TESTING ON CUSTOM CSV DATASETS
############################################

if __name__ == "__main__":
    # For reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Hyperparameters (adjust these for your datasets)
    T_in = 24       # e.g., 24 months = 2 years input
    T_out = 6       # e.g., forecast 6 months ahead
    num_scales = 3
    input_channels = 1
    filter_size = 5
    hidden_dim = 16
    output_dim = 1
    learning_rate = 0.001
    num_epochs = 15
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Choose one of your datasets:
    # Option 1: Electric Production dataset
    electric_csv = r"C:\Users\yotam\code_projects\APDTFlow\Electric_Production.csv"
    date_col_elec = "DATE"
    value_col_elec = "IPG2211A2N"
    
    # Option 2: Monthly Beer Production dataset
    beer_csv = r"C:\Users\yotam\code_projects\APDTFlow\monthly-beer-production-in-austr.csv"
    date_col_beer = "Month"
    value_col_beer = "Monthly beer production"
    
    # For example, we use the Electric Production dataset.
    dataset = TimeSeriesWindowDataset(
        csv_file=electric_csv,
        date_col=date_col_elec,
        value_col=value_col_elec,
        T_in=T_in,
        T_out=T_out,
        transform=lambda x: time_warp(scaling(jitter(x)))
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Instantiate the model
    model = APDTFlow(num_scales, input_channels, filter_size, hidden_dim, output_dim, T_out)
    
    # Train the model
    loss_history = train_apdtflow(model, train_loader, num_epochs, learning_rate, device)
    
    # Plot training loss over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Inference on one batch from the dataset
    model.eval()
    with torch.no_grad():
        x_sample, y_sample = next(iter(train_loader))
        preds, pred_logvars = infer_multi_step(model, x_sample, T_out, device)
    
    # Visualize one sample's forecast with uncertainty bands
    sample_idx = 0
    x_input = x_sample[sample_idx].squeeze().cpu().numpy()
    y_true = y_sample[sample_idx].squeeze().cpu().numpy()
    y_pred = preds[sample_idx].squeeze().cpu().numpy()
    # Compute standard deviation from predicted log-variance:
    y_pred_logvars = pred_logvars[sample_idx].squeeze().cpu().numpy()
    sigma = np.sqrt(np.exp(y_pred_logvars))
    upper = y_pred + 2 * sigma
    lower = y_pred - 2 * sigma
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(T_in), x_input, 'b-', label='Input History')
    plt.plot(range(T_in, T_in+T_out), y_true, 'go-', label='True Future')
    plt.plot(range(T_in, T_in+T_out), y_pred, 'rx--', label='Forecast')
    plt.fill_between(range(T_in, T_in+T_out), lower, upper, color='gray', alpha=0.3, label='Uncertainty (±2σ)')
    plt.title('Multi-step Forecast vs Test Data')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

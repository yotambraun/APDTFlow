import torch
import torch.nn as nn
from .embedding import TimeSeriesEmbedding
from .multi_scale_decomposer import ResidualMultiScaleDecomposer
from .dynamics import HierarchicalNeuralDynamics, adaptive_hierarchical_ode_solver
from apdtflow.evaluation.regression_evaluator import RegressionEvaluator
from apdtflow.logger_util import get_logger
from .fusion import ProbScaleFusion
from .decoder import TimeAwareTransformerDecoder

logger = get_logger("evaluation.log")

class APDTFlow(nn.Module):
    def __init__(
        self,
        num_scales,
        input_channels,
        filter_size,
        hidden_dim,
        output_dim,
        forecast_horizon,
        use_embedding=True,
        num_exog_features=0,
        exog_fusion_type='gated'
    ):
        """
        Initializes the APDTFlow model with optional exogenous variable support.

        Args:
          num_scales: Number of scales for multi-scale decomposition.
          input_channels: Number of input channels (expected to be 1 after embedding/projection).
          filter_size: Convolution filter size for the decomposer.
          hidden_dim: Dimensionality for the neural ODE dynamics and embedding.
          output_dim: Output dimension of the forecast (e.g., 1).
          forecast_horizon: Number of future time steps to forecast.
          use_embedding: If True, apply the learnable TimeSeriesEmbedding to the input.
          num_exog_features: Number of exogenous variables (NEW in v0.2.0)
          exog_fusion_type: How to fuse exog vars: 'concat', 'gated', or 'attention' (NEW in v0.2.0)
        """
        super(APDTFlow, self).__init__()
        self.use_embedding = use_embedding
        self.num_exog_features = num_exog_features
        self.has_exog = num_exog_features > 0

        # Exogenous fusion (NEW in v0.2.0)
        if self.has_exog:
            from apdtflow.exogenous import ExogenousFeatureFusion
            self.exog_fusion = ExogenousFeatureFusion(
                hidden_dim=hidden_dim,
                num_exog_features=num_exog_features,
                fusion_type=exog_fusion_type
            )

        if self.use_embedding:
            self.embedding = TimeSeriesEmbedding(embed_dim=hidden_dim, calendar_dim=None, dropout=0.1)
            self.input_proj = nn.Linear(hidden_dim, 1)
        self.decomposer = ResidualMultiScaleDecomposer(num_scales, input_channels=1, filter_size=filter_size)
        self.dynamics_module = HierarchicalNeuralDynamics(hidden_dim=hidden_dim, input_dim=1)
        self.fusion = ProbScaleFusion(hidden_dim, num_scales)
        self.decoder = TimeAwareTransformerDecoder(hidden_dim, output_dim, forecast_horizon)

    def forward(self, x, t_span, exog=None):
        """
        Args:
          x: Input tensor of shape (batch, 1, T_in).
          t_span: 1D tensor of time indices of length T_in (normalized to [0,1]).
          exog: Optional exogenous variables of shape (batch, num_exog, T_in).
        Returns:
          Tuple (outputs, out_logvars) each of shape (batch, forecast_horizon, output_dim).
        """
        batch_size, _, T_in = x.size()
        if self.use_embedding:
            time_indices = t_span.unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
            embedded = self.embedding(time_indices, time_indices)
            projected = self.input_proj(embedded)
            x = projected.transpose(1, 2)

        # Apply exogenous fusion if available
        if self.has_exog and exog is not None:
            x = self.exog_fusion(x, exog)

        scale_components = self.decomposer(x)
        
        latent_means = []
        latent_logvars = []
        for comp in scale_components:
            comp_t = comp.transpose(1, 2)
            h0 = torch.zeros(batch_size, self.dynamics_module.hidden_dim, device=x.device)
            logvar0 = torch.zeros(batch_size, self.dynamics_module.hidden_dim, device=x.device)
            h_sol, logvar_sol = adaptive_hierarchical_ode_solver(self.dynamics_module, h0, logvar0, t_span, comp_t)
            latent_means.append(h_sol[:, -1, :])
            latent_logvars.append(logvar_sol[:, -1, :])
        
        fused_state = self.fusion(latent_means, latent_logvars)
        hidden = fused_state.unsqueeze(0)
        outputs, out_logvars = self.decoder(hidden)
        return outputs, out_logvars

    def train_model(self, train_loader, num_epochs, learning_rate, device):
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0.0
            for batch_data in train_loader:
                # Handle both with and without exogenous variables
                if len(batch_data) == 4:
                    x_batch, y_batch, exog_x, exog_y = batch_data
                    exog_x = exog_x.to(device)
                    exog_y = exog_y.to(device)
                else:
                    x_batch, y_batch = batch_data
                    exog_x = None

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                if x_batch.dim() == 4 and x_batch.size(1) == 1:
                    x_batch = x_batch.squeeze(1)
                batch_size, _, T_in = x_batch.size()
                t_span = torch.linspace(0, 1, steps=T_in, device=device)
                optimizer.zero_grad()
                preds, pred_logvars = self(x_batch, t_span, exog=exog_x)
                mse = (preds - y_batch.transpose(1, 2)) ** 2
                loss = torch.mean(0.5 * (mse / (pred_logvars.exp() + 1e-6)) + 0.5 * pred_logvars)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_size
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader.dataset):.4f}")
        return

    def predict(self, new_x, forecast_horizon, device, exog=None):
        self.eval()
        new_x = new_x.to(device)
        if new_x.dim() == 4 and new_x.size(1) == 1:
            new_x = new_x.squeeze(1)
        batch_size, _, T_in = new_x.size()
        t_span = torch.linspace(0, 1, steps=T_in, device=device)

        if exog is not None:
            exog = exog.to(device)

        with torch.no_grad():
            preds, pred_logvars = self(new_x, t_span, exog=exog)
        return preds, pred_logvars

    def evaluate(self, test_loader, device, metrics=["MSE", "MAE"]):
        self.eval()
        evaluator = RegressionEvaluator(metrics)
        total_metrics = {m: 0.0 for m in metrics}
        total_samples = 0
        with torch.no_grad():
            for batch_data in test_loader:
                # Handle both with and without exogenous variables
                if len(batch_data) == 4:
                    x_batch, y_batch, exog_x, exog_y = batch_data
                    exog_x = exog_x.to(device)
                else:
                    x_batch, y_batch = batch_data
                    exog_x = None

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                if x_batch.dim() == 4 and x_batch.size(1) == 1:
                    x_batch = x_batch.squeeze(1)
                batch_size, _, T_in = x_batch.size()
                t_span = torch.linspace(0, 1, steps=T_in, device=device)
                preds, _ = self(x_batch, t_span, exog=exog_x)
                batch_results = evaluator.evaluate(preds, y_batch.transpose(1, 2))
                for m in metrics:
                    total_metrics[m] += batch_results[m] * batch_size
                total_samples += batch_size
        avg_metrics = {m: total_metrics[m] / total_samples for m in metrics}
        logger.info("Evaluation -> " + ", ".join([f"{m}: {avg_metrics[m]:.4f}" for m in metrics]))
        return avg_metrics
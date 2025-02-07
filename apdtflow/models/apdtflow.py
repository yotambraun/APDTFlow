import torch
from .multi_scale_decomposer import ResidualMultiScaleDecomposer
from .dynamics import HierarchicalNeuralDynamics, adaptive_hierarchical_ode_solver
from apdtflow.evaluation.regression_evaluator import RegressionEvaluator
from apdtflow.logger_util import get_logger
from .fusion import ProbScaleFusion
from .decoder import TimeAwareTransformerDecoder
from .base_forecaster import BaseForecaster

logger = get_logger("evaluation.log")


class APDTFlow(BaseForecaster):
    def __init__(
        self,
        num_scales,
        input_channels,
        filter_size,
        hidden_dim,
        output_dim,
        forecast_horizon,
    ):
        super(APDTFlow, self).__init__()
        self.hidden_dim = hidden_dim
        self.decomposer = ResidualMultiScaleDecomposer(
            num_scales, input_channels, filter_size
        )
        self.num_scales = num_scales
        self.dynamics_module = HierarchicalNeuralDynamics(
            hidden_dim=hidden_dim, input_dim=1
        )
        self.fusion = ProbScaleFusion(hidden_dim, num_scales)
        self.decoder = TimeAwareTransformerDecoder(
            hidden_dim, output_dim, forecast_horizon
        )

    def forward(self, x, t_span):
        scale_components = self.decomposer(x)
        latent_means = []
        latent_logvars = []
        batch_size, _, T_in = scale_components[0].size()
        for i in range(self.num_scales):
            x_scale_seq = scale_components[i].squeeze(1).unsqueeze(-1)
            h0 = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            logvar0 = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            h_sol, logvar_sol = adaptive_hierarchical_ode_solver(
                self.dynamics_module, h0, logvar0, t_span, x_scale_seq
            )
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
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                if x_batch.dim() == 4 and x_batch.size(1) == 1:
                    x_batch = x_batch.squeeze(1)
                batch_size, _, T_in = x_batch.size()
                t_span = torch.linspace(0, 1, steps=T_in, device=device)
                optimizer.zero_grad()
                preds, pred_logvars = self(x_batch, t_span)
                mse = (preds - y_batch.transpose(1, 2)) ** 2
                loss = torch.mean(
                    0.5 * (mse / (pred_logvars.exp() + 1e-6)) + 0.5 * pred_logvars
                )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_size
            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader.dataset):.4f}"
            )
        return

    def predict(self, new_x, forecast_horizon, device):
        self.eval()
        new_x = new_x.to(device)
        if new_x.dim() == 4 and new_x.size(1) == 1:
            new_x = new_x.squeeze(1)
        batch_size, _, T_in = new_x.size()
        t_span = torch.linspace(0, 1, steps=T_in, device=device)
        with torch.no_grad():
            preds, pred_logvars = self(new_x, t_span)
        return preds, pred_logvars

    def evaluate(self, test_loader, device, metrics=["MSE", "MAE"]):
        self.eval()
        evaluator = RegressionEvaluator(metrics)
        total_metrics = {m: 0.0 for m in metrics}
        total_samples = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                if x_batch.dim() == 4 and x_batch.size(1) == 1:
                    x_batch = x_batch.squeeze(1)
                batch_size, _, T_in = x_batch.size()
                t_span = torch.linspace(0, 1, steps=T_in, device=device)
                preds, _ = self(x_batch, t_span)
                batch_results = evaluator.evaluate(preds, y_batch.transpose(1, 2))
                for m in metrics:
                    total_metrics[m] += batch_results[m] * batch_size
                total_samples += batch_size
        avg_metrics = {m: total_metrics[m] / total_samples for m in metrics}
        logger.info(
            "Evaluation -> "
            + ", ".join([f"{m}: {avg_metrics[m]:.4f}" for m in metrics])
        )
        return avg_metrics

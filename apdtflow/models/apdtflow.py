import torch
import torch.nn as nn

from apdtflow.evaluation.regression_evaluator import RegressionEvaluator
from apdtflow.logger_util import get_logger

from .decoder import TimeAwareTransformerDecoder
from .dynamics import NeuralDynamics, solve_latent_ode
from .embedding import TimeSeriesEmbedding
from .fusion import ProbScaleFusion
from .multi_scale_decomposer import ResidualMultiScaleDecomposer

logger = get_logger("apdtflow.models.apdtflow")


class APDTFlow(nn.Module):
    def __init__(
        self,
        num_scales,
        input_channels,
        filter_size,
        hidden_dim,
        output_dim,
        forecast_horizon,
        history_length,
        use_embedding=True,
        num_exog_features=0,
        exog_fusion_type='gated',
        ode_method='rk4',
        decoder_type='transformer',
        n_input_channels=1,
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
          history_length: Length of the input window (T_in); sizes the
            linear residual skip from the input window to the forecast.
          use_embedding: If True, fuse a learnable time embedding into the input.
          num_exog_features: Number of exogenous variables.
          exog_fusion_type: How to fuse exog vars: 'concat', 'gated', or 'attention'.
          ode_method: ODE solver, 'rk4' (default) or 'dopri5_adjoint'.
          decoder_type: 'transformer' (default) or 'continuous'
            (continuous-time ODE decoder enabling predict_at/predict_when).
          n_input_channels: Number of input series channels. When > 1, a
            learned health-indicator fusion layer (Conv1d, kernel 1) fuses
            the channels into one series in front of the pipeline; its
            weights are a readable sensor-importance vector.
        """
        super(APDTFlow, self).__init__()
        self.use_embedding = use_embedding
        self.num_exog_features = num_exog_features
        self.has_exog = num_exog_features > 0
        self.ode_method = ode_method
        self.decoder_type = decoder_type
        self.history_length = history_length
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim
        self.n_input_channels = n_input_channels

        if n_input_channels > 1:
            # Learned health-indicator fusion: n channels -> 1 series.
            self.input_fusion = nn.Conv1d(n_input_channels, 1, kernel_size=1)
        else:
            self.input_fusion = None

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
        self.dynamics_module = NeuralDynamics(hidden_dim=hidden_dim, input_dim=1)
        self.fusion = ProbScaleFusion(hidden_dim, num_scales)

        if decoder_type == 'transformer':
            self.decoder = TimeAwareTransformerDecoder(hidden_dim, output_dim, forecast_horizon)
        elif decoder_type == 'continuous':
            from .continuous_decoder import ContinuousODEDecoder
            self.decoder = ContinuousODEDecoder(
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                history_length=history_length,
                forecast_horizon=forecast_horizon,
            )
        else:
            raise ValueError(
                f"Unknown decoder_type {decoder_type!r}; expected 'transformer' or 'continuous'."
            )

        # N-BEATS/DLinear-style residual: a direct linear map from the input
        # window to the forecast; the deep ODE+transformer path learns the
        # residual. Sized at construction time (never lazy) so that
        # load_state_dict works on freshly constructed models. The continuous
        # decoder carries its own per-step skip, so this one is
        # transformer-only — applying both would double-count the skip and
        # make predict() disagree with predict_at() at integer offsets.
        if decoder_type == 'transformer':
            self.skip_proj = nn.Linear(history_length, forecast_horizon * output_dim)

    def _encode(self, x, t_span, exog=None):
        """Run the encoder path and return the fused latent trajectory.

        Returns:
          Tuple ``(fused_seq, x_orig)`` where ``fused_seq`` has shape
          ``(B, T_in, H)`` and ``x_orig`` is the raw input window used by
          the residual skip.
        """
        if self.input_fusion is not None:
            x = self.input_fusion(x)  # (B, C, T) -> (B, 1, T)
        batch_size, _, T_in = x.size()
        x_orig = x  # raw (fused) window, used by the residual skip
        if self.use_embedding:
            time_indices = t_span.unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
            embedded = self.embedding(time_indices, time_indices)
            projected = self.input_proj(embedded)
            # Fuse the time embedding with the actual series values instead
            # of replacing them.
            x = x + projected.transpose(1, 2)

        if self.has_exog and exog is not None:
            x = self.exog_fusion(x, exog)

        scale_components = self.decomposer(x)

        latent_means = []
        latent_logvars = []
        for comp in scale_components:
            comp_t = comp.transpose(1, 2)
            h0 = torch.zeros(batch_size, self.dynamics_module.hidden_dim, device=x.device)
            logvar0 = torch.zeros(batch_size, self.dynamics_module.hidden_dim, device=x.device)
            h_sol, logvar_sol = solve_latent_ode(
                self.dynamics_module, h0, logvar0, t_span, comp_t,
                ode_method=self.ode_method,
            )
            # Keep the FULL latent trajectory (B, T_in, H) so the decoder
            # can attend over the whole encoded window.
            latent_means.append(h_sol)
            latent_logvars.append(logvar_sol)

        fused_seq = self.fusion(latent_means, latent_logvars)  # (B, T_in, H)
        return fused_seq, x_orig

    def forward(self, x, t_span, exog=None):
        """
        Args:
          x: Input tensor of shape (batch, 1, T_in).
          t_span: 1D tensor of time indices of length T_in (normalized to [0,1]).
          exog: Optional exogenous variables of shape (batch, num_exog, T_in).
        Returns:
          Tuple (outputs, out_logvars) each of shape (batch, forecast_horizon, output_dim).
        """
        batch_size = x.size(0)
        fused_seq, x_orig = self._encode(x, t_span, exog=exog)

        if self.decoder_type == 'continuous':
            h_T = fused_seq[:, -1, :]
            outputs, out_logvars = self.decoder(h_T, x_orig.squeeze(1))
        else:
            memory = fused_seq.transpose(0, 1)  # (T_in, B, H)
            outputs, out_logvars = self.decoder(memory)
            skip = self.skip_proj(x_orig.squeeze(1)).view(
                batch_size, self.forecast_horizon, self.output_dim
            )
            outputs = outputs + skip
        return outputs, out_logvars

    def forward_at(self, x, t_span, query_offsets, exog=None):
        """Forecast at arbitrary real-valued time offsets.

        Requires ``decoder_type='continuous'``. Offsets are in forecast
        steps: 1.0 is one step past the end of the input window; values
        beyond ``forecast_horizon`` extrapolate past the trained horizon.

        Args:
          x: Input tensor of shape (batch, 1, T_in).
          t_span: 1D tensor of time indices of length T_in.
          query_offsets: 1D tensor of positive float offsets.
          exog: Optional exogenous variables of shape (batch, num_exog, T_in).

        Returns:
          Tuple (values, logvars) of shape (batch, len(query_offsets), output_dim).
        """
        if self.decoder_type != 'continuous':
            raise RuntimeError(
                "forward_at requires decoder_type='continuous'; this model was "
                f"built with decoder_type={self.decoder_type!r}."
            )
        fused_seq, x_orig = self._encode(x, t_span, exog=exog)
        h_T = fused_seq[:, -1, :]
        return self.decoder(h_T, x_orig.squeeze(1), query_offsets)

    def _continuous_training_loss(self, x_batch, y_batch, t_span, exog=None):
        """Randomized-query training loss for the continuous decoder.

        Samples query times uniformly in (0, horizon] each batch and
        linearly interpolates targets, so off-grid accuracy is trained
        rather than emergent. Combined with the grid loss to keep integer
        offsets sharp.
        """
        device = x_batch.device
        horizon = self.forecast_horizon
        y = y_batch.transpose(1, 2)  # (B, H, out)

        # Grid loss at integer offsets.
        preds_grid, _ = self(x_batch, t_span, exog=exog)
        loss_grid = ((preds_grid - y) ** 2).mean()

        # Randomized off-grid queries. The target at offset 0 is the last
        # observed value, allowing interpolation in (0, 1].
        taus = (torch.rand(horizon, device=device) * horizon).clamp(min=1e-3)
        preds_rand, _ = self.forward_at(x_batch, t_span, taus, exog=exog)
        anchor = x_batch[:, 0, -1].reshape(-1, 1, 1)  # (B, 1, 1)
        grid_targets = torch.cat([anchor, y], dim=1)  # offsets 0..H
        base = taus.floor().clamp(max=horizon - 1).long()  # in [0, H-1]
        frac = (taus - base.float()).reshape(1, -1, 1)
        lo = grid_targets[:, base, :]
        hi = grid_targets[:, base + 1, :]
        rand_targets = lo * (1 - frac) + hi * frac
        loss_rand = ((preds_rand - rand_targets) ** 2).mean()

        return 0.5 * (loss_grid + loss_rand)

    def train_model(self, train_loader, num_epochs, learning_rate, device, loss_type="mse"):
        """Train the model.

        Args:
          loss_type: "mse" (default) or "nll". Under MSE training the
            log-variance head is untrained and its values are meaningless;
            uncertainty must then come from conformal calibration.
        """
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0.0
            for batch_data in train_loader:
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
                if self.decoder_type == 'continuous':
                    loss = self._continuous_training_loss(
                        x_batch, y_batch, t_span, exog=exog_x
                    )
                else:
                    preds, pred_logvars = self(x_batch, t_span, exog=exog_x)
                    mse = (preds - y_batch.transpose(1, 2)) ** 2
                    if loss_type == "nll":
                        loss = torch.mean(0.5 * (mse / (pred_logvars.exp() + 1e-6)) + 0.5 * pred_logvars)
                    else:
                        loss = mse.mean()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_size
            logger.info(
                "Epoch %d/%d, Loss: %.4f",
                epoch + 1, num_epochs, epoch_loss / len(train_loader.dataset),
            )
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

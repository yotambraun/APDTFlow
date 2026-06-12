import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

ODE_METHODS = ("rk4", "dopri5_adjoint")


class NeuralDynamics(nn.Module):
    """Neural ODE dynamics for the latent state.

    Maps ``(t, (h, x_t))`` to the time-derivatives of the latent mean and
    log-variance.
    """

    def __init__(self, hidden_dim, input_dim):
        super(NeuralDynamics, self).__init__()
        self.hidden_dim = hidden_dim
        self.global_net = nn.Sequential(
            nn.Linear(hidden_dim + input_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim * 2),
        )

    def forward(self, t, state):
        h, x_scale = state
        t_scalar = t if t.dim() == 0 else t[0]
        time_feature = t_scalar.unsqueeze(0).expand(h.size(0), 1)
        inp = torch.cat([h, x_scale, time_feature], dim=-1)
        update = self.global_net(inp)
        delta_mu, delta_logvar = torch.chunk(update, 2, dim=-1)
        return delta_mu, delta_logvar


# Deprecated alias, kept for one release (the class was renamed in v0.4.0).
HierarchicalNeuralDynamics = NeuralDynamics


def solve_latent_ode(dynamics, h0, logvar0, t_span, x_scale_sequence, ode_method="rk4"):
    """Integrate the latent Neural ODE over ``t_span``.

    Args:
        dynamics: A :class:`NeuralDynamics` instance.
        h0: Initial hidden state tensor of shape ``(B, H)``.
        logvar0: Initial log-variance tensor of shape ``(B, H)``.
        t_span: 1D tensor of time points for integration.
        x_scale_sequence: Per-time-step conditioning of shape ``(B, T, C)``.
        ode_method: ``"rk4"`` (fixed-step, default, ~10x faster on CPU) or
            ``"dopri5_adjoint"`` (adaptive with adjoint backprop, lower
            memory for long sequences).

    Returns:
        Tuple ``(h_sol, logvar_sol)``, each of shape ``(B, T, H)`` — the
        full latent trajectory.
    """

    class ODEFunc(nn.Module):
        def __init__(self, dynamics, x_sequence, t_span):
            super(ODEFunc, self).__init__()
            self.dynamics = dynamics
            self.x_sequence = x_sequence
            self.t_span = t_span

        def forward(self, t, state):
            h, logvar = state
            idx = (torch.abs(self.t_span - t)).argmin()
            x_t = self.x_sequence[:, idx, :]
            return self.dynamics(t, (h, x_t))

    func = ODEFunc(dynamics, x_scale_sequence, t_span)
    state0 = (h0, logvar0)
    if ode_method == "rk4":
        sol = odeint(func, state0, t_span, method="rk4")
    elif ode_method == "dopri5_adjoint":
        sol = odeint_adjoint(func, state0, t_span, rtol=1e-3, atol=1e-3)
    else:
        raise ValueError(
            f"Unknown ode_method {ode_method!r}; expected one of {ODE_METHODS}."
        )
    h_sol = sol[0].transpose(0, 1)
    logvar_sol = sol[1].transpose(0, 1)
    return h_sol, logvar_sol


# Deprecated alias, kept for one release (the function was renamed in v0.4.0).
adaptive_hierarchical_ode_solver = solve_latent_ode

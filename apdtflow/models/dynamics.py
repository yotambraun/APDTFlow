import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint


class HierarchicalNeuralDynamics(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(HierarchicalNeuralDynamics, self).__init__()
        self.hidden_dim = hidden_dim  
        
        self.global_net = nn.Sequential(
            nn.Linear(hidden_dim + input_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim * 2),
        )
        self.local_net = nn.Sequential(
            nn.Linear(hidden_dim + input_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim * 2),
        )

    def forward(self, t, state, level="global"):
        h, x_scale = state
        t_scalar = t if t.dim() == 0 else t[0]
        time_feature = t_scalar.unsqueeze(0).expand(h.size(0), 1)
        inp = torch.cat([h, x_scale, time_feature], dim=-1)
        if level == "global":
            update = self.global_net(inp)
        else:
            update = self.local_net(inp)
        delta_mu, delta_logvar = torch.chunk(update, 2, dim=-1)
        return delta_mu, delta_logvar


def adaptive_hierarchical_ode_solver(dynamics, h0, logvar0, t_span, x_scale_sequence):
    """
    Solves the ODE for hierarchical dynamics using odeint_adjoint.
    Currently, only the 'global' level is used.
    
    Args:
        dynamics: An instance of HierarchicalNeuralDynamics.
        h0: Initial hidden state tensor.
        logvar0: Initial log-variance tensor.
        t_span: 1D tensor of time points for integration.
        x_scale_sequence: A tensor containing the scale information for each time point.
    
    Returns:
        A tuple (h_sol, logvar_sol) with the integrated solutions.
    """

    class ODEFunc(nn.Module):
        def __init__(self, dynamics, x_sequence, t_span, level):
            super(ODEFunc, self).__init__()
            self.dynamics = dynamics
            self.x_sequence = x_sequence
            self.t_span = t_span
            self.level = level

        def forward(self, t, state):
            h, logvar = state
            idx = (torch.abs(self.t_span - t)).argmin()
            x_t = self.x_sequence[:, idx, :]
            return self.dynamics(t, (h, x_t), level=self.level)

    global_func = ODEFunc(dynamics, x_scale_sequence, t_span, level="global")
    state0 = (h0, logvar0)
    sol_global = odeint_adjoint(global_func, state0, t_span, rtol=1e-3, atol=1e-3)
    h_sol = sol_global[0].transpose(0, 1)
    logvar_sol = sol_global[1].transpose(0, 1)
    return h_sol, logvar_sol

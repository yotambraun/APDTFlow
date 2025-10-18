import torch
from apdtflow.models.dynamics import (
    HierarchicalNeuralDynamics,
    adaptive_hierarchical_ode_solver,
)


def test_neural_dynamics_forward():
    batch_size = 2
    hidden_dim = 8
    input_dim = 1
    dynamics = HierarchicalNeuralDynamics(hidden_dim=hidden_dim, input_dim=input_dim)
    h = torch.randn(batch_size, hidden_dim)
    x_scale = torch.randn(batch_size, input_dim)
    t = torch.tensor(0.5)
    delta_mu, delta_logvar = dynamics(t, (h, x_scale), level="global")
    assert delta_mu.shape == (batch_size, hidden_dim)
    assert delta_logvar.shape == (batch_size, hidden_dim)


def test_adaptive_ode_solver():
    batch_size = 2
    T_in = 10
    hidden_dim = 8
    input_dim = 1
    dynamics = HierarchicalNeuralDynamics(hidden_dim=hidden_dim, input_dim=input_dim)
    h0 = torch.zeros(batch_size, hidden_dim)
    logvar0 = torch.zeros(batch_size, hidden_dim)
    t_span = torch.linspace(0, 1, steps=T_in)
    x_seq = torch.randn(batch_size, T_in, input_dim)
    h_sol, logvar_sol = adaptive_hierarchical_ode_solver(
        dynamics, h0, logvar0, t_span, x_seq
    )
    assert h_sol.shape == (batch_size, T_in, hidden_dim)
    assert logvar_sol.shape == (batch_size, T_in, hidden_dim)

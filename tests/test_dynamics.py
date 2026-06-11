import pytest
import torch

from apdtflow.models.dynamics import (
    HierarchicalNeuralDynamics,
    NeuralDynamics,
    adaptive_hierarchical_ode_solver,
    solve_latent_ode,
)


def test_neural_dynamics_forward():
    batch_size = 2
    hidden_dim = 8
    input_dim = 1
    dynamics = NeuralDynamics(hidden_dim=hidden_dim, input_dim=input_dim)
    h = torch.randn(batch_size, hidden_dim)
    x_scale = torch.randn(batch_size, input_dim)
    t = torch.tensor(0.5)
    delta_mu, delta_logvar = dynamics(t, (h, x_scale))
    assert delta_mu.shape == (batch_size, hidden_dim)
    assert delta_logvar.shape == (batch_size, hidden_dim)


def test_deprecated_aliases():
    assert HierarchicalNeuralDynamics is NeuralDynamics
    assert adaptive_hierarchical_ode_solver is solve_latent_ode


@pytest.mark.parametrize("ode_method", ["rk4", "dopri5_adjoint"])
def test_latent_ode_solver(ode_method):
    batch_size = 2
    T_in = 10
    hidden_dim = 8
    input_dim = 1
    dynamics = NeuralDynamics(hidden_dim=hidden_dim, input_dim=input_dim)
    h0 = torch.zeros(batch_size, hidden_dim)
    logvar0 = torch.zeros(batch_size, hidden_dim)
    t_span = torch.linspace(0, 1, steps=T_in)
    x_seq = torch.randn(batch_size, T_in, input_dim)
    h_sol, logvar_sol = solve_latent_ode(
        dynamics, h0, logvar0, t_span, x_seq, ode_method=ode_method
    )
    assert h_sol.shape == (batch_size, T_in, hidden_dim)
    assert logvar_sol.shape == (batch_size, T_in, hidden_dim)


def test_unknown_ode_method_raises():
    dynamics = NeuralDynamics(hidden_dim=4, input_dim=1)
    h0 = torch.zeros(1, 4)
    logvar0 = torch.zeros(1, 4)
    t_span = torch.linspace(0, 1, steps=5)
    x_seq = torch.randn(1, 5, 1)
    with pytest.raises(ValueError, match="ode_method"):
        solve_latent_ode(dynamics, h0, logvar0, t_span, x_seq, ode_method="euler5")

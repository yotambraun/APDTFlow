"""ODE-RNN encoder gate experiment — a documented NEGATIVE result.

RESEARCH TRACK ONLY. This prototype exists to keep the spec Section 11.2
finding reproducible (APDTFlow_MASTER_TRANSFORMATION.md): a latent-ODE /
ODE-RNN encoder (Rubanova et al. 2019 style — the latent state evolves
through observation gaps by an ODE and is updated by a GRU cell ONLY at
observed points) was implemented and gated at ~50% bursty missingness, and
LOST to every simple baseline (prototype numbers: ODE-RNN 2.10 vs
deep-grid+ffill 1.98 vs seasonal-naive 1.72 vs linear+ffill 1.50). This is
consistent with the published literature questioning ODE-based encoders for
irregular series (arXiv:2505.00590). Consequently there is NO
``model_type='latent_ode'`` release feature and NO irregular-data claims
anywhere in APDTFlow; do not promote this code to the package without a
benchmark win on this gate.

Setup: the Section 11.1 synthetic series (trend + daily + weekly cycles),
~50% bursty missingness, 80/20 split. Contestants, all scored on held-out
windows against the TRUE (pre-masking) series:

    - ODE-RNN encoder (torchdiffeq, RK4 between observations) + the
      production ContinuousODEDecoder; trained on TRUE targets — a
      deliberate advantage, so a loss here is conclusive
    - production APDTFlowForecaster on the forward-filled series
      (targets are the filled series — all it would see in practice)
    - seasonal-naive and linear regression on the forward-filled series

Outputs a results table to stdout and JSON to
``experiments/results/odernn_gate.json``.

Usage:
    python experiments/prototypes/odernn_gate.py              # full run
    python experiments/prototypes/odernn_gate.py --epochs 1   # smoke run
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchdiffeq import odeint

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from apdtflow import APDTFlowForecaster  # noqa: E402
from apdtflow.models.continuous_decoder import ContinuousODEDecoder  # noqa: E402

warnings.filterwarnings("ignore")

N = 1200
T_IN, T_OUT = 36, 12
SEASON = 24
HIDDEN = 32
RESULTS_DIR = REPO / "experiments" / "results"


# --------------------------------------------------------------------------
# Data: the Section 11.1 series with ~50% bursty missingness
# --------------------------------------------------------------------------

def make_data(rng):
    t = np.arange(N)
    true = (0.02 * t + 4 * np.sin(2 * np.pi * t / 24)
            + 2 * np.sin(2 * np.pi * t / 168) + rng.normal(0, 0.5, N))
    obs = np.ones(N, dtype=bool)
    i = 0
    while i < N:
        if rng.random() < 0.10:           # gap-start probability
            gap = rng.geometric(0.10)     # geometric gap lengths -> ~50%
            obs[i:i + gap] = False
            i += gap
        else:
            i += 1
    filled = np.empty(N)
    last = true[0]
    for i in range(N):
        if obs[i]:
            last = true[i]
        filled[i] = last
    return true, obs, filled


# --------------------------------------------------------------------------
# ODE-RNN encoder (Rubanova et al. 2019 style)
# --------------------------------------------------------------------------

class ODERNNEncoder(nn.Module):
    """Latent state evolves by an ODE through gaps; GRU update only at
    observed points. The inter-observation flow is integrated with
    torchdiffeq's fixed-step RK4."""

    def __init__(self, hidden_dim=HIDDEN):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.f = nn.Sequential(
            nn.Linear(hidden_dim + 1, 64), nn.Tanh(), nn.Linear(64, hidden_dim))
        self.gru = nn.GRUCell(1, hidden_dim)

    def _dynamics(self, t, h):
        t_feature = t.reshape(1, 1).expand(h.size(0), 1)
        return self.f(torch.cat([h, t_feature], dim=-1))

    def forward(self, vals, mask, t_span):
        """vals: (B, T) observed values (positions with mask=0 are ignored);
        mask: (B, T) 1=observed; t_span: (T,) normalized times."""
        B, T = vals.shape
        h = torch.zeros(B, self.hidden_dim, device=vals.device)
        dt = float(t_span[1] - t_span[0])
        for i in range(T):
            if i > 0:
                # Evolve the latent state through the gap [t_{i-1}, t_i].
                h = odeint(self._dynamics, h, t_span[i - 1:i + 1],
                           method="rk4", options={"step_size": dt / 2})[-1]
            # Update ONLY where a real observation exists.
            upd = self.gru(vals[:, i:i + 1], h)
            m = mask[:, i:i + 1]
            h = m * upd + (1 - m) * h
        return h


class LatentODEForecaster(nn.Module):
    """ODE-RNN encoder + the production continuous ODE decoder (which
    carries its own input-window skip, same as the grid model — fair)."""

    def __init__(self, t_in=T_IN, t_out=T_OUT):
        super().__init__()
        self.encoder = ODERNNEncoder(HIDDEN)
        self.decoder = ContinuousODEDecoder(
            hidden_dim=HIDDEN, output_dim=1,
            history_length=t_in, forecast_horizon=t_out)

    def forward(self, x_filled, mask):
        # x_filled: (B, 1, T) forward-filled window (== true where mask=1)
        T = x_filled.size(-1)
        t_span = torch.linspace(0, 1, T, device=x_filled.device)
        h = self.encoder(x_filled.squeeze(1), mask, t_span)
        values, logvars = self.decoder(h, x_filled.squeeze(1))  # (B, T_out, 1)
        return values, logvars


# --------------------------------------------------------------------------
# Windows / baselines
# --------------------------------------------------------------------------

def window_starts(lo, hi):
    return range(lo, hi - T_IN - T_OUT)


def make_windows(arr, lo, hi):
    return np.asarray([arr[i:i + T_IN] for i in window_starts(lo, hi)],
                      dtype=np.float32)


def make_targets(arr, lo, hi):
    return np.asarray(
        [arr[i + T_IN:i + T_IN + T_OUT] for i in window_starts(lo, hi)],
        dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=40,
                        help="training epochs for both models (default 40)")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    true, obs, filled = make_data(rng)
    split = int(N * 0.8)
    print(f"missing fraction: {1 - obs.mean():.1%} "
          f"({N} points, train on first {split})")

    mean, std = filled[:split].mean(), filled[:split].std()
    nf = (filled - mean) / std        # normalized filled series (inputs)
    nt = (true - mean) / std          # normalized true series (targets)
    maskf = obs.astype(np.float32)

    Xtr = torch.tensor(make_windows(nf, 0, split)).unsqueeze(1)
    Ytr = torch.tensor(make_targets(nt, 0, split)).unsqueeze(-1)
    Mtr = torch.tensor(make_windows(maskf, 0, split))
    Xte = torch.tensor(make_windows(nf, split, N)).unsqueeze(1)
    Mte = torch.tensor(make_windows(maskf, split, N))
    A = make_targets(true, split, N)  # held-out truth, original scale

    res = {}

    # ---- ODE-RNN (trained on TRUE targets: deliberate advantage) ----------
    model = LatentODEForecaster()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr, Ytr, Mtr),
        batch_size=64, shuffle=True)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for xb, yb, mb in loader:
            opt.zero_grad()
            p, _ = model(xb, mb)
            loss = ((p - yb) ** 2).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        if args.verbose:
            print(f"  ODE-RNN epoch {ep + 1}/{args.epochs} "
                  f"loss {tot / len(Xtr):.4f}")
    print(f"ODE-RNN trained in {time.time() - t0:.1f}s "
          f"(final loss {tot / len(Xtr):.4f})")

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(Xte), 256):
            p, _ = model(Xte[start:start + 256], Mte[start:start + 256])
            preds.append(p.squeeze(-1).numpy())
    P = np.concatenate(preds) * std + mean
    res["ODE-RNN encoder"] = float(np.abs(P - A).mean())

    # ---- production APDTFlowForecaster on the forward-filled series -------
    t0 = time.time()
    forecaster = APDTFlowForecaster(
        forecast_horizon=T_OUT, history_length=T_IN, hidden_dim=HIDDEN,
        num_scales=3, filter_size=5, learning_rate=1e-3, batch_size=64,
        num_epochs=args.epochs, verbose=args.verbose)
    forecaster.fit(pd.DataFrame({"y": filled[:split]}), target_col="y")
    print(f"APDTFlow(ffill) trained in {time.time() - t0:.1f}s")
    nf2 = (filled - forecaster.scaler_mean_) / forecaster.scaler_std_
    Xte2 = torch.tensor(make_windows(nf2, split, N)).unsqueeze(1)
    t_span = torch.linspace(0, 1, T_IN, device=forecaster.device)
    preds = []
    forecaster.model.eval()
    with torch.no_grad():
        for start in range(0, len(Xte2), 256):
            p, _ = forecaster.model(
                Xte2[start:start + 256].to(forecaster.device), t_span)
            preds.append(p.squeeze(-1).cpu().numpy())
    P = (np.concatenate(preds) * forecaster.scaler_std_
         + forecaster.scaler_mean_)
    res["APDTFlow + ffill"] = float(np.abs(P - A).mean())

    # ---- baselines on the forward-filled series ----------------------------
    Xf = make_windows(nf, 0, split)
    Yf = make_targets(nt, 0, split)  # true targets, same advantage as ODE-RNN
    Xe = make_windows(nf, split, N)
    W, *_ = np.linalg.lstsq(
        np.hstack([Xf, np.ones((len(Xf), 1))]), Yf, rcond=None)
    Pl = (np.hstack([Xe, np.ones((len(Xe), 1))]) @ W) * std + mean
    res["linear + ffill"] = float(np.abs(Pl - A).mean())

    sn = Xe[:, -SEASON:-SEASON + T_OUT] * std + mean
    res["seasonal-naive + ffill"] = float(np.abs(sn - A).mean())

    # ---- table -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"ODE-RNN gate @ {1 - obs.mean():.0%} missingness - held-out MAE "
          f"vs TRUE series (epochs {args.epochs})")
    print("=" * 60)
    ranked = sorted(res.items(), key=lambda kv: kv[1])
    for name, mae in ranked:
        print(f"  {name:26s} MAE {mae:7.3f}")
    print("-" * 60)
    if ranked[0][0] == "ODE-RNN encoder":
        print("NOTE: ODE-RNN won on this run - the Section 11.2 gate did "
              "not reproduce; re-examine before changing its status.")
    else:
        print("NEGATIVE RESULT CONFIRMED: the ODE-RNN encoder does not beat "
              "simple baselines (spec Section 11.2; research track only).")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "odernn_gate.json"
    payload = {
        "meta": {
            "script": "experiments/prototypes/odernn_gate.py",
            "status": "research track only — NOT a release feature "
                      "(spec Section 11.2; arXiv:2505.00590)",
            "missing_fraction": float(1 - obs.mean()),
            "horizon": T_OUT,
            "history": T_IN,
            "epochs": args.epochs,
            "seed": args.seed,
            "n_test_windows": int(len(Xte)),
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "mae_vs_true": res,
        "odernn_beats_all_baselines": ranked[0][0] == "ODE-RNN encoder",
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()

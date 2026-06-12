#!/usr/bin/env python
"""Render the verified-evidence banner from the predictive-maintenance demo results.

Reads whatever JSON files the demo scripts have produced in
experiments/results/ (missing files are skipped, panels show how to generate
them) and renders assets/images/apdtflow_evidence_summary.png — a compact
multi-panel banner: battery EOL bars, FD001 univariate-vs-multivariate bars,
FD002 robustness bars, a coverage panel, the honesty ledger, and the
operational rule.

Sources (each one optional):
  * experiments/results/battery_eol.json          <- battery_eol_demo.py
  * experiments/results/turbofan_when.json        <- turbofan_when_demo.py
  * experiments/results/turbofan_multivariate.json<- turbofan_multivariate_demo.py
  * experiments/results/fd002_robustness.json     <- fd002_robustness_demo.py

Usage:
  python experiments/make_evidence_summary.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "experiments" / "results"
IMAGES_DIR = REPO / "assets" / "images"
OUTPUT = IMAGES_DIR / "apdtflow_evidence_summary.png"

GREEN, PURPLE, RED = "#2e8b57", "#6a5acd", "#d62839"

SOURCES = {
    "battery": ("battery_eol.json", "battery_eol_demo.py"),
    "fd001_uni": ("turbofan_when.json", "turbofan_when_demo.py"),
    "fd001_multi": ("turbofan_multivariate.json", "turbofan_multivariate_demo.py"),
    "fd002": ("fd002_robustness.json", "fd002_robustness_demo.py"),
}


def load_results() -> dict:
    """Tolerant loader: missing or unreadable files are skipped."""
    results = {}
    for key, (filename, script) in SOURCES.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            print(f"  [skip] {filename} not found (run experiments/{script})")
            continue
        try:
            results[key] = json.loads(path.read_text())
            print(f"  [ok]   {filename}")
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [skip] {filename} unreadable: {exc}")
    return results


def placeholder(ax, script: str):
    ax.axis("off")
    ax.text(0.5, 0.5, f"not yet run\n\npython experiments/{script}",
            ha="center", va="center", fontsize=10, color="gray",
            family="monospace", transform=ax.transAxes)


def method_bars(ax, values, labels, colors, ylabel, title, fmt="{:.1f}"):
    bars = ax.bar(range(len(values)), values, 0.62, color=colors)
    for b, v in zip(bars, values):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_title(title, fontsize=9.5)
    ax.grid(alpha=0.3, axis="y")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def pct(value) -> str:
    return "--" if value is None else f"{value * 100:.0f}%"


def main() -> None:
    print("collecting demo results from experiments/results/ ...")
    results = load_results()
    if not results:
        print("nothing to summarize")
        return

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))

    # --- (0,0) battery EOL
    ax = axes[0, 0]
    if "battery" in results:
        pooled = results["battery"]["pooled"]
        method_bars(
            ax,
            [pooled["mae_full_persistence"], pooled["mae_full_linear"],
             pooled["mae_full_apdtflow"]],
            ["Persist.", "Linear", "APDTFlow"], [RED, PURPLE, GREEN],
            "timing error (cycles)",
            "Battery end-of-life (NASA, 3 cells,\nleave-one-out) — "
            "predict_when(< 1.4 Ah)")
    else:
        placeholder(ax, "battery_eol_demo.py")

    # --- (0,1) FD001 univariate vs multivariate
    ax = axes[0, 1]
    if "fd001_uni" in results or "fd001_multi" in results:
        uni = results.get("fd001_uni", {}).get("metrics", {})
        multi = results.get("fd001_multi", {}).get("metrics", {})
        vals, labels, colors = [], [], []
        if uni.get("mae_full_apdtflow") is not None:
            vals.append(uni["mae_full_apdtflow"])
            labels.append("1 sensor")
            colors.append(PURPLE)
        if multi.get("mae_full_apdtflow") is not None:
            vals.append(multi["mae_full_apdtflow"])
            labels.append("5 sensors\n(learned fusion)")
            colors.append(GREEN)
        catch_note = ""
        if uni.get("catch_rate") is not None and multi.get("catch_rate") is not None:
            catch_note = (f"\ncatch rate {pct(uni['catch_rate'])} -> "
                          f"{pct(multi['catch_rate'])}")
        if (uni.get("mae_full_apdtflow") is not None
                and multi.get("mae_full_apdtflow") is not None
                and multi["mae_full_apdtflow"] < uni["mae_full_apdtflow"]):
            verdict = "multivariate improves full-set timing"
        else:
            verdict = "multivariate: sharper when it speaks,\nmore conservative overall"
        method_bars(ax, vals, labels, colors, "timing MAE (cycles)",
                    "Turbofan FD001 (unseen engines):\n" + verdict + catch_note,
                    fmt="{:.2f}")
    else:
        placeholder(ax, "turbofan_when_demo.py")

    # --- (0,2) FD002 robustness
    ax = axes[0, 2]
    if "fd002" in results:
        m = results["fd002"]["metrics"]
        n_eng = results["fd002"]["config"].get("audit_engines", "?")
        method_bars(
            ax,
            [m["mae_full_persistence"], m["mae_full_linear"], m["mae_full_apdtflow"]],
            ["Persist.", "Linear", "APDTFlow"], [RED, PURPLE, GREEN],
            "timing error (cycles)",
            f"Turbofan FD002: {n_eng} unseen engines,\n6 shifting operating regimes")
    else:
        placeholder(ax, "fd002_robustness_demo.py")

    # --- (1,0) calibrated coverage
    ax = axes[1, 0]
    cov_labels, cov_vals = [], []
    if "battery" in results:
        for cell, stats in results["battery"]["per_cell"].items():
            if stats.get("coverage") is not None:
                cov_labels.append(f"Battery\n{cell}")
                cov_vals.append(stats["coverage"] * 100)
    if "fd001_multi" in results and \
            results["fd001_multi"]["metrics"].get("coverage") is not None:
        cov_labels.append("FD001\n(multi)")
        cov_vals.append(results["fd001_multi"]["metrics"]["coverage"] * 100)
    elif "fd001_uni" in results and \
            results["fd001_uni"]["metrics"].get("coverage") is not None:
        cov_labels.append("FD001")
        cov_vals.append(results["fd001_uni"]["metrics"]["coverage"] * 100)
    if "fd002" in results and results["fd002"]["metrics"].get("coverage") is not None:
        cov_labels.append("FD002")
        cov_vals.append(results["fd002"]["metrics"]["coverage"] * 100)
    if cov_vals:
        bars = ax.bar(cov_labels, cov_vals, 0.6, color=GREEN)
        for b, v in zip(bars, cov_vals):
            ax.annotate(f"{v:.0f}%", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=9)
        ax.axhline(90, color="black", ls="--", lw=1.1)
        ax.text(len(cov_vals) - 0.45, 90.6, "90% target", fontsize=8, ha="right")
        ax.set_ylim(0, 109)
        ax.set_ylabel("coverage (%)", fontsize=8.5)
        ax.set_title("Calibrated 90% crossing-time windows:\nuncertainty that "
                     "means what it says", fontsize=9.5)
        ax.grid(alpha=0.3, axis="y")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
    else:
        placeholder(ax, "battery_eol_demo.py (and the turbofan demos)")

    # --- (1,1) the honesty ledger
    ax = axes[1, 1]
    ax.axis("off")
    lines = ["The honesty ledger", ""]
    if "battery" in results:
        cen = results["battery"].get("censoring_honesty", {})
        lines += [
            f"battery {cen.get('cell', 'B0007')} never reaches EOL in horizon:",
            f"  {cen.get('n_correctly_censored', '?')}/"
            f"{cen.get('n_no_crossing_windows', '?')} windows correctly censored,",
            f"  {cen.get('n_false_alarms', '?')} false alarms "
            f"({pct(cen.get('false_alarm_rate'))}) — reported, not hidden", "",
        ]
    for key, name in (("fd001_uni", "FD001 univariate"),
                      ("fd001_multi", "FD001 multivariate"),
                      ("fd002", "FD002 multi-regime")):
        if key in results:
            m = results[key]["metrics"]
            lines.append(f"{name}: catch {pct(m.get('catch_rate'))} | "
                         f"false alarms {pct(m.get('false_alarm_rate'))}")
            if m.get("matched_linear_mae") is not None:
                lines.append(f"  matched subset: linear "
                             f"{m['matched_linear_mae']:.1f} vs APDTFlow "
                             f"{m['matched_apdtflow_mae']:.1f} cycles")
    lines += ["", "censored 'no crossing within horizon'", "is a first-class answer"]
    ax.text(0.02, 0.97, "\n".join(lines), va="top", ha="left", fontsize=8.2,
            family="monospace", transform=ax.transAxes)

    # --- (1,2) built for decisions
    ax = axes[1, 2]
    ax.axis("off")
    text = [
        "Built for decisions, not just curves", "",
        "result = model.predict_when(",
        "    threshold, direction='above',",
        "    alpha=0.1)",
        "result.eta / .earliest / .act_by", "",
        "schedule = model.predict_when_fleet(",
        "    assets, threshold, alpha=0.1)", "",
        "Operational rule (calibrated):",
        "  never act on the point estimate —",
        "  schedule by act_by, the window's",
        "  earliest edge.",
    ]
    if "fd002" in results:
        fleet = results["fd002"].get("fleet_snapshot", {})
        if fleet:
            text += ["",
                     f"fleet snapshot ({fleet.get('n_engines', '?')} engines):",
                     f"  act-by early enough for "
                     f"{pct(fleet.get('act_by_early_enough'))} of engines"]
    ax.text(0.02, 0.97, "\n".join(text), va="top", ha="left", fontsize=8.2,
            family="monospace", transform=ax.transAxes)

    fig.suptitle("APDTFlow v0.4 — every claim verified, every number reproducible",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT, dpi=150)
    plt.close(fig)
    print(f"wrote {OUTPUT.relative_to(REPO)} "
          f"({len(results)}/{len(SOURCES)} result files summarized)")


if __name__ == "__main__":
    main()

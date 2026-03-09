#!/usr/bin/env python
"""Plot experiment summary panels from stored JSON artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def labelize(name: str) -> str:
    replacements = {
        "HSVAE-": "",
        "SimpleSparseBaseline": "Baseline",
        "SCVAE-lista": "SC-VAE/LISTA",
        "lista-softft": "LISTA soft-ft",
        "librimix_": "",
        "_": " ",
        "binary baseline": "v4 bin base",
        "binary n256 s8": "v4 256 s8",
        "binary n512 s8": "v4 512 s8",
        "binary n256": "v4 256",
    }
    text = name
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _norm_column(values: list[float | None], *, higher_is_better: bool = True) -> np.ndarray:
    arr = np.array([np.nan if value is None else float(value) for value in values], dtype=float)
    valid = ~np.isnan(arr)
    out = np.full_like(arr, np.nan)
    if valid.sum() == 0:
        return out
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if math.isclose(float(lo), float(hi)):
        out[valid] = 1.0
        return out
    scaled = (arr - lo) / (hi - lo)
    if not higher_is_better:
        scaled = 1.0 - scaled
    out[valid] = scaled[valid]
    return out


def plot_sinusoid_spectrum(sinusoid_rows: list[dict[str, Any]], output_path: Path) -> None:
    ordered = sorted(sinusoid_rows, key=lambda row: row["condition"])
    labels = [labelize(row["condition"]) for row in ordered]
    raw_cols = {
        "support_f1": [safe_float(row.get("support_f1_mean")) for row in ordered],
        "topk_f1": [safe_float(row.get("topk_f1_mean")) for row in ordered],
        "amp_corr": [safe_float(row.get("amplitude_corr_mean")) for row in ordered],
        "recon_inv": [safe_float(row.get("recon_mse_mean")) for row in ordered],
        "sparsity": [safe_float(row.get("sparsity_mean")) for row in ordered],
    }
    heatmap = np.vstack(
        [
            _norm_column(raw_cols["support_f1"], higher_is_better=True),
            _norm_column(raw_cols["topk_f1"], higher_is_better=True),
            _norm_column(raw_cols["amp_corr"], higher_is_better=True),
            _norm_column(raw_cols["recon_inv"], higher_is_better=False),
            _norm_column(raw_cols["sparsity"], higher_is_better=True),
        ]
    ).T

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#e6e6e6")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    im = ax.imshow(heatmap, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(5), ["support_f1", "topk_f1", "amp_corr", "low_recon", "sparsity"], rotation=20, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Sinusoid Sparse-Recovery Spectrum (JSON summaries)")

    annotation_keys = ["support_f1_mean", "topk_f1_mean", "amplitude_corr_mean", "recon_mse_mean", "sparsity_mean"]
    for i, row in enumerate(ordered):
        for j, key in enumerate(annotation_keys):
            value = safe_float(row.get(key))
            text = "-" if value is None else f"{value:.3f}"
            ax.text(j, i, text, ha="center", va="center", color="white" if not np.isnan(heatmap[i, j]) and heatmap[i, j] > 0.45 else "black", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("within-panel normalized score")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cross_experiment_tradeoffs(
    sinusoid_rows: list[dict[str, Any]],
    mnist_seed_rows: list[dict[str, Any]],
    fsdd_rows: list[dict[str, Any]],
    librimix_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # Sinusoid support vs active atoms.
    ax = axes[0, 0]
    for row in sinusoid_rows:
        if row["condition"] == "SimpleSparseBaseline":
            continue
        x = safe_float(row.get("n_active_frame_mean"))
        y = safe_float(row.get("support_f1_mean"))
        if x is None or y is None:
            continue
        size = 120 + 300 * (safe_float(row.get("amplitude_corr_mean")) or 0.0)
        color = "#d95f02" if row.get("kl_normalization") == "batch" else "#1b9e77"
        ax.scatter(x, y, s=size, c=color, alpha=0.85, edgecolors="black", linewidths=0.5)
        ax.text(x + 1.2, y, labelize(row["condition"]), fontsize=9)
    ax.set_xlabel("n_active_frame")
    ax.set_ylabel("support_f1")
    ax.set_title("Sinusoid: sparsity pressure vs support recovery")

    # MNIST per-seed recon vs active atoms.
    ax = axes[0, 1]
    regime_colors = {"lowk": "#7570b3", "highk": "#e7298a"}
    for row in mnist_seed_rows:
        x = safe_float(row.get("n_active_frame"))
        y = safe_float(row.get("recon"))
        regime = row.get("regime")
        if x is None or y is None or regime is None:
            continue
        ax.scatter(x, y, c=regime_colors.get(regime, "#666666"), s=80, alpha=0.9, edgecolors="black", linewidths=0.4)
        ax.text(x + 1.0, y, f"{regime}-{row.get('seed')}", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("n_active_frame")
    ax.set_ylabel("recon (log scale)")
    ax.set_title("MNIST: site-normalized runs stay dense")

    # FSDD recon vs sparsity.
    ax = axes[1, 0]
    for row in fsdd_rows:
        x = safe_float(row.get("sparsity"))
        y = safe_float(row.get("recon"))
        if x is None or y is None:
            continue
        color = "#66a61e" if row.get("structure_mode") == "binary" else "#e6ab02"
        ax.scatter(x, y, c=color, s=170, alpha=0.9, edgecolors="black", linewidths=0.6)
        ax.text(x + 0.01, y, labelize(row["run_name"]), fontsize=9)
    ax.set_xlabel("delta zero fraction")
    ax.set_ylabel("recon")
    ax.set_title("FSDD: sparsity/interpretablity vs reconstruction")

    # LibriMix best SI-SDRi vs final active atoms.
    ax = axes[1, 1]
    for row in librimix_rows:
        x = safe_float(row.get("n_active_total_final"))
        y = safe_float(row.get("best_val_metric"))
        if x is None or y is None:
            continue
        objective = str(row.get("objective", ""))
        if "supervised" in objective:
            color = "#1f78b4"
        else:
            color = "#b2b2b2"
        ax.scatter(x, y, c=color, s=90, alpha=0.9, edgecolors="black", linewidths=0.4)
        ax.text(x + 5.0, y, labelize(row["run_name"]), fontsize=8)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
    ax.set_xlabel("final n_active_total")
    ax.set_ylabel("best SI-SDRi (dB)")
    ax.set_title("LibriMix: positive SI-SDRi is not enough if codes stay dense")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_librimix_progress(librimix_artifacts: list[dict[str, Any]], output_path: Path) -> None:
    selected = {
        "librimix_direct_mask": ("Direct mask", "#1f78b4"),
        "librimix_hybrid_partition": ("Hybrid old-KL", "#d95f02"),
        "librimix_hybrid_partition_randomcrop": ("Hybrid old-KL rcrop", "#7570b3"),
        "librimix_hybrid_partition_klsafe": ("Hybrid KL-safe", "#1b9e77"),
    }
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for artifact in librimix_artifacts:
        name = artifact.get("metadata", {}).get("run_name")
        if name not in selected:
            continue
        label, color = selected[name]
        val_epochs = artifact.get("new_log", {}).get("val_epochs", {})
        if not val_epochs:
            continue
        epochs = sorted(int(epoch) for epoch in val_epochs.keys())
        values = [safe_float(val_epochs[str(epoch)].get("si_sdri_db")) if isinstance(next(iter(val_epochs.keys())), str) else safe_float(val_epochs[epoch].get("si_sdri_db")) for epoch in epochs]
        pairs = [(epoch, value) for epoch, value in zip(epochs, values) if value is not None]
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=4.0, label=label, color=color)
        collapse_epoch = safe_float(artifact.get("row", {}).get("collapse_epoch"))
        if collapse_epoch is not None:
            ax.axvline(collapse_epoch, color=color, linestyle=":", linewidth=1.2, alpha=0.7)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("dev SI-SDRi (dB)")
    ax.set_title("LibriMix validation progress from run-summary JSON")
    ax.legend(frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot summary panels from experiment JSON artifacts")
    parser.add_argument(
        "--paper-summary-json",
        type=Path,
        default=REPO_ROOT / "report/tables/paper_batch_summaries/paper_batch_summary.json",
    )
    parser.add_argument(
        "--run-summaries-dir",
        type=Path,
        default=REPO_ROOT / "report/tables/run_summaries",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "report/figures/json_summary",
    )
    args = parser.parse_args()

    paper_summary = load_json(args.paper_summary_json)
    sinusoid_rows = paper_summary.get("sinusoid_regime_multiseed", [])
    mnist_seed_rows = paper_summary.get("mnist_regime_seeds", [])
    fsdd_rows = paper_summary.get("fsdd_case_study", [])

    librimix_artifacts = []
    for path in sorted(args.run_summaries_dir.glob("*.json")):
        artifact = load_json(path)
        if artifact.get("row", {}).get("dataset") == "librimix":
            librimix_artifacts.append(artifact)
    librimix_rows = [artifact.get("row", {}) for artifact in librimix_artifacts]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_sinusoid_spectrum(sinusoid_rows, args.output_dir / "sinusoid_metric_spectrum.png")
    plot_cross_experiment_tradeoffs(
        sinusoid_rows=sinusoid_rows,
        mnist_seed_rows=mnist_seed_rows,
        fsdd_rows=fsdd_rows,
        librimix_rows=librimix_rows,
        output_path=args.output_dir / "cross_experiment_tradeoffs.png",
    )
    plot_librimix_progress(librimix_artifacts, args.output_dir / "librimix_progress.png")
    print(f"Saved JSON-summary figures to {args.output_dir}")


if __name__ == "__main__":
    main()

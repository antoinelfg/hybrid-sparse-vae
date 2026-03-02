#!/usr/bin/env python
"""Generate a publication-quality multi-seed overlay plot.

Usage:
  python scripts/plot_multiseed_overlay.py
  python scripts/plot_multiseed_overlay.py --results-dir results/multiseed_champion
"""
import argparse, re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Curated colour palette ──────────────────────────────────────────
SEED_PALETTE = {
    42:   "#E67E22",   # warm orange
    123:  "#3498DB",   # sky blue
    456:  "#E91E63",   # pink
    789:  "#9B59B6",   # purple
    1337: "#2ECC71",   # emerald
}
MEAN_COLOR = "#1A1A2E"

PHASE_BOUNDS = [0, 400, 500, 1000]  # P1 end, P2 end, P3 end
PHASE_NAMES  = ["P1: Soft", "P2: Stoch.", "P3: β Ramp", "P4: Convergence"]
PHASE_COLORS = ["#BBDEFB", "#FFF9C4", "#E1BEE7", "#C8E6C9"]


def ema(x, alpha=0.05):
    """Exponential moving average."""
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def parse_log(log_path: Path):
    """Return arrays of (epoch, recon, sparsity, kl_gamma)."""
    pattern = re.compile(
        r"Epoch\s+(\d+).*?"
        r"recon\s+([\d.]+).*?"
        r"kl_γ\s+([\d.]+).*?"
        r"δ₀=([\d.]+)%"
    )
    epochs, recons, kl_gammas, sparsities = [], [], [], []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                recons.append(float(m.group(2)))
                kl_gammas.append(float(m.group(3)))
                sparsities.append(float(m.group(4)) / 100.0)
    return np.array(epochs), np.array(recons), np.array(sparsities), np.array(kl_gammas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/multiseed_champion")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--smooth", type=float, default=0.05)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = REPO_ROOT / results_dir

    seed_dirs = sorted(results_dir.glob("seed_*"))
    if not seed_dirs:
        print(f"No seed_* directories found in {results_dir}")
        sys.exit(1)

    # ── Setup figure ────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
    })

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.subplots_adjust(hspace=0.12, top=0.93, bottom=0.08, left=0.08, right=0.95)

    all_recons, all_sparsities, all_kl = [], [], []
    common_epochs = None

    for sd in seed_dirs:
        log_path = sd / "train.log"
        if not log_path.exists():
            continue

        seed_num = int(sd.name.split("_")[-1])
        color = SEED_PALETTE.get(seed_num, "#888888")
        epochs, recons, sparsities, kl_gammas = parse_log(log_path)
        if len(epochs) == 0:
            continue

        sm_r = ema(recons, args.smooth)
        sm_s = ema(sparsities, args.smooth) * 100
        sm_k = ema(kl_gammas, args.smooth)

        lbl = f"seed {seed_num} (final: {recons[-1]:.2f})"

        # Recon — full range
        axes[0].plot(epochs, sm_r, color=color, linewidth=1.6, alpha=0.85, label=lbl)

        # Sparsity — only from P2 (epoch >= 410) to avoid the 0% P1 artifacts
        mask = epochs >= 410
        axes[1].plot(epochs[mask], sm_s[mask], color=color, linewidth=1.6, alpha=0.85)

        # KL Gamma — full range
        axes[2].plot(epochs, sm_k, color=color, linewidth=1.6, alpha=0.85)

        all_recons.append(sm_r)
        all_sparsities.append(sm_s)
        all_kl.append(sm_k)
        if common_epochs is None:
            common_epochs = epochs

    # ── Mean ± Std band ─────────────────────────────────────────────
    if len(all_recons) > 1 and common_epochs is not None:
        min_len = min(len(r) for r in all_recons)
        ep = common_epochs[:min_len]

        # Recon band
        sr = np.stack([r[:min_len] for r in all_recons])
        mean_r, std_r = sr.mean(0), sr.std(0)
        axes[0].fill_between(ep, mean_r - std_r, mean_r + std_r,
                             color=MEAN_COLOR, alpha=0.07, linewidth=0)
        axes[0].plot(ep, mean_r, color=MEAN_COLOR, linewidth=2.5,
                     linestyle="--", alpha=0.8, label="Mean", zorder=10)

        # Sparsity band (P2+ only)
        ss = np.stack([s[:min_len] for s in all_sparsities])
        mean_s, std_s = ss.mean(0), ss.std(0)
        mask = ep >= 410
        axes[1].fill_between(ep[mask], (mean_s - std_s)[mask], (mean_s + std_s)[mask],
                             color=MEAN_COLOR, alpha=0.07, linewidth=0)
        axes[1].plot(ep[mask], mean_s[mask], color=MEAN_COLOR, linewidth=2.5,
                     linestyle="--", alpha=0.8, zorder=10)

        # KL band
        sk = np.stack([k[:min_len] for k in all_kl])
        mean_k, std_k = sk.mean(0), sk.std(0)
        axes[2].fill_between(ep, mean_k - std_k, mean_k + std_k,
                             color=MEAN_COLOR, alpha=0.07, linewidth=0)
        axes[2].plot(ep, mean_k, color=MEAN_COLOR, linewidth=2.5,
                     linestyle="--", alpha=0.8, zorder=10)

    # ── Phase shading ───────────────────────────────────────────────
    max_epoch = common_epochs[-1] if common_epochs is not None else 3000
    bounds = PHASE_BOUNDS + [max_epoch]
    for ax in axes:
        for i in range(len(PHASE_NAMES)):
            ax.axvspan(bounds[i], bounds[i + 1],
                       color=PHASE_COLORS[i], alpha=0.35, zorder=0, linewidth=0)
        ax.set_xlim(0, max_epoch)
        ax.grid(True, alpha=0.2, linewidth=0.4, color="#888")

    # Phase labels on top
    for i in range(len(PHASE_NAMES)):
        mid = (bounds[i] + bounds[i + 1]) / 2
        axes[0].text(mid, 9.3, PHASE_NAMES[i], ha="center", va="top",
                     fontsize=8.5, fontweight="bold", color="#444",
                     bbox=dict(boxstyle="round,pad=0.15", fc="white",
                               ec="none", alpha=0.8))

    # ── Labels and limits ───────────────────────────────────────────
    axes[0].set_ylabel("Reconstruction MSE")
    axes[0].set_ylim(0, 10)
    axes[0].legend(loc="upper right", framealpha=0.9, ncol=2, edgecolor="#ccc",
                   fontsize=9)
    axes[0].set_title("Hybrid Sparse VAE — 5-Seed Training Curves",
                      fontsize=14, fontweight="bold", pad=10)

    axes[1].set_ylabel("Sparsity δ₀ (%)")
    axes[1].set_ylim(40, 82)
    # Annotate the P1 region
    axes[1].text(200, 43, "P1: δ₀ = 0% (soft mode)", fontsize=8,
                 fontstyle="italic", color="#999")

    axes[2].set_ylabel("KL Gamma (γ)")
    axes[2].set_xlabel("Epoch")

    # ── Save ────────────────────────────────────────────────────────
    out_path = args.output or str(results_dir / "multiseed_overlay.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"✓ Saved → {out_path}")
    plt.close()


if __name__ == "__main__":
    main()

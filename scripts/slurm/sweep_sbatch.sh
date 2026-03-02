#!/bin/bash
#SBATCH --job-name=hvae_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=sweep_%j.out
#SBATCH --error=sweep_%j.err

# =============================================================================
#  Overnight Hyperparameter Sweep — Hybrid Sparse VAE (SLURM)
#  18 configs × 2000 epochs ≈ 3h on 1 GPU
# =============================================================================

cd /home/alaforgu/scratch/longitudinal_experiments/hybrid-sparse-vae
export PYTHONPATH=.

RESULTS_DIR="./results/sweep_$(date +%Y%m%d_%H%M)"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "  Hybrid Sparse VAE — Overnight Sweep"
echo "  Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "  Results → $RESULTS_DIR"
echo "  Started at: $(date)"
echo "============================================"

run_config() {
    local name="$1"
    shift
    local run_dir="$RESULTS_DIR/$name"
    mkdir -p "$run_dir"

    echo ""
    echo ">>> [$name] Starting at $(date +%H:%M:%S)"
    echo ">>> Config: $@"

    python train.py \
        hydra.run.dir="$run_dir" \
        "$@" \
        2>&1 | tee "$run_dir/train.log"

    echo ">>> [$name] Finished at $(date +%H:%M:%S)"
    echo "---"
}

# === SWEEP 1: β_γ exploration (key variable) ===
run_config "bg0.001_bd0.1_k0.5" \
    epochs=2000 beta_gamma_final=0.001 beta_delta_final=0.1 k_0=0.5

run_config "bg0.005_bd0.1_k0.5" \
    epochs=2000 beta_gamma_final=0.005 beta_delta_final=0.1 k_0=0.5

run_config "bg0.01_bd0.1_k0.5" \
    epochs=2000 beta_gamma_final=0.01 beta_delta_final=0.1 k_0=0.5

run_config "bg0.02_bd0.1_k0.5" \
    epochs=2000 beta_gamma_final=0.02 beta_delta_final=0.1 k_0=0.5

run_config "bg0.05_bd0.1_k0.5" \
    epochs=2000 beta_gamma_final=0.05 beta_delta_final=0.1 k_0=0.5

# === SWEEP 2: k₀ exploration (prior shape) ===
run_config "bg0.01_bd0.1_k0.3" \
    epochs=2000 beta_gamma_final=0.01 beta_delta_final=0.1 k_0=0.3

run_config "bg0.01_bd0.1_k1.0" \
    epochs=2000 beta_gamma_final=0.01 beta_delta_final=0.1 k_0=1.0

run_config "bg0.01_bd0.1_k2.0" \
    epochs=2000 beta_gamma_final=0.01 beta_delta_final=0.1 k_0=2.0

# === SWEEP 3: β_δ exploration (sparsity pressure) ===
run_config "bg0.01_bd0.05_k0.5" \
    epochs=2000 beta_gamma_final=0.01 beta_delta_final=0.05 k_0=0.5

run_config "bg0.01_bd0.2_k0.5" \
    epochs=2000 beta_gamma_final=0.01 beta_delta_final=0.2 k_0=0.5

run_config "bg0.01_bd0.5_k0.5" \
    epochs=2000 beta_gamma_final=0.01 beta_delta_final=0.5 k_0=0.5

# === SWEEP 4: Sweet spot combos ===
run_config "bg0.005_bd0.1_k0.3" \
    epochs=2000 beta_gamma_final=0.005 beta_delta_final=0.1 k_0=0.3

run_config "bg0.005_bd0.1_k1.0" \
    epochs=2000 beta_gamma_final=0.005 beta_delta_final=0.1 k_0=1.0

run_config "bg0.001_bd0.1_k1.0" \
    epochs=2000 beta_gamma_final=0.001 beta_delta_final=0.1 k_0=1.0

# === SWEEP 5: Delta prior exploration ===
run_config "bg0.01_bd0.1_k0.5_dp80" \
    epochs=2000 beta_gamma_final=0.01 beta_delta_final=0.1 k_0=0.5 \
    'delta_prior=0.10,0.80,0.10'

run_config "bg0.01_bd0.1_k0.5_dp60" \
    epochs=2000 beta_gamma_final=0.01 beta_delta_final=0.1 k_0=0.5 \
    'delta_prior=0.20,0.60,0.20'

# === SWEEP 6: Phase schedule exploration ===
run_config "bg0.01_bd0.1_k0.5_p1_200" \
    epochs=2000 beta_gamma_final=0.01 beta_delta_final=0.1 k_0=0.5 \
    phase1_end=200 phase2_end=400

run_config "bg0.01_bd0.1_k0.5_p1_50" \
    epochs=2000 beta_gamma_final=0.01 beta_delta_final=0.1 k_0=0.5 \
    phase1_end=50 phase2_end=100

echo ""
echo "============================================"
echo "  Sweep complete at: $(date)"
echo "  Results in: $RESULTS_DIR"
echo "============================================"

# Summary
echo ""
echo "=== FINAL EPOCH SUMMARY ==="
for dir in "$RESULTS_DIR"/*/; do
    name=$(basename "$dir")
    last_line=$(grep "Epoch" "$dir/train.log" 2>/dev/null | tail -1 || echo "NO DATA")
    echo "[$name] $last_line"
done

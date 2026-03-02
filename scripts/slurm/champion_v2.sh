#!/bin/bash
#SBATCH --job-name=hvae_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=champion_v2_%j.out
#SBATCH --error=champion_v2_%j.err

# =============================================================================
#  Champion V2 — with cosine LR decay in Phase 4
#  4-Phase: soft(400) → stoch(100) → ramp(500) → stationary+LR_decay(2000)
#  β_γ=0.005, β_δ=0.1, k₀=0.3, lr=3e-4→3e-5, 3000 epochs
# =============================================================================

cd /home/alaforgu/scratch/longitudinal_experiments/hybrid-sparse-vae
export PYTHONPATH=.

echo "============================================"
echo "  Champion V2 (LR decay) — $(date)"
echo "  Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================"

python train.py \
    hydra.run.dir="./results/champion_v2_$(date +%Y%m%d_%H%M)" \
    2>&1

echo "============================================"
echo "  Champion V2 complete — $(date)"
echo "============================================"

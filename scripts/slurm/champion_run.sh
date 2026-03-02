#!/bin/bash
#SBATCH --job-name=hvae_champ
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=champion_%j.out
#SBATCH --error=champion_%j.err

# =============================================================================
#  Champion Run — Hybrid Sparse VAE
#  4-Phase: soft(400) → stoch(100) → ramp(500) → stationary(2000)
#  β_γ=0.005, β_δ=0.1, k₀=0.3, 3000 epochs
# =============================================================================

cd /home/alaforgu/scratch/longitudinal_experiments/hybrid-sparse-vae
export PYTHONPATH=.

echo "============================================"
echo "  Champion Run — $(date)"
echo "  Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================"

python train.py \
    hydra.run.dir="./results/champion_$(date +%Y%m%d_%H%M)" \
    2>&1

echo "============================================"
echo "  Champion Run complete — $(date)"
echo "============================================"

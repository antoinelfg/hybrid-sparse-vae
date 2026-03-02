#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=baselines_%j.out
#SBATCH --error=baselines_%j.err

cd /home/alaforgu/scratch/longitudinal_experiments/hybrid-sparse-vae
export PYTHONPATH=.

echo "============================================"
echo "  Baselines Comparison — $(date)"
echo "  Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================"

python run_baselines.py 2>&1

echo "============================================"
echo "  Baselines complete — $(date)"
echo "============================================"

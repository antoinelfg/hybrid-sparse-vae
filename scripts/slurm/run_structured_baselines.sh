#!/bin/bash
#SBATCH --job-name=struct_bl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=struct_baselines_%j.out
#SBATCH --error=struct_baselines_%j.err

cd /home/alaforgu/scratch/longitudinal_experiments/hybrid-sparse-vae
export PYTHONPATH=.

echo "============================================"
echo "  Structured Baselines — $(date)"
echo "  Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================"

python run_structured_baselines.py 2>&1

echo "============================================"
echo "  Complete — $(date)"
echo "============================================"

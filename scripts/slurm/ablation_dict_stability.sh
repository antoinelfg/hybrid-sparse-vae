#!/bin/bash
#SBATCH --job-name=hvae_dict_stab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=results/ablation_dict_stability/slurm_%A_%a.out
#SBATCH --error=results/ablation_dict_stability/slurm_%A_%a.err
#SBATCH --array=0-29

cd /home/alaforgu/scratch/longitudinal_experiments/hybrid-sparse-vae
export PYTHONPATH=.

# There are 6 configs * 5 seeds = 30 tasks total.
# SLURM_ARRAY_TASK_ID will go from 0 to 29.

ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "Starting Dictionary Stability Ablation Task ID: ${ARRAY_TASK_ID}"

# Run just the specific configuration and seed for this task ID
python scripts/ablation_dict_stability.py --task-id ${ARRAY_TASK_ID}

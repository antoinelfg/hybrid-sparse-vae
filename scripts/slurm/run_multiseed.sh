#!/bin/bash
#SBATCH --job-name=hvae_multi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=results/multiseed_%A_%a.out
#SBATCH --error=results/multiseed_%A_%a.err

cd /home/alaforgu/scratch/longitudinal_experiments/hybrid-sparse-vae
export PYTHONPATH=.

# NOTE: This script provides an example for SLURM array jobs.
# You can run it with: sbatch --array=0-4 scripts/slurm/run_multiseed.sh

# If the array variable is empty, default to 0 (for testing)
ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Define the 5 seeds
SEEDS=(42 123 456 789 1337)

# Get the seed for this task
SEED=${SEEDS[$ARRAY_TASK_ID]}

echo "Running task ${ARRAY_TASK_ID} with seed ${SEED}"

# Run the training
# The log for this specific seed will be stored correctly, and the
# run_multiseed.py driver script can aggregate them later if they
# use a deterministic hydra.run.dir layout.
python train.py seed=$SEED \
  dict_init=random \
  dict_lr_mult=0.1 \
  hydra.run.dir=results/multiseed_champion/seed_$SEED \
  save_dir=results/multiseed_champion/seed_$SEED

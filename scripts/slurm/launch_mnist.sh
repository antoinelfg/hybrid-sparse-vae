#!/bin/bash
#SBATCH --job-name=mnist_hvae
#SBATCH --output=mnist_hvae_%A_%a.out
#SBATCH --error=mnist_hvae_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-5

# 3 seeds x 2 sizes = 6 jobs (0 to 5)
SEEDS=(42 123 456)
ATOMS=(128 256)

# Calculate indices
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 3))
ATOM_IDX=$((SLURM_ARRAY_TASK_ID / 3))

SEED=${SEEDS[$SEED_IDX]}
ATOM=${ATOMS[$ATOM_IDX]}

OUT_DIR="results/mnist_campaign/atoms_${ATOM}_seed_${SEED}"
mkdir -p "$OUT_DIR"

echo "========================================="
echo "Starting MNIST Campaign Job"
echo "Seed: $SEED"
echo "Atoms: $ATOM"
echo "Output Directory: $OUT_DIR"
echo "========================================="

# Run the training script
python train.py \
    dataset=mnist \
    input_length=784 \
    n_atoms=$ATOM \
    seed=$SEED \
    k_0=40.0 \
    k_min=10.0 \
    beta_gamma_final=0.001 \
    beta_delta_final=0.01 \
    hydra.run.dir=$OUT_DIR \
    save_dir=$OUT_DIR

echo "========================================="
echo "Job Completed."
echo "========================================="

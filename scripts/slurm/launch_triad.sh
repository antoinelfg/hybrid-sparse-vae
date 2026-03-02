#!/bin/bash
#SBATCH --job-name=mnist_triad
#SBATCH --output=mnist_triad_%A_%a.out
#SBATCH --error=mnist_triad_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-2

# The Triad: 3 parallel hypotheses for MNIST Posterior Collapse
# Array index 0: Gaussian Magnitude
# Array index 1: Dense Warmup (Gamma, beta_delta=0)
# Array index 2: Safety Net (Gamma, k_0=100.0)

SEED=42
ATOM=256

# Base Output Directory
BASE_DIR="results/mnist_triad/seed_${SEED}"

echo "========================================="
echo "Starting MNIST Triad Campaign Job"
echo "Seed: $SEED | Atoms: $ATOM | Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "========================================="

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    echo ">> RUN 1: Gaussian Magnitude <<"
    OUT_DIR="$BASE_DIR/run1_gaussian"
    mkdir -p "$OUT_DIR"
    python train.py \
        dataset=mnist \
        input_length=784 \
        n_atoms=$ATOM \
        seed=$SEED \
        magnitude_dist=gaussian \
        hydra.run.dir=$OUT_DIR \
        save_dir=$OUT_DIR

elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    echo ">> RUN 2: Dense Warmup (Gamma, beta_delta=0) <<"
    OUT_DIR="$BASE_DIR/run2_dense"
    mkdir -p "$OUT_DIR"
    python train.py \
        dataset=mnist \
        input_length=784 \
        n_atoms=$ATOM \
        seed=$SEED \
        magnitude_dist=gamma \
        k_0=40.0 \
        k_min=10.0 \
        beta_gamma_final=0.001 \
        beta_delta_final=0.0 \
        hydra.run.dir=$OUT_DIR \
        save_dir=$OUT_DIR

elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    echo ">> RUN 3: Safety Net (Gamma, k=100, clamp active) <<"
    OUT_DIR="$BASE_DIR/run3_safe"
    mkdir -p "$OUT_DIR"
    python train.py \
        dataset=mnist \
        input_length=784 \
        n_atoms=$ATOM \
        seed=$SEED \
        magnitude_dist=gamma \
        k_0=100.0 \
        k_min=10.0 \
        beta_gamma_final=0.001 \
        beta_delta_final=0.01 \
        hydra.run.dir=$OUT_DIR \
        save_dir=$OUT_DIR
fi

echo "========================================="
echo "Job Completed."
echo "========================================="

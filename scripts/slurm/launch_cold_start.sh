#!/bin/bash
#SBATCH --job-name=mnist_cold
#SBATCH --output=mnist_cold_%A_%a.out
#SBATCH --error=mnist_cold_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-3

# The "Cold Start" Campaign: No Warmup (Directly to Phase 4)
# Array index 0: Gaussian Cold Start
# Array index 1: Gamma Safe (k_min=10.0) -> ~30% noise
# Array index 2: Gamma Ambitious (k_min=1.0) -> 100% noise
# Array index 3: Gamma Extreme (k_min=0.1) -> 316% noise

SEED=42
ATOM=256
EPOCHS=1000

# Base Output Directory
BASE_DIR="results/mnist_cold/seed_${SEED}"

echo "========================================="
echo "Starting MNIST Cold Start Campaign Job"
echo "Seed: $SEED | Atoms: $ATOM | Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "========================================="

# Common args (No warmup)
COMMON_ARGS="dataset=mnist input_length=784 n_atoms=$ATOM seed=$SEED epochs=$EPOCHS phase1_end=0 phase2_end=0 phase3_end=0"

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    echo ">> RUN 0: Gaussian Cold Start <<"
    OUT_DIR="$BASE_DIR/run0_gaussian"
    mkdir -p "$OUT_DIR"
    python train.py $COMMON_ARGS \
        magnitude_dist=gaussian \
        hydra.run.dir=$OUT_DIR \
        save_dir=$OUT_DIR

elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    echo ">> RUN 1: Gamma Safe (k_min=10.0, k_0=50.0) <<"
    OUT_DIR="$BASE_DIR/run1_safe"
    mkdir -p "$OUT_DIR"
    python train.py $COMMON_ARGS \
        magnitude_dist=gamma \
        k_0=50.0 \
        k_min=10.0 \
        hydra.run.dir=$OUT_DIR \
        save_dir=$OUT_DIR

elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    echo ">> RUN 2: Gamma Ambitious (k_min=1.0, k_0=10.0) <<"
    OUT_DIR="$BASE_DIR/run2_ambitious"
    mkdir -p "$OUT_DIR"
    python train.py $COMMON_ARGS \
        magnitude_dist=gamma \
        k_0=10.0 \
        k_min=1.0 \
        hydra.run.dir=$OUT_DIR \
        save_dir=$OUT_DIR

elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    echo ">> RUN 3: Gamma Extreme (k_min=0.1, k_0=2.0) <<"
    OUT_DIR="$BASE_DIR/run3_extreme"
    mkdir -p "$OUT_DIR"
    python train.py $COMMON_ARGS \
        magnitude_dist=gamma \
        k_0=2.0 \
        k_min=0.1 \
        hydra.run.dir=$OUT_DIR \
        save_dir=$OUT_DIR
fi

echo "========================================="
echo "Job Completed."
echo "========================================="

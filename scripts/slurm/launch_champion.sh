#!/bin/bash
# =============================================================================
#  Champion Run: ConvLISTA + ConvNMF
#
#  Two physical fixes:
#    1. Overlap-Add: motif_width=64 (4x stride) for smooth sinusoidal motifs
#    2. k_max=1.5 clamp: forces Gamma distribution to stay in the sparse regime
#
#  Also includes a 3rd variant: ResNet encoder (no LISTA) with the same fixes
#  to isolate the contribution of unrolled inference.
#
#  Array IDs:
#    0 → ConvLISTA + ConvNMF (motif_width=64, k_max=1.5)   — CHAMPION
#    1 → ResNet   + ConvNMF  (motif_width=64, k_max=1.5)   — ablation: no LISTA
#    2 → ConvLISTA + ConvNMF (motif_width=64, k_max=inf)   — ablation: no k clamp
# =============================================================================
#SBATCH --job-name=champion_convnmf
#SBATCH --output=champion_%A_%a.out
#SBATCH --error=champion_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-2

source ~/.bashrc
conda activate base

export PYTHONPATH=.

SEED=42
EPOCHS=3000

echo "========================================="
echo "Champion Run — ConvLISTA + Overlap-Add"
echo "Array ID: $SLURM_ARRAY_TASK_ID | Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================="

# Shared Overlap-Add fix
COMMON="dataset=sinusoid encoder_output_dim=128 n_atoms=128 seed=$SEED epochs=$EPOCHS motif_width=64"

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    echo ">> CHAMPION: ConvLISTA + ConvNMF (motif_width=64, k_max=1.5) <<"
    OUT_DIR="results/champion/lista_convnmf_kmax1.5"
    mkdir -p "$OUT_DIR"
    python train.py $COMMON \
        encoder_type=lista \
        decoder_type=convnmf \
        k_max=1.5 \
        save_dir=$OUT_DIR \
        hydra.run.dir=$OUT_DIR \
        2>&1 | tee "$OUT_DIR/train.log"

elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    echo ">> ABLATION: ResNet + ConvNMF (motif_width=64, k_max=1.5, no LISTA) <<"
    OUT_DIR="results/champion/resnet_convnmf_kmax1.5"
    mkdir -p "$OUT_DIR"
    python train.py $COMMON \
        encoder_type=resnet \
        decoder_type=convnmf \
        k_max=1.5 \
        save_dir=$OUT_DIR \
        hydra.run.dir=$OUT_DIR \
        2>&1 | tee "$OUT_DIR/train.log"

elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    echo ">> ABLATION: ConvLISTA + ConvNMF (motif_width=64, k_max=inf, no k clamp) <<"
    OUT_DIR="results/champion/lista_convnmf_kmax_inf"
    mkdir -p "$OUT_DIR"
    python train.py $COMMON \
        encoder_type=lista \
        decoder_type=convnmf \
        save_dir=$OUT_DIR \
        hydra.run.dir=$OUT_DIR \
        2>&1 | tee "$OUT_DIR/train.log"
fi

echo "========================================="
echo "Champion Run Completed at $(date)"
echo "========================================="

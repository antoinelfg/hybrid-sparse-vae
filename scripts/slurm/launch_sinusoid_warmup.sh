#!/bin/bash
# =============================================================================
#  Sinusoid 1D — Dense NMF vs ConvNMF with default warmup schedule
#  Job Array:
#    0 → Dense NMF (linear encoder + linear decoder, warmup on)
#    1 → ConvNMF   (resnet encoder + convnmf decoder, warmup on)
#
#  Launched alongside parallel local runs (no warmup) for comparison.
# =============================================================================
#SBATCH --job-name=sinusoid_warmup
#SBATCH --output=sinusoid_warmup_%A_%a.out
#SBATCH --error=sinusoid_warmup_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-1

source ~/.bashrc
conda activate base

export PYTHONPATH=.

SEED=42
EPOCHS=3000

echo "========================================="
echo "Sinusoid Warmup Campaign"
echo "Array Job ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================="

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    echo ">> RUN 0: Dense NMF (linear encoder/decoder, warmup on) <<"
    OUT_DIR="results/sinusoid_warmup/dense_nmf"
    mkdir -p "$OUT_DIR"
    python train.py \
        dataset=sinusoid \
        encoder_type=linear \
        decoder_type=linear \
        seed=$SEED \
        epochs=$EPOCHS \
        save_dir=$OUT_DIR \
        hydra.run.dir=$OUT_DIR \
        2>&1 | tee "$OUT_DIR/train.log"

elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    echo ">> RUN 1: ConvNMF (resnet encoder + convnmf decoder, warmup on) <<"
    OUT_DIR="results/sinusoid_warmup/conv_nmf"
    mkdir -p "$OUT_DIR"
    python train.py \
        dataset=sinusoid \
        encoder_type=resnet \
        decoder_type=convnmf \
        seed=$SEED \
        epochs=$EPOCHS \
        save_dir=$OUT_DIR \
        hydra.run.dir=$OUT_DIR \
        2>&1 | tee "$OUT_DIR/train.log"
fi

echo "========================================="
echo "Job Completed at $(date)"
echo "========================================="

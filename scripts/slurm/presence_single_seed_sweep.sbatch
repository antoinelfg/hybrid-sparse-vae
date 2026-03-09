#!/bin/bash
#SBATCH --job-name=fpolar_presence_1seed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=presence_single_seed_%j.out
#SBATCH --error=presence_single_seed_%j.err

set -euo pipefail

REPO_ROOT="/home/alaforgu/scratch/longitudinal_experiments/hybrid-sparse-vae"
cd "$REPO_ROOT"
export PYTHONPATH=.

SEED="${SEED:-456}"
EPOCHS="${EPOCHS:-1000}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/regime_study_presence}"
CONDITIONS="${CONDITIONS:-HSVAE-fully-polar-lista-softprior-presweep-p15-bp020,HSVAE-fully-polar-lista-softprior-presweep-p20-bp020,HSVAE-fully-polar-lista-softprior-presweep-p25-bp020,HSVAE-fully-polar-lista-softprior-presweep-p30-bp020,HSVAE-fully-polar-lista-softprior-presweep-p15-bp030,HSVAE-fully-polar-lista-softprior-presweep-p20-bp030,HSVAE-fully-polar-lista-softprior-presweep-p25-bp030,HSVAE-fully-polar-lista-softprior-presweep-p30-bp030}"

echo "============================================"
echo " Fully Polar presence sweep (single seed)"
echo " Node: $(hostname)"
echo " Seed: $SEED"
echo " Epochs: $EPOCHS"
echo " Device: $DEVICE"
echo " Output root: $OUTPUT_ROOT"
echo " Conditions: $CONDITIONS"
echo " Started at: $(date)"
echo "============================================"

python scripts/run_regime_multiseed.py \
  --conditions "$CONDITIONS" \
  --seeds "$SEED" \
  --epochs "$EPOCHS" \
  --device "$DEVICE" \
  --output-root "$OUTPUT_ROOT" \
  --skip-existing

echo "============================================"
echo " Finished at: $(date)"
echo "============================================"

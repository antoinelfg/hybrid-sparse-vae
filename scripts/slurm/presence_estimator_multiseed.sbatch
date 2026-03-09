#!/bin/bash
#SBATCH --job-name=fpolar_presence_est
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=presence_estimator_%j.out
#SBATCH --error=presence_estimator_%j.err

set -euo pipefail

REPO_ROOT="/home/alaforgu/scratch/longitudinal_experiments/hybrid-sparse-vae"
cd "$REPO_ROOT"
export PYTHONPATH=.

SEEDS="${SEEDS:-42,123,456,789,1337}"
EPOCHS="${EPOCHS:-1000}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/regime_study_presence}"
CONDITIONS="${CONDITIONS:-HSVAE-fully-polar-lista-softprior-p20-bp020-estimator-gumbel,HSVAE-fully-polar-lista-softprior-p20-bp020-estimator-sparsemax,HSVAE-fully-polar-lista-softprior-p20-bp020-estimator-entmax15}"

echo "============================================"
echo " Fully Polar presence estimator sweep"
echo " Node: $(hostname)"
echo " Seeds: $SEEDS"
echo " Epochs: $EPOCHS"
echo " Device: $DEVICE"
echo " Output root: $OUTPUT_ROOT"
echo " Conditions: $CONDITIONS"
echo " Started at: $(date)"
echo "============================================"

python scripts/run_regime_multiseed.py \
  --conditions "$CONDITIONS" \
  --seeds "$SEEDS" \
  --epochs "$EPOCHS" \
  --device "$DEVICE" \
  --output-root "$OUTPUT_ROOT" \
  --skip-existing

echo "============================================"
echo " Finished at: $(date)"
echo "============================================"

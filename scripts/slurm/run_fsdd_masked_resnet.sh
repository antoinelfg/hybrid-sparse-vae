#!/bin/bash
#SBATCH --job-name=fsdd_masked_resnet
#SBATCH --output=fsdd_masked_resnet_%j.out
#SBATCH --error=fsdd_masked_resnet_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

source ~/.bashrc
conda activate base

# Changes vs fsdd_denoised_resnet:
#   masked_recon: True     (MSE computed only on non-zero signal pixels)
#   lambda_silence: 0.1    (light hallucination penalty on silence zones)
#   → The model is now FORCED to learn atom shapes that align with spectral peaks
#     rather than minimising error across the 91% silence sea.

PYTHONPATH=. python train.py \
    dataset="fsdd" \
    decoder_type="convnmf" \
    encoder_type="resnet" \
    motif_width=32 \
    n_atoms=128 \
    latent_dim=64 \
    epochs=2000 \
    phase1_end=400 \
    phase2_end=500 \
    phase3_end=1000 \
    k_max=0.8 \
    delta_prior="'0.01,0.98,0.01'" \
    denoise=True \
    masked_recon=True \
    lambda_silence=0.1 \
    save_dir="./checkpoints/fsdd_masked_resnet"

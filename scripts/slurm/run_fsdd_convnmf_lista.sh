#!/bin/bash
#SBATCH --job-name=fsdd_lista_conv
#SBATCH --output=fsdd_lista_conv_%j.out
#SBATCH --error=fsdd_lista_conv_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

source ~/.bashrc
conda activate base

PYTHONPATH=. python train.py \
    dataset="fsdd" \
    decoder_type="convnmf" \
    encoder_type="lista" \
    motif_width=16 \
    n_atoms=128 \
    latent_dim=64 \
    epochs=2000 \
    phase1_end=400 \
    phase2_end=500 \
    phase3_end=1000 \
    save_dir="./checkpoints/fsdd_convnmf_lista"

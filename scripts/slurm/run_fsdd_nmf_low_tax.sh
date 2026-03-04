#!/bin/bash
#SBATCH --job-name=fsdd_nmf_low
#SBATCH --output=fsdd_nmf_low_%j.out
#SBATCH --error=fsdd_nmf_low_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

source ~/.bashrc
conda activate base

python train.py \
    dataset="fsdd" \
    input_length=8256 \
    decoder_type="linear_positive" \
    encoder_type="linear" \
    n_atoms=128 \
    latent_dim=32 \
    dict_init="random_positive" \
    k_min=0.01 \
    magnitude_dist="gamma" \
    structure_mode="binary" \
    epochs=2000 \
    phase1_end=400 \
    phase2_end=800 \
    phase3_end=1200 \
    beta_delta_final=0.01 \
    beta_gamma_final=0.005 \
    save_dir="./checkpoints/fsdd_nmf_low_tax"

import torch
from models.hybrid_vae import HybridSparseVAE


def test_hybrid_vae_convnmf():
    model = HybridSparseVAE(
        input_channels=80, 
        input_length=128, 
        encoder_type="resnet", 
        decoder_type="convnmf", 
        n_atoms=64, 
        motif_width=16,
        decoder_stride=16,
    )
    x = torch.randn(4, 80, 128)
    x_recon, info = model(x)
    print(f"Input: {x.shape} -> Output: {x_recon.shape}")
    print(f"Latent params: B shape {info['B'].shape}, z shape {info['z'].shape}")
    assert x_recon.shape == x.shape


def test_convnmf_encoder_stride_matches_decoder_stride():
    model = HybridSparseVAE(
        input_channels=80,
        input_length=256,
        encoder_type="resnet",
        decoder_type="convnmf",
        n_atoms=64,
        motif_width=16,
        decoder_stride=4,
        match_encoder_decoder_stride=True,
    )
    x = torch.randn(2, 80, 256)
    _, info = model(x)

    # With the architecture-safe patch, the ResNet encoder must reduce time
    # by exactly decoder_stride for ConvNMF models.
    assert info["B"].shape == (2, 64, 64)
    assert getattr(model.encoder, "temporal_downsample_factor", None) == 4


def test_convnmf_legacy_encoder_stride_is_unchanged_by_default():
    model = HybridSparseVAE(
        input_channels=80,
        input_length=256,
        encoder_type="resnet",
        decoder_type="convnmf",
        n_atoms=64,
        motif_width=16,
        decoder_stride=4,
    )
    x = torch.randn(2, 80, 256)
    _, info = model(x)

    assert info["B"].shape == (2, 64, 16)
    assert getattr(model.encoder, "temporal_downsample_factor", None) == 16

if __name__ == "__main__":
    test_hybrid_vae_convnmf()

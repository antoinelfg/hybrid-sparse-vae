import torch
from models.hybrid_vae import HybridSparseVAE

def test_hybrid_vae_dense():
    model = HybridSparseVAE(
        input_channels=80, 
        input_length=128, 
        encoder_type="resnet", 
        decoder_type="linear_positive", 
        n_atoms=64, 
    )
    x = torch.randn(4, 80, 128)
    x_recon, info = model(x)
    print(f"Input: {x.shape} -> Output: {x_recon.shape}")
    print(f"Latent params: B shape {info['B'].shape}, z shape {info['z'].shape}")
    assert x_recon.shape == x.shape
    assert info['B'].shape == (4, 64)

if __name__ == "__main__":
    test_hybrid_vae_dense()

import torch
from models.hybrid_vae import HybridSparseVAE

model = HybridSparseVAE(
    input_channels=1, 
    input_length=128, 
    encoder_type="resnet", 
    decoder_type="convnmf", 
    n_atoms=64, 
)
x = torch.randn(4, 1, 128)
x_recon, info = model(x)
print("info['logits'] shape:", info['logits'].shape)

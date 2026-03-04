import torch
from modules.latent_space import StructuredLatentSpace

model = StructuredLatentSpace(input_dim=256, n_atoms=64, latent_dim=128, temporal_mode=True, structure_mode="ternary")
h = torch.randn(4, 256, 16)
z, info = model(h, temp=1.0, sampling="stochastic")
print("z shape:", z.shape)
print("delta shape:", info["delta"].shape)
print("k shape:", info["k"].shape)
print("Truncated Gumbel successful!")

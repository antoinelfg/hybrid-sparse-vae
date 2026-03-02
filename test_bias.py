import torch
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path.cwd()))
from scripts.visualize import build_model
import argparse

class DummyArgs:
    dataset = "mnist"
    input_length = 784
    encoder_type = "linear"
    decoder_type = "linear"
    encoder_output_dim = 256
    n_atoms = 256
    latent_dim = 64
    dict_init = "random"

args = DummyArgs()
device = "cpu"
model = build_model(args)

ckpt_path = Path("results/mnist_cold/seed_42/run1_safe/hybrid_vae_final.pt")
try:
    payload = torch.load(ckpt_path, map_location=device, weights_only=True)
except TypeError:
    payload = torch.load(ckpt_path, map_location=device)
state_dict = payload["state_dict"] if "state_dict" in payload else payload
if all(k.startswith("model.") for k in state_dict.keys()):
    state_dict = {k[len("model."):]: v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()

with torch.no_grad():
    z0 = torch.zeros(1, 64)
    x0 = model.decoder(z0).view(28, 28)
    
    # Let's check how the mean image looks
    import matplotlib.pyplot as plt
    plt.imshow(x0.numpy(), cmap="gray")
    plt.savefig("test_decoder_zero.png")
    print("Saved test_decoder_zero.png")

    # Plot delta
    z_one_hot = torch.eye(256) * 50.0  # Scale
    z_continuous = model.latent.dictionary(z_one_hot)
    x_atoms = model.decoder(z_continuous)
    
    delta = x_atoms - model.decoder(z0)
    delta_img = delta.view(256, 1, 28, 28)
    
    import torchvision.utils as vutils
    grid = vutils.make_grid(delta_img, nrow=16, normalize=True, scale_each=True)
    plt.figure(figsize=(20, 20))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="bwr")
    plt.savefig("test_atoms_delta.png")
    print("Saved test_atoms_delta.png")

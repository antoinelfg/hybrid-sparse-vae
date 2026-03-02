import torch
import sys
from pathlib import Path

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

from data.datasets import get_mnist_dataset
from torch.utils.data import DataLoader
dataset = get_mnist_dataset(data_dir="data", flatten=True)
loader = DataLoader(dataset, batch_size=2048, shuffle=True)
batch = next(iter(loader))
real_batch = batch[0]

with torch.no_grad():
    x_recon, info = model(real_batch, temp=0.05, sampling="deterministic")
    delta = info.get("delta")
    if delta is not None:
        prob = delta.abs().mean(dim=0)
        print("Max prob:", prob.max().item())
        print("Min prob:", prob.min().item())
        print("Mean prob:", prob.mean().item())

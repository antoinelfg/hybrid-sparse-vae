import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path.cwd()))
from scripts.visualize import build_model
from data.datasets import get_mnist_dataset

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

for run in ["run1_safe", "run0_gaussian"]:
    ckpt_path = Path(f"results/mnist_cold/seed_42/{run}/hybrid_vae_final.pt")
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = payload["state_dict"] if "state_dict" in payload else payload
    if all(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    dataset = get_mnist_dataset(data_dir="data", flatten=True)
    loader = DataLoader(dataset, batch_size=2048, shuffle=True)
    batch = next(iter(loader))[0]

    with torch.no_grad():
        _, info = model(batch, temp=0.05, sampling="deterministic")
        gamma = info["gamma"]
        delta = info["delta"]
        B = info["B"]
        
        active_mask = delta.abs() > 0.5
        # Average magnitude when active
        mag_when_active = gamma.abs() * active_mask
        sum_mag = mag_when_active.sum(dim=0)
        count_active = active_mask.sum(dim=0)
        
        avg_mag = torch.zeros_like(sum_mag)
        avg_mag[count_active > 0] = sum_mag[count_active > 0] / count_active[count_active > 0].float()
        
        print(f"=== {run} ===")
        print(f"B max: {B.abs().max().item():.4f}")
        print(f"Avg mag when active max: {avg_mag.max().item():.4f}")
        print(f"Avg mag when active mean: {avg_mag.mean().item():.4f}")
        print(f"Count active mean: {count_active.float().mean().item():.4f}")

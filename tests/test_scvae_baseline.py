from __future__ import annotations

from pathlib import Path

import torch

from scripts.baselines.run_scvae_sinusoid import LISTASCVAE
from scripts.eval_scvae_sparse_recovery import evaluate_scvae_sparse_recovery, resolve_sinusoid_stress_arg
from data.sinusoid_recovery import SinusoidRecoverySpec


def test_scvae_lista_forward_shapes():
    model = LISTASCVAE(input_dim=128, latent_dim=32, n_steps=3, prior_scale=0.25, threshold_init=0.1)
    x = torch.randn(4, 1, 128)
    recon, info = model(x)

    assert recon.shape == (4, 1, 128)
    assert info["mu"].shape == (4, 32)
    assert info["logvar"].shape == (4, 32)
    assert info["active"].shape == (4, 32)


def test_eval_scvae_sparse_recovery_smoke(tmp_path: Path):
    checkpoint = tmp_path / "scvae.pt"
    model = LISTASCVAE(input_dim=128, latent_dim=16, n_steps=2, prior_scale=0.25, threshold_init=0.1)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "length": 128,
                "latent_dim": 16,
                "n_steps": 2,
                "prior_scale": 0.25,
                "threshold_init": 0.1,
                "gain_distribution": "log_uniform",
                "gain_min": 0.1,
                "gain_max": 10.0,
                "normalize_divisor": 40.0,
            },
        },
        checkpoint,
    )

    result = evaluate_scvae_sparse_recovery(
        checkpoint=checkpoint,
        spec=SinusoidRecoverySpec(
            n_samples=8,
            length=128,
            n_components=3,
            seed=0,
            max_frequency=19,
            gain_distribution="log_uniform",
            gain_min=0.1,
            gain_max=10.0,
            normalize_divisor=40.0,
        ),
        min_atom_corr=0.7,
        min_subspace_score=0.7,
    )

    assert result["dataset"] == "sinusoid"
    assert result["baseline_name"] == "scvae_lista"
    assert result["dataset_spec"]["gain_distribution"] == "log_uniform"
    assert result["dataset_spec"]["normalize_divisor"] == 40.0
    assert result["reconstruction"]["recon_mse_per_example"] >= 0.0
    assert 0.0 <= result["support_eval"]["support_f1_mean"] <= 1.0


def test_resolve_sinusoid_stress_arg_prefers_cli_then_checkpoint():
    checkpoint_cfg = {
        "gain_distribution": "log_uniform",
        "gain_min": 0.1,
        "gain_max": 10.0,
        "normalize_divisor": 40.0,
    }

    assert resolve_sinusoid_stress_arg("uniform", checkpoint_cfg, "gain_distribution", "none") == "uniform"
    assert resolve_sinusoid_stress_arg(None, checkpoint_cfg, "gain_distribution", "none") == "log_uniform"
    assert resolve_sinusoid_stress_arg(None, checkpoint_cfg, "normalize_divisor", 4.0) == 40.0

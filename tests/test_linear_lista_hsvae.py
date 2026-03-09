from __future__ import annotations

import json
from pathlib import Path

import torch

from models.hybrid_vae import HybridSparseVAE
from modules.latent_space import binary_presence_projection
from scripts.eval_sparse_recovery import evaluate_sparse_recovery
from data.sinusoid_recovery import SinusoidRecoverySpec
from train import TrainConfig, get_phase


def test_hsvae_linear_lista_forward():
    model = HybridSparseVAE(
        input_channels=1,
        input_length=128,
        encoder_type="lista",
        encoder_output_dim=256,
        n_atoms=32,
        latent_dim=16,
        decoder_type="linear",
        dict_init="random",
        normalize_dict=True,
        k_min=0.01,
        k_max=0.8,
        magnitude_dist="gamma",
        structure_mode="ternary",
        lista_iterations=3,
        lista_threshold_init=0.1,
    )
    x = torch.randn(4, 1, 128)
    recon, info = model(x, sampling="deterministic")

    assert recon.shape == (4, 1, 128)
    assert info["delta"].shape == (4, 32, 1)
    assert info["k"].shape == (4, 32, 1)


def test_hsvae_linear_lista_forward_with_l2norm_delta_head():
    model = HybridSparseVAE(
        input_channels=1,
        input_length=128,
        encoder_type="lista",
        encoder_output_dim=256,
        n_atoms=32,
        latent_dim=128,
        decoder_type="linear",
        dict_init="random",
        normalize_dict=True,
        k_min=0.01,
        k_max=0.8,
        magnitude_dist="gamma",
        structure_mode="ternary",
        lista_iterations=3,
        lista_threshold_init=0.1,
        delta_head_mode="l2norm",
    )
    x = torch.randn(4, 1, 128)
    recon, info = model(x, sampling="deterministic")

    assert recon.shape == (4, 1, 128)
    assert info["delta"].shape == (4, 32, 1)
    assert "encoder.structure_proj.weight" in model.state_dict()


def test_hsvae_polar_lista_forward():
    model = HybridSparseVAE(
        input_channels=1,
        input_length=128,
        encoder_type="polar_lista",
        encoder_output_dim=256,
        n_atoms=32,
        latent_dim=128,
        decoder_type="linear",
        dict_init="random",
        normalize_dict=True,
        k_min=0.01,
        k_max=0.8,
        magnitude_dist="gamma",
        structure_mode="ternary",
        polar_encoder=True,
        delta_factorization="presence_sign",
        presence_estimator="gumbel_binary",
        lista_iterations=3,
        lista_threshold_init=0.1,
    )
    x = torch.randn(4, 1, 128)
    recon, info = model(x, sampling="deterministic")

    assert recon.shape == (4, 1, 128)
    assert info["delta"].shape == (4, 32)
    assert info["presence_probs"].shape == (4, 32)
    assert info["sign_values"].shape == (4, 32)
    assert "shape_state" in info


def test_hsvae_fully_polar_lista_forward():
    model = HybridSparseVAE(
        input_channels=1,
        input_length=128,
        encoder_type="fully_polar_lista",
        encoder_output_dim=256,
        n_atoms=32,
        latent_dim=128,
        decoder_type="linear",
        dict_init="random",
        normalize_dict=True,
        k_min=0.01,
        k_max=0.8,
        magnitude_dist="gamma",
        structure_mode="ternary",
        fully_polar_encoder=True,
        delta_factorization="presence_sign",
        presence_estimator="gumbel_binary",
        sign_estimator="gumbel_binary",
        lista_iterations=3,
        lista_threshold_init=0.1,
    )
    x = torch.randn(4, 1, 128)
    recon, info = model(x, sampling="deterministic")

    assert recon.shape == (4, 1, 128)
    assert info["delta"].shape == (4, 32)
    assert info["presence_probs"].shape == (4, 32)
    assert info["sign_probs"].shape == (4, 32)
    assert info["theta_tilde"].shape == (4, 32)
    assert info["theta_final"].shape == (4, 32)


def test_fully_polar_theta_is_equivariant_by_construction():
    model = HybridSparseVAE(
        input_channels=1,
        input_length=128,
        encoder_type="fully_polar_lista",
        encoder_output_dim=256,
        n_atoms=16,
        latent_dim=32,
        decoder_type="linear",
        dict_init="random",
        normalize_dict=True,
        k_min=0.01,
        k_max=0.8,
        magnitude_dist="gamma",
        structure_mode="ternary",
        fully_polar_encoder=True,
        delta_factorization="presence_sign",
        presence_estimator="gumbel_binary",
        sign_estimator="gumbel_binary",
        lista_iterations=3,
        lista_threshold_init=0.1,
    )
    x = torch.randn(2, 1, 128)
    _, info = model(x, sampling="deterministic")
    _, info_scaled = model(3.0 * x, sampling="deterministic")

    assert torch.allclose(
        info_scaled["theta_final"],
        3.0 * info["theta_final"],
        atol=1e-4,
        rtol=1e-4,
    )


def test_binary_presence_projection_is_per_atom_not_global():
    logits = torch.tensor(
        [[[[ -2.0,  4.0]], [[-2.0, 4.0]], [[-2.0, 4.0]], [[-2.0, 4.0]]]],
        dtype=torch.float32,
    )
    presence_probs, _ = binary_presence_projection(
        logits,
        estimator="sparsemax_binary",
        sampling="soft",
        temp=1.0,
        tau_presence_eval=0.5,
        presence_alpha=1.5,
    )
    assert presence_probs.shape == (1, 4, 1)
    assert float(presence_probs.sum().item()) > 1.5
    assert torch.allclose(presence_probs.squeeze(-1), torch.ones(1, 4))


def test_binary_presence_projection_entmax_has_exact_zeros():
    logits = torch.tensor(
        [[[[4.0, -4.0]], [[0.0, 0.0]], [[-4.0, 4.0]]]],
        dtype=torch.float32,
    )
    presence_probs, _ = binary_presence_projection(
        logits,
        estimator="entmax15_binary",
        sampling="soft",
        temp=1.0,
        tau_presence_eval=0.5,
        presence_alpha=1.5,
    )
    assert float((presence_probs == 0).float().mean().item()) > 0.0


def test_eval_sparse_recovery_handles_linear_lista_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / ".hydra").mkdir(parents=True)
    (run_dir / ".hydra/config.yaml").write_text(
        "\n".join(
            [
                "k_min: 0.01",
                "lista_iterations: 3",
                "sinusoid_gain_distribution: log_uniform",
                "sinusoid_gain_min: 0.1",
                "sinusoid_gain_max: 10.0",
                "sinusoid_normalize_divisor: 40.0",
            ]
        ),
        encoding="utf-8",
    )

    model = HybridSparseVAE(
        input_channels=1,
        input_length=128,
        encoder_type="lista",
        encoder_output_dim=256,
        n_atoms=32,
        latent_dim=16,
        decoder_type="linear",
        dict_init="random",
        normalize_dict=True,
        k_min=0.01,
        k_max=0.8,
        magnitude_dist="gamma",
        structure_mode="ternary",
        lista_iterations=3,
        lista_threshold_init=0.1,
    )
    checkpoint = run_dir / "hybrid_vae_final.pt"
    torch.save({"model_state": model.state_dict()}, checkpoint)

    result = evaluate_sparse_recovery(
        checkpoint=checkpoint,
        k_min=0.01,
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
        hydra_config=run_dir / ".hydra/config.yaml",
    )

    assert result["dataset"] == "sinusoid"
    assert result["dataset_spec"]["gain_distribution"] == "log_uniform"
    assert result["dataset_spec"]["gain_max"] == 10.0
    assert result["reconstruction"]["recon_mse_per_example"] >= 0.0
    assert 0.0 <= result["support_eval"]["support_f1_mean"] <= 1.0


def test_eval_sparse_recovery_handles_l2norm_delta_head_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "run_l2"
    (run_dir / ".hydra").mkdir(parents=True)
    (run_dir / ".hydra/config.yaml").write_text(
        "\n".join(
            [
                "k_min: 0.01",
                "lista_iterations: 3",
                "delta_head_mode: l2norm",
            ]
        ),
        encoding="utf-8",
    )

    model = HybridSparseVAE(
        input_channels=1,
        input_length=128,
        encoder_type="lista",
        encoder_output_dim=256,
        n_atoms=32,
        latent_dim=128,
        decoder_type="linear",
        dict_init="random",
        normalize_dict=True,
        k_min=0.01,
        k_max=0.8,
        magnitude_dist="gamma",
        structure_mode="ternary",
        lista_iterations=3,
        lista_threshold_init=0.1,
        delta_head_mode="l2norm",
    )
    checkpoint = run_dir / "hybrid_vae_final.pt"
    torch.save({"model_state": model.state_dict()}, checkpoint)

    result = evaluate_sparse_recovery(
        checkpoint=checkpoint,
        k_min=0.01,
        spec=SinusoidRecoverySpec(n_samples=8, length=128, n_components=3, seed=0, max_frequency=19),
        min_atom_corr=0.7,
        min_subspace_score=0.7,
        hydra_config=run_dir / ".hydra/config.yaml",
    )

    assert result["dataset"] == "sinusoid"
    assert result["reconstruction"]["recon_mse_per_example"] >= 0.0
    assert 0.0 <= result["support_eval"]["support_precision_mean"] <= 1.0


def test_eval_sparse_recovery_handles_polar_lista_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "run_polar"
    (run_dir / ".hydra").mkdir(parents=True)
    (run_dir / ".hydra/config.yaml").write_text(
        "\n".join(
            [
                "k_min: 0.01",
                "k_max: 0.8",
                "lista_iterations: 3",
                "encoder_type: polar_lista",
                "polar_encoder: true",
                "delta_factorization: presence_sign",
                "presence_estimator: sparsemax_binary",
                "presence_alpha: 1.5",
                "tau_presence_eval: 0.5",
                "gain_feature: log_l2",
            ]
        ),
        encoding="utf-8",
    )

    model = HybridSparseVAE(
        input_channels=1,
        input_length=128,
        encoder_type="polar_lista",
        encoder_output_dim=256,
        n_atoms=32,
        latent_dim=128,
        decoder_type="linear",
        dict_init="random",
        normalize_dict=True,
        k_min=0.01,
        k_max=0.8,
        magnitude_dist="gamma",
        structure_mode="ternary",
        polar_encoder=True,
        delta_factorization="presence_sign",
        presence_estimator="sparsemax_binary",
        presence_alpha=1.5,
        tau_presence_eval=0.5,
        lista_iterations=3,
        lista_threshold_init=0.1,
    )
    checkpoint = run_dir / "hybrid_vae_final.pt"
    torch.save({"model_state": model.state_dict()}, checkpoint)

    result = evaluate_sparse_recovery(
        checkpoint=checkpoint,
        k_min=0.01,
        spec=SinusoidRecoverySpec(n_samples=8, length=128, n_components=3, seed=0, max_frequency=19),
        min_atom_corr=0.7,
        min_subspace_score=0.7,
        hydra_config=run_dir / ".hydra/config.yaml",
    )

    assert result["dataset"] == "sinusoid"
    assert "invariance_eval" in result
    assert 0.0 <= result["invariance_eval"]["support_consistency_scaled_mean"] <= 1.0
    assert result["reconstruction"]["recon_mse_per_example"] >= 0.0


def test_eval_sparse_recovery_handles_fully_polar_lista_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "run_fully_polar"
    (run_dir / ".hydra").mkdir(parents=True)
    (run_dir / ".hydra/config.yaml").write_text(
        "\n".join(
            [
                "k_min: 0.01",
                "k_max: 0.8",
                "lista_iterations: 3",
                "encoder_type: fully_polar_lista",
                "fully_polar_encoder: true",
                "delta_factorization: presence_sign",
                "presence_estimator: gumbel_binary",
                "sign_estimator: gumbel_binary",
                "tau_presence_eval: 0.5",
            ]
        ),
        encoding="utf-8",
    )

    model = HybridSparseVAE(
        input_channels=1,
        input_length=128,
        encoder_type="fully_polar_lista",
        encoder_output_dim=256,
        n_atoms=32,
        latent_dim=128,
        decoder_type="linear",
        dict_init="random",
        normalize_dict=True,
        k_min=0.01,
        k_max=0.8,
        magnitude_dist="gamma",
        structure_mode="ternary",
        fully_polar_encoder=True,
        delta_factorization="presence_sign",
        presence_estimator="gumbel_binary",
        sign_estimator="gumbel_binary",
        lista_iterations=3,
        lista_threshold_init=0.1,
    )
    checkpoint = run_dir / "hybrid_vae_final.pt"
    torch.save({"model_state": model.state_dict()}, checkpoint)

    result = evaluate_sparse_recovery(
        checkpoint=checkpoint,
        k_min=0.01,
        spec=SinusoidRecoverySpec(n_samples=8, length=128, n_components=3, seed=0, max_frequency=19),
        min_atom_corr=0.7,
        min_subspace_score=0.7,
        hydra_config=run_dir / ".hydra/config.yaml",
    )

    assert result["dataset"] == "sinusoid"
    assert "invariance_eval" in result
    assert "theta_equivariance_error_mean" in result["invariance_eval"]
    assert "sign_entropy_mean" in result["invariance_eval"]


def test_get_phase_soft_finetune_override():
    cfg = TrainConfig(
        phase1_end=2,
        phase2_end=4,
        phase3_end=6,
        soft_finetune_start=8,
    )

    assert get_phase(1, cfg) == ("soft", 0.0)
    assert get_phase(3, cfg) == ("stochastic", 0.0)
    sampling, beta = get_phase(5, cfg)
    assert sampling == "stochastic"
    assert 0.0 < beta < 1.0
    assert get_phase(7, cfg) == ("stochastic", 1.0)
    assert get_phase(8, cfg) == ("soft", 1.0)

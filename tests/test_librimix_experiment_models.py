from argparse import Namespace
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.librimix_experiments import HybridLatentPartitionSeparator, SupervisedTFMaskSeparator
from scripts.train_librimix_experiments import compute_losses
from scripts.inference_bss import load_state_dict as load_inference_state_dict
from scripts.recap_diagnostics import load_state_dict as load_recap_state_dict


def test_supervised_tf_mask_separator_is_mixture_consistent():
    model = SupervisedTFMaskSeparator(
        n_freq_bins=65,
        hidden_channels=64,
        num_blocks=3,
        kernel_size=3,
    ).eval()
    mix_mag = torch.rand(2, 65, 48)

    out = model(mix_mag)

    assert out["source_mags"].shape == (2, 2, 65, 48)
    assert out["masks"].shape == (2, 2, 65, 48)
    assert torch.allclose(out["masks"].sum(dim=1), torch.ones_like(mix_mag), atol=1e-5)
    assert torch.allclose(out["source_mags"].sum(dim=1), mix_mag, atol=1e-5)


def test_hybrid_latent_partition_separator_is_additive():
    model = HybridLatentPartitionSeparator(
        input_channels=65,
        input_length=64,
        encoder_output_dim=64,
        n_atoms=16,
        latent_dim=8,
        motif_width=8,
        decoder_stride=4,
        encoder_type="resnet",
        structure_mode="binary",
        match_encoder_decoder_stride=True,
        assignment_hidden=32,
    ).eval()
    mix_mag = torch.rand(2, 65, 64)

    out = model(mix_mag, temp=0.05, sampling="deterministic")

    assert out["source_mags"].shape == (2, 2, 65, 64)
    assert out["source_assign"].shape == (2, 2, 16, 16)
    assert torch.allclose(out["source_assign"].sum(dim=1), torch.ones_like(out["source_assign"][:, 0]), atol=1e-5)
    assert torch.allclose(out["source_mags"].sum(dim=1), out["mixture_recon"], atol=1e-5)


def test_hybrid_compute_losses_normalizes_kl_per_site_when_requested():
    batch_size = 2
    n_atoms = 8
    latent_frames = 5
    freq_bins = 6
    spec_frames = 7

    source_mags = torch.rand(batch_size, 2, freq_bins, spec_frames)
    model_out = {
        "source_mags": source_mags,
        "mixture_recon": source_mags.sum(dim=1),
        "k": torch.full((batch_size, n_atoms, latent_frames), 0.6),
        "theta": torch.full((batch_size, n_atoms, latent_frames), 1.2),
        "logits": torch.zeros(batch_size, n_atoms, latent_frames, 2),
        "source_assign": torch.full((batch_size, 2, n_atoms, latent_frames), 0.5),
        "source_acts": torch.rand(batch_size, 2, n_atoms, latent_frames),
    }
    mix_mag = torch.rand(batch_size, freq_bins, spec_frames)
    s1_mag = torch.rand(batch_size, freq_bins, spec_frames)
    s2_mag = torch.rand(batch_size, freq_bins, spec_frames)

    common = dict(
        experiment="hybrid_partition",
        source_loss_type="l1",
        mix_loss_type="mse",
        lambda_source=1.0,
        lambda_mix=1.0,
        delta_prior="0.98,0.02",
        structure_mode="binary",
        k0=0.3,
        theta0=1.0,
        beta_gamma_final=0.005,
        beta_delta_final=0.05,
    )

    _, metrics_batch = compute_losses(
        model_out=model_out,
        mix_mag=mix_mag,
        source1_mag=s1_mag,
        source2_mag=s2_mag,
        args=Namespace(**common, kl_normalization="batch"),
        device=torch.device("cpu"),
        beta_frac=1.0,
    )
    _, metrics_site = compute_losses(
        model_out=model_out,
        mix_mag=mix_mag,
        source1_mag=s1_mag,
        source2_mag=s2_mag,
        args=Namespace(**common, kl_normalization="site"),
        device=torch.device("cpu"),
        beta_frac=1.0,
    )

    per_example_sites = n_atoms * latent_frames
    assert metrics_batch["kl_norm_factor"] == float(batch_size)
    assert metrics_site["kl_norm_factor"] == float(batch_size * per_example_sites)
    assert abs(metrics_batch["kl_gamma"] / per_example_sites - metrics_site["kl_gamma"]) < 1e-6
    assert abs(metrics_batch["kl_delta"] / per_example_sites - metrics_site["kl_delta"]) < 1e-6
    assert abs(metrics_batch["weighted_kl_gamma"] / per_example_sites - metrics_site["weighted_kl_gamma"]) < 1e-6
    assert abs(metrics_batch["weighted_kl_delta"] / per_example_sites - metrics_site["weighted_kl_delta"]) < 1e-6


def test_existing_librimix_loaders_accept_model_state_payload(tmp_path):
    ckpt_path = tmp_path / "wrapped.pt"
    expected = {"decoder.weight": torch.randn(4, 65, 8)}
    torch.save({"model_state": expected}, ckpt_path)

    loaded_inference = load_inference_state_dict(ckpt_path, device=torch.device("cpu"))
    loaded_recap = load_recap_state_dict(ckpt_path, device=torch.device("cpu"))

    assert loaded_inference.keys() == expected.keys()
    assert loaded_recap.keys() == expected.keys()
    assert torch.allclose(loaded_inference["decoder.weight"], expected["decoder.weight"])
    assert torch.allclose(loaded_recap["decoder.weight"], expected["decoder.weight"])

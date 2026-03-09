from __future__ import annotations

import torch

from data.sinusoid_recovery import (
    SinusoidRecoverySpec,
    generate_sinusoid_recovery_batch,
    seed_stability_jaccard,
    support_metrics_from_scores,
)
from train import generate_toy_sinusoid_tensors


def test_generate_sinusoid_recovery_batch_has_exact_support_metadata():
    spec = SinusoidRecoverySpec(n_samples=8, length=64, n_components=3, seed=7, max_frequency=19)
    batch = generate_sinusoid_recovery_batch(spec)

    assert batch["x"].shape == (8, 1, 64)
    assert batch["freqs"].shape == (8, 3)
    assert batch["gt_support"].shape == (8, 19)
    assert batch["gt_amp_scores"].shape == (8, 19)
    assert torch.all(batch["gt_support"].sum(dim=1) <= 3)


def test_support_metrics_are_perfect_when_predictions_match_ground_truth():
    gt_support = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    gt_amp_scores = torch.tensor([[0.9, 0.0, 0.4], [0.0, 0.8, 0.0]])
    pred_scores = gt_amp_scores.clone()

    metrics = support_metrics_from_scores(pred_scores, gt_support, gt_amp_scores)

    assert metrics["support_precision_mean"] == 1.0
    assert metrics["support_recall_mean"] == 1.0
    assert metrics["support_f1_mean"] == 1.0
    assert metrics["topk_f1_mean"] == 1.0


def test_seed_stability_jaccard_matches_pairwise_overlap():
    support_a = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    support_b = torch.tensor([[1, 0, 1], [0, 0, 1]], dtype=torch.bool)
    score = seed_stability_jaccard([support_a, support_b])
    # Shared positives: 2. Union positives: 4.
    assert abs(score - 0.5) < 1e-6


def test_generate_toy_sinusoid_tensors_defaults_are_stable_and_energy_stress_varies_amplitude():
    torch.manual_seed(0)
    x_default, freqs_default, amps_default, phases_default = generate_toy_sinusoid_tensors(
        n_samples=16,
        length=64,
        n_components=3,
        seed=0,
    )
    torch.manual_seed(0)
    x_explicit, freqs_explicit, amps_explicit, phases_explicit = generate_toy_sinusoid_tensors(
        n_samples=16,
        length=64,
        n_components=3,
        seed=0,
        gain_distribution="none",
        gain_min=1.0,
        gain_max=1.0,
        normalize_divisor=4.0,
    )

    assert torch.allclose(x_default, x_explicit)
    assert torch.equal(freqs_default, freqs_explicit)
    assert torch.allclose(amps_default, amps_explicit)
    assert torch.allclose(phases_default, phases_explicit)

    torch.manual_seed(1)
    x_stress, _, amps_stress, _ = generate_toy_sinusoid_tensors(
        n_samples=256,
        length=64,
        n_components=3,
        seed=1,
        gain_distribution="log_uniform",
        gain_min=0.1,
        gain_max=10.0,
        normalize_divisor=40.0,
    )

    sample_energy = x_stress.square().sum(dim=(1, 2)).sqrt()
    assert float(sample_energy.max().item() / sample_energy.min().item()) > 15.0
    assert float(amps_stress.max().item() / amps_stress.min().item()) > 15.0


def test_generate_sinusoid_recovery_batch_propagates_energy_stress_spec():
    spec = SinusoidRecoverySpec(
        n_samples=64,
        length=64,
        n_components=3,
        seed=9,
        max_frequency=19,
        gain_distribution="log_uniform",
        gain_min=0.1,
        gain_max=10.0,
        normalize_divisor=40.0,
    )
    batch = generate_sinusoid_recovery_batch(spec)

    assert batch["spec"]["gain_distribution"] == "log_uniform"
    assert batch["spec"]["gain_min"] == 0.1
    assert batch["spec"]["gain_max"] == 10.0
    assert batch["spec"]["normalize_divisor"] == 40.0
    assert float(batch["gt_amp_scores"].max().item() / batch["gt_amp_scores"][batch["gt_amp_scores"] > 0].min().item()) > 10.0

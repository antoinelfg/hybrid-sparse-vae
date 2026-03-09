from __future__ import annotations

from data.sinusoid_recovery import SinusoidRecoverySpec
from scripts.eval_omp_sparse_recovery import evaluate_omp_sparse_recovery


def test_evaluate_omp_sparse_recovery_smoke():
    spec = SinusoidRecoverySpec(
        n_samples=16,
        length=64,
        n_components=3,
        seed=0,
        max_frequency=19,
        gain_distribution="log_uniform",
        gain_min=0.1,
        gain_max=10.0,
        normalize_divisor=40.0,
    )

    result = evaluate_omp_sparse_recovery(
        spec,
        n_atoms=32,
        n_nonzero=8,
        min_atom_corr=0.7,
        min_subspace_score=0.7,
    )

    assert result["baseline_name"] == "omp_8"
    assert result["dataset_spec"]["gain_distribution"] == "log_uniform"
    assert result["reconstruction"]["recon_mse_per_example"] >= 0.0
    assert 0.0 <= result["support_eval"]["support_f1_mean"] <= 1.0

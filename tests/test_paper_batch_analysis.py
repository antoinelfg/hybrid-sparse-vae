from __future__ import annotations

import json
from pathlib import Path

from utils.regime_analysis import robust_positive_aggregate, summarize_sinusoid_condition_dir, summarize_training_run_dir


def test_summarize_sinusoid_condition_dir_handles_complete_and_running_seeds(tmp_path: Path):
    condition_dir = tmp_path / "HSVAE-lowk-full"
    complete_dir = condition_dir / "seed_42"
    running_dir = condition_dir / "seed_123"
    (complete_dir / ".hydra").mkdir(parents=True)
    (running_dir / ".hydra").mkdir(parents=True)

    (complete_dir / ".hydra/config.yaml").write_text(
        "\n".join(
            [
                "epochs: 3000",
                "kl_normalization: site",
                "structure_mode: ternary",
                "k_min: 0.01",
                "k_max: 0.8",
                "beta_gamma_final: 0.005",
                "beta_delta_final: 0.1",
            ]
        ),
        encoding="utf-8",
    )
    (complete_dir / "sparse_recovery.json").write_text(
        json.dumps(
            {
                "k_min": 0.01,
                "reconstruction": {"recon_mse_per_example": 10.0},
                "dataset_spec": {
                    "gain_distribution": "log_uniform",
                    "gain_min": 0.1,
                    "gain_max": 10.0,
                    "normalize_divisor": 40.0,
                },
                "latents": {
                    "collapsed": False,
                    "k_active_mean": 0.7,
                    "n_active_frame_mean": 8.0,
                    "n_active_total_mean": 12.0,
                    "sparsity": 0.75,
                },
                "atoms": {
                    "top1_fourier_corr_mean": 0.93,
                    "high_conf_atom_fraction": 0.8,
                },
                "support_eval": {
                    "support_precision_mean": 0.65,
                    "support_recall_mean": 0.58,
                    "support_f1_mean": 0.6,
                    "topk_f1_mean": 0.8,
                    "pred_gt_amplitude_corr_mean": 0.5,
                },
                "subspace_eval": {"support_f1_mean": 0.7},
            }
        ),
        encoding="utf-8",
    )

    (running_dir / ".hydra/config.yaml").write_text("epochs: 3000\nk_min: 0.01\n", encoding="utf-8")
    (running_dir / "train.log").write_text(
        "[2026-03-06 18:30:47,052][__main__][INFO] - Epoch  200 [P1:soft] | loss 29.5192 | recon 29.5192 | kl_γ 1.7875 | kl_δ 0.2583 | wkl_γ 0.0000 | wkl_δ 0.0000 | coh 0.0000 | k̄=10.362  k_act=10.362  n_act_frame=128.0  n_act_total=128.0/128  δ₀=0.00%  collapse=0  β=0.0000  τ=0.050  Δdict=0.0242\n",
        encoding="utf-8",
    )

    seed_rows, aggregate_row = summarize_sinusoid_condition_dir(condition_dir)

    assert len(seed_rows) == 2
    assert aggregate_row["status"] == "partial"
    assert aggregate_row["completed_seeds"] == 1
    assert abs(aggregate_row["support_f1_mean"] - 0.6) < 1e-9
    assert abs(aggregate_row["support_precision_mean"] - 0.65) < 1e-9
    assert abs(aggregate_row["support_recall_mean"] - 0.58) < 1e-9
    assert abs(aggregate_row["top1_fourier_corr_mean"] - 0.93) < 1e-9
    assert aggregate_row["gain_distribution"] == "log_uniform"
    assert abs(aggregate_row["recon_mse_mean"] - 10.0) < 1e-9


def test_summarize_training_run_dir_extracts_progress_and_collapse(tmp_path: Path):
    run_dir = tmp_path / "lowk_seed_42"
    (run_dir / ".hydra").mkdir(parents=True)
    (run_dir / ".hydra/config.yaml").write_text(
        "\n".join(
            [
                "seed: 42",
                "epochs: 3000",
                "structure_mode: ternary",
                "kl_normalization: site",
                "k_min: 0.1",
                "k_max: 0.8",
                "masked_recon: false",
                "denoise: false",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "train.log").write_text(
        "[2026-03-06 18:30:47,052][__main__][INFO] - Epoch  200 [P1:soft] | loss 29.5192 | recon 29.5192 | kl_γ 1.7875 | kl_δ 0.2583 | wkl_γ 0.0100 | wkl_δ 0.0200 | coh 0.0000 | k̄=0.795  k_act=0.900  n_act_frame=127.8  n_act_total=127.8/128  δ₀=0.13%  collapse=0  β=0.0000  τ=0.050  Δdict=0.0192\n",
        encoding="utf-8",
    )

    row = summarize_training_run_dir(run_dir, dataset="mnist", regime="lowk")

    assert row["status"] == "running"
    assert abs(row["progress"] - (200.0 / 3000.0)) < 1e-9
    assert abs(row["recon"] - 29.5192) < 1e-6
    assert row["collapse"] is False
    assert abs(row["weighted_kl_gamma"] - 0.01) < 1e-9


def test_summarize_sinusoid_condition_dir_prefers_posthoc_baseline_sparse_recovery(tmp_path: Path):
    condition_dir = tmp_path / "SimpleSparseBaseline"
    seed_dir = condition_dir / "seed_42"
    seed_dir.mkdir(parents=True)
    (seed_dir / "baseline_metrics.json").write_text(
        json.dumps({"results": {"sparse_ae": {"recon_mse": 1.0, "sparsity": 0.9}}}),
        encoding="utf-8",
    )
    (seed_dir / "baseline_sparse_recovery.json").write_text(
        json.dumps(
            {
                "selected_baseline": "omp_33",
                "baselines": {
                    "omp_33": {
                        "reconstruction": {"recon_mse_per_example": 0.5},
                        "latents": {
                            "collapsed": False,
                            "n_active_frame_mean": 10.0,
                            "n_active_total_mean": 10.0,
                            "sparsity": 0.8,
                        },
                        "support_eval": {
                            "support_f1_mean": 0.7,
                            "topk_f1_mean": 0.8,
                            "pred_gt_amplitude_corr_mean": 0.6,
                        },
                        "subspace_eval": {"support_f1_mean": 0.75},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    seed_rows, aggregate_row = summarize_sinusoid_condition_dir(condition_dir)

    assert seed_rows[0]["model_family"] == "baseline"
    assert seed_rows[0]["baseline_name"] == "omp_33"
    assert abs(seed_rows[0]["support_f1_mean"] - 0.7) < 1e-9
    assert abs(aggregate_row["support_f1_mean"] - 0.7) < 1e-9


def test_summarize_sinusoid_condition_dir_marks_sparse_recovery_baseline_family(tmp_path: Path):
    condition_dir = tmp_path / "SCVAE-lista-energy"
    seed_dir = condition_dir / "seed_42"
    seed_dir.mkdir(parents=True)
    (seed_dir / "sparse_recovery.json").write_text(
        json.dumps(
            {
                "baseline_name": "scvae_lista",
                "dataset_spec": {
                    "gain_distribution": "log_uniform",
                    "gain_min": 0.1,
                    "gain_max": 10.0,
                    "normalize_divisor": 40.0,
                },
                "reconstruction": {"recon_mse_per_example": 0.25},
                "latents": {
                    "collapsed": False,
                    "n_active_frame_mean": 5.0,
                    "n_active_total_mean": 5.0,
                    "sparsity": 0.95,
                },
                "atoms": {
                    "top1_fourier_corr_mean": 0.75,
                    "high_conf_atom_fraction": 0.6,
                },
                "support_eval": {
                    "support_precision_mean": 0.99,
                    "support_recall_mean": 0.95,
                    "support_f1_mean": 0.97,
                    "topk_f1_mean": 0.98,
                    "pred_gt_amplitude_corr_mean": 0.8,
                },
                "subspace_eval": {"support_f1_mean": 0.96},
            }
        ),
        encoding="utf-8",
    )

    seed_rows, aggregate_row = summarize_sinusoid_condition_dir(condition_dir)

    assert seed_rows[0]["model_family"] == "baseline"
    assert seed_rows[0]["baseline_name"] == "scvae_lista"
    assert aggregate_row["model_family"] == "baseline"
    assert aggregate_row["gain_max"] == 10.0


def test_robust_positive_aggregate_flags_large_outlier():
    stats = robust_positive_aggregate([101.0, 102.0, 618542.0])

    assert abs(stats["median"] - 102.0) < 1e-9
    assert abs(stats["stable_mean"] - 101.5) < 1e-9
    assert stats["outlier_count"] == 1

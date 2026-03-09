from __future__ import annotations

from pathlib import Path

from utils.regime_analysis import parse_librimix_experiment_log, parse_old_train_log


def test_parse_librimix_experiment_log_extracts_best_and_collapse(tmp_path: Path):
    log_path = tmp_path / "hybrid.err"
    log_path.write_text(
        "\n".join(
            [
                "2026-03-06 15:31:57 | INFO | Epoch   10 | loss 0.3501 | source 0.2008 | mix 0.1493 | kl_g 1.4646 | kl_d 0.1263 | wkl_g 0.0000 | wkl_d 0.0000 | n_act_frame 512.0 | n_act_total 512.0/512 | H(assign) 0.299 | temp 0.921 | 39.2s",
                "2026-03-06 15:31:57 | INFO | Validation epoch   10 | SI-SDRi 1.860 dB | SI-SDR 1.853 dB | source 0.1965 | mix 0.1195 | eval_n=8",
                "2026-03-06 15:31:57 | INFO | Validation epoch   10 | oracle SI-SDRi 13.977 dB | oracle gap 12.117 dB",
                "2026-03-06 15:57:58 | INFO | Epoch   50 | loss 0.2285 | source 0.1579 | mix 0.0706 | kl_g 1.4619 | kl_d 0.3020 | wkl_g 0.0000 | wkl_d 0.0000 | n_act_frame 512.0 | n_act_total 512.0/512 | H(assign) 0.275 | temp 0.604 | 38.6s",
                "2026-03-06 15:57:58 | INFO | Validation epoch   50 | SI-SDRi 3.414 dB | SI-SDR 3.406 dB | source 0.1749 | mix 0.0580 | eval_n=8",
                "2026-03-06 16:07:52 | INFO | Epoch   65 | loss 0.3296 | source 0.1801 | mix 0.1494 | kl_g 1.0492 | kl_d 0.7406 | wkl_g 0.0000 | wkl_d 0.0000 | n_act_frame 0.0 | n_act_total 0.0/512 | H(assign) 0.320 | temp 0.485 | 40.0s",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_librimix_experiment_log(log_path)

    assert parsed["best_epoch"] == 50
    assert parsed["best_val"]["si_sdri_db"] == 3.414
    assert parsed["val_epochs"][10]["oracle_gap_db"] == 12.117
    assert parsed["collapse_epoch"] == 65


def test_parse_old_train_log_extracts_final_metrics(tmp_path: Path):
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "[2026-03-06 04:17:03,149][__main__][INFO] - Epoch  200 [P4:conv] | loss 7070.6892 | recon 6350.7650 | kl_γ 16771.9960 | kl_δ 12721.2852 | k̄=0.507  k_act=0.792  n_act_frame=47.8  n_act_total=480.1/512  δ₀=90.66%  β=0.0050  τ=0.050  Δdict=0.9233\n",
        encoding="utf-8",
    )

    parsed = parse_old_train_log(log_path)

    assert parsed["last_epoch"] == 200
    assert abs(parsed["last_metrics"]["n_active_total"] - 480.1) < 1e-6
    assert abs(parsed["last_metrics"]["k_mean"] - 0.507) < 1e-6

#!/usr/bin/env python
"""Run systematic ablation studies across components.

Ablations:
  1. Gaussian magnitude instead of Gamma.
  2. Binary structure instead of Ternary.
  3. No dictionary projection (Identity init).
  4. No phase schedule (rocket launch off).
  5. No dictionary column normalization.

Each is run over multiple seeds to compute robust `mean ± std` metrics.

Usage:
  python scripts/run_ablation.py
"""

import sys
import json
import subprocess
from pathlib import Path
import numpy as np

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_multiseed import extract_metrics

SEEDS = [42, 123, 456]

ABLATIONS = {
    "baseline": [
        "dict_init=random",
        "dict_lr_mult=0.1"
    ],
    "gaussian_magnitude": [
        "dict_init=random",
        "dict_lr_mult=0.1",
        "magnitude_dist=gaussian"
    ],
    "binary_structure": [
        "dict_init=random",
        "dict_lr_mult=0.1",
        "structure_mode=binary",
        "delta_prior=0.85,0.15"  # Prior format must match n_logits=2
    ],
    "no_dictionary": [
        "dict_init=identity",
        "n_atoms=64"  # Match latent_dim
    ],
    "no_phase_schedule": [
        "dict_init=random",
        "dict_lr_mult=0.1",
        "phase1_end=0",
        "phase2_end=0",
        "phase3_end=0"
    ],
    "no_dict_norm": [
        "dict_init=random",
        "dict_lr_mult=0.1",
        "normalize_dict=false"
    ]
}


import argparse

def main():
    parser = argparse.ArgumentParser(description="Main Ablation Study")
    parser.add_argument("--task-id", type=int, default=None,
                        help="SLURM array task ID to run a single configuration/seed.")
    args = parser.parse_args()

    out_dir = REPO_ROOT / "results" / "ablation_table"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate the flat list of tasks
    tasks = []
    for abl_name, hydra_args in ABLATIONS.items():
        for seed in SEEDS:
            tasks.append((abl_name, hydra_args, seed))
            
    if args.task_id is not None:
        if args.task_id < 0 or args.task_id >= len(tasks):
            print(f"ERROR: Task ID {args.task_id} out of bounds (0-{len(tasks)-1})")
            sys.exit(1)
            
        abl_name, hydra_args, seed = tasks[args.task_id]
        print(f"--- SLURM TASK {args.task_id} ---")
        print(f"Ablation: {abl_name}")
        print(f"Seed: {seed}")
        
        run_dir = out_dir / abl_name / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [sys.executable, str(REPO_ROOT / "train.py"), f"seed={seed}"] + hydra_args
        cmd.append(f"hydra.run.dir={run_dir}")
        cmd.append(f"save_dir={run_dir}")
        
        import subprocess
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        sys.exit(result.returncode)
    
    print(f"Starting ablation study over {len(ABLATIONS)} configurations.")
    print(f"Seeds per config: {SEEDS}")
    print("=" * 80)
    
    summary = {}
    
    for abl_name, hydra_args in ABLATIONS.items():
        print(f"\n--- Running Ablation: {abl_name} ---")
        
        config_results = []
        for seed in SEEDS:
            run_dir = out_dir / abl_name / f"seed_{seed}"
            
            log_path = run_dir / "train.log"
            if not log_path.exists():
                run_dir.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, str(REPO_ROOT / "train.py"), f"seed={seed}"] + hydra_args
                cmd.append(f"hydra.run.dir={run_dir}")
                cmd.append(f"save_dir={run_dir}")
                
                print(f"  [{seed}] Starting...", end=" ", flush=True)
                import subprocess
                result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True)
                
                if result.returncode != 0:
                    print(f"FAILED!")
                    err_str = result.stderr.decode("utf-8")
                    print("\n".join(err_str.split("\n")[-10:]))
                    continue
                
            metrics = extract_metrics(log_path)
            
            if metrics:
                recon = metrics["recon"]
                sparsity = metrics["sparsity"]
                print(f"Recon MSE: {recon:.4f} | Sparsity: {sparsity:.2%}")
                config_results.append({
                    "seed": seed,
                    "metrics": metrics
                })
            else:
                print(f"Could not extract metrics.")
                
        if not config_results:
            print(f"No successful runs for {abl_name}")
            continue
            
        # Aggregate
        def agg(key):
            vals = [r["metrics"][key] for r in config_results]
            return float(np.mean(vals)), float(np.std(vals))
            
        r_mean, r_std = agg("recon")
        s_mean, s_std = agg("sparsity")
        k_mean, k_std = agg("k_mean")
        
        summary[abl_name] = {
            "recon_mean": r_mean, "recon_std": r_std,
            "sparsity_mean": s_mean, "sparsity_std": s_std,
            "k_mean_mean": k_mean, "k_mean_std": k_std,
            "runs": len(config_results)
        }
        
    print("\n" + "=" * 90)
    print(" ABLATION TABLE RESULTS ")
    print("=" * 90)
    print(f"{'Ablation':<20} | {'Recon MSE':<20} | {'Sparsity':<20} | {'k_mean'}")
    print("-" * 90)
    
    for name, s in summary.items():
        r_str = f"{s['recon_mean']:.4f} ± {s['recon_std']:.4f}"
        sp_str = f"{s['sparsity_mean']:.2%} ± {s['sparsity_std']:.2%}"
        k_str = f"{s['k_mean_mean']:.3f} ± {s['k_mean_std']:.3f}"
        print(f"{name:<20} | {r_str:<20} | {sp_str:<20} | {k_str:<20}")
        
    print("=" * 90)
    
    json_path = out_dir / "ablation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved JSON summary to {json_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train matched models with different diffusion-time sampling distributions and
evaluate each checkpoint at multiple Euler NFEs.

This is designed to test the low-t score-learning hypothesis.

The models are identical except for:
    uniform
    log_uniform
    mixed

By default alpha=3, beta=3 and mixed uses 50% log-uniform samples.

Example
-------
python scripts/run_time_sampling_sweep.py \
  --strategies uniform,log_uniform,mixed \
  --alpha 3 --beta 3 \
  --max-epochs 100 \
  --steps 5,10,20,50,100 \
  --seed 2

For a more serious run:
python scripts/run_time_sampling_sweep.py \
  --strategies uniform,log_uniform,mixed \
  --alpha 3 --beta 3 \
  --max-epochs 1000 \
  --steps 5,10,20,50,100 \
  --seed 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_csv(text, cast=str):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def run(cmd, cwd: Path, log_path: Path):
    print("\n$ " + " ".join(cmd), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)
            f.flush()
        code = p.wait()

    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(rows, path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default="uniform,log_uniform,mixed")
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--mixed-log-fraction", type=float, default=0.5)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--steps", default="5,10,20,50,100")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--experiment", default="paper_k2_npz")
    ap.add_argument("--checkpoint-policy", choices=["last"], default="last")
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("sweeps/time_sampling"),
    )
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--force-eval", action="store_true")
    args = ap.parse_args()

    project = Path.cwd().resolve()
    if not (project / "train.py").exists():
        raise FileNotFoundError("Run from the project root containing train.py.")

    strategies = parse_csv(args.strategies, str)
    allowed = {"uniform", "log_uniform", "mixed"}
    bad = [s for s in strategies if s not in allowed]
    if bad:
        raise ValueError(f"Unsupported strategies: {bad}")

    steps = parse_csv(args.steps, int)
    root = args.root if args.root.is_absolute() else (project / args.root)
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for strategy in strategies:
        tag = (
            f"{strategy}_a{args.alpha:g}_b{args.beta:g}_"
            f"mix{args.mixed_log_fraction:g}_seed{args.seed}"
        ).replace(".", "p")

        run_dir = root / tag
        ckpt = run_dir / "checkpoints/last.ckpt"

        if args.force_train or not ckpt.exists():
            cmd = [
                sys.executable,
                "train.py",
                f"experiment={args.experiment}",
                f"seed={args.seed}",
                f"name={tag}",
                f"trainer.max_epochs={args.max_epochs}",
                "save_top_k=1",
                "train=true",
                "test=false",
                f"model.sde.drift_decay_rate={args.alpha}",
                f"model.sde.zero_mode_decay_rate={args.beta}",
                f"model.time_sampling_strategy={strategy}",
                (
                    "model.time_sampling_log_fraction="
                    f"{args.mixed_log_fraction}"
                ),
                f"hydra.run.dir={run_dir}",
                "hydra.job.chdir=true",
            ]
            run(cmd, project, run_dir / "train.log")
        else:
            print(f"\nReusing checkpoint: {ckpt}")

        if not ckpt.exists():
            raise FileNotFoundError(f"Training did not create {ckpt}")

        for nfe in steps:
            eval_dir = run_dir / "evaluation_nfe" / f"euler_{nfe}"
            summary = eval_dir / "summary.json"

            if args.force_eval or not summary.exists():
                cmd = [
                    sys.executable,
                    "evaluate.py",
                    f"experiment={args.experiment}",
                    f"seed={args.seed}",
                    f"checkpoint={ckpt.resolve()}",
                    f"model.sde.drift_decay_rate={args.alpha}",
                    f"model.sde.zero_mode_decay_rate={args.beta}",
                    f"model.time_sampling_strategy={strategy}",
                    (
                        "model.time_sampling_log_fraction="
                        f"{args.mixed_log_fraction}"
                    ),
                    "model.sampler.method=euler",
                    f"model.sampler.num_steps={nfe}",
                    f"evaluation_out={eval_dir}",
                    f"hydra.run.dir={run_dir / 'hydra_eval' / f'euler_{nfe}'}",
                    "hydra.job.chdir=false",
                ]
                run(cmd, project, eval_dir / "evaluate.log")
            else:
                print(f"Reusing evaluation: {summary}")

            s = read_json(summary)
            row = {
                "strategy": strategy,
                "alpha": args.alpha,
                "beta": args.beta,
                "mixed_log_fraction": args.mixed_log_fraction,
                "seed": args.seed,
                "epochs": args.max_epochs,
                "requested_steps": nfe,
            }
            for key in [
                "latent_si_sdr_db_mean",
                "source_image_si_sdr_db_mean",
                "nmse_mean",
                "nmae_mean",
                "failure_rate_lt_5db",
                "measurement_residual_mean",
                "mean_nfe",
            ]:
                if key in s:
                    row[key] = s[key]

            all_rows.append(row)
            write_csv(all_rows, root / "results.csv")

    print("\n" + "=" * 105)
    print("TIME-SAMPLING SWEEP")
    print("=" * 105)
    print(
        f"{'strategy':>12} {'steps':>7} {'latent SI-SDR':>16} "
        f"{'image SI-SDR':>16} {'failure':>11} {'residual':>12}"
    )
    print("-" * 105)

    for r in all_rows:
        print(
            f"{r['strategy']:>12} "
            f"{int(r['requested_steps']):7d} "
            f"{float(r.get('latent_si_sdr_db_mean', np.nan)):16.4f} "
            f"{float(r.get('source_image_si_sdr_db_mean', np.nan)):16.4f} "
            f"{float(r.get('failure_rate_lt_5db', np.nan)):11.4f} "
            f"{float(r.get('measurement_residual_mean', np.nan)):12.6f}"
        )

    print("=" * 105)
    print(f"Saved: {root / 'results.csv'}")


if __name__ == "__main__":
    main()

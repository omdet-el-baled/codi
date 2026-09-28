#!/usr/bin/env python3
"""Train/evaluate matched physical-score vs whitened-score models.

This script is intentionally simple and resumable.  It trains one checkpoint
per parameterization, then evaluates the same checkpoint at several Euler NFEs.

Example
-------
python scripts/run_score_parameterization_sweep.py \
  --parameterizations physical,whitened \
  --alpha 3 --beta 3 \
  --max-epochs 100 \
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


def parse_csv(text: str, cast=str):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def run_and_tee(cmd: list[str], cwd: Path, log_path: Path) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            logf.write(line)
            logf.flush()
        rc = proc.wait()
    if rc:
        raise subprocess.CalledProcessError(rc, cmd)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parameterizations", default="physical,whitened")
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--steps", default="5,10,20,50,100")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--experiment", default="paper_k2_npz")
    ap.add_argument("--root", type=Path, default=Path("sweeps/score_parameterization"))
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--force-eval", action="store_true")
    args = ap.parse_args()

    project = Path.cwd().resolve()
    if not (project / "train.py").exists() or not (project / "evaluate.py").exists():
        raise FileNotFoundError("Run this script from the project root.")

    params = parse_csv(args.parameterizations)
    allowed = {"physical", "whitened"}
    invalid = [p for p in params if p not in allowed]
    if invalid:
        raise ValueError(f"Unsupported parameterizations: {invalid}")

    nfes = parse_csv(args.steps, int)
    if any(n <= 0 for n in nfes):
        raise ValueError("All step counts must be positive.")

    root = args.root if args.root.is_absolute() else project / args.root
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for parameterization in params:
        tag = (
            f"{parameterization}_a{args.alpha:g}_b{args.beta:g}_seed{args.seed}"
            .replace(".", "p")
        )
        run_dir = root / tag
        ckpt = run_dir / "checkpoints" / "last.ckpt"

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
                f"model.score_parameterization={parameterization}",
                "model.time_sampling_strategy=uniform",
                f"hydra.run.dir={run_dir}",
                "hydra.job.chdir=true",
            ]
            run_and_tee(cmd, project, run_dir / "train.log")
        else:
            print(f"Reusing checkpoint: {ckpt}")

        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint missing after training: {ckpt}")

        for nfe in nfes:
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
                    f"model.score_parameterization={parameterization}",
                    "model.time_sampling_strategy=uniform",
                    "model.sampler.method=euler",
                    f"model.sampler.num_steps={nfe}",
                    f"evaluation_out={eval_dir}",
                    f"hydra.run.dir={run_dir / 'hydra_eval' / f'euler_{nfe}'}",
                    "hydra.job.chdir=false",
                ]
                run_and_tee(cmd, project, eval_dir / "evaluate.log")
            else:
                print(f"Reusing evaluation: {summary}")

            metrics = load_json(summary)
            row = {
                "parameterization": parameterization,
                "alpha": args.alpha,
                "beta": args.beta,
                "seed": args.seed,
                "epochs": args.max_epochs,
                "requested_steps": nfe,
            }
            for key in (
                "latent_si_sdr_db_mean",
                "source_image_si_sdr_db_mean",
                "failure_rate_lt_5db",
                "measurement_residual_mean",
                "nmse_mean",
                "nmae_mean",
                "mean_nfe",
            ):
                if key in metrics:
                    row[key] = metrics[key]
            rows.append(row)
            write_csv(rows, root / "results.csv")

    print(f"\nSaved: {root / 'results.csv'}")


if __name__ == "__main__":
    main()

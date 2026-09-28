#!/usr/bin/env python3
"""
Matched alpha/beta training + evaluation sweep.

This script trains a SEPARATE model for each (alpha,beta) pair and evaluates
that model with the SAME (alpha,beta) values used during training.

Everything else is held fixed unless explicitly overridden:
  - dataset / filters
  - score network
  - sigma_min / sigma_max
  - mismatch probability p_sel
  - optimizer
  - number of training epochs
  - seed
  - sampler and NFE

This is the correct experiment for determining whether smaller alpha/beta
actually improve reverse separation, rather than merely creating a
training/inference mismatch.

Examples
--------
Minimal factorial test isolating alpha and beta:

    python scripts/run_alpha_beta_sweep.py \
      --alphas 1,3 \
      --betas 1,3 \
      --max-epochs 1000

Include the paper setting alpha=4,beta=3:

    python scripts/run_alpha_beta_sweep.py \
      --pairs '1,1;1,3;3,1;3,3;4,3' \
      --max-epochs 1000

Three seeds:

    python scripts/run_alpha_beta_sweep.py \
      --pairs '1,1;1,3;3,1;3,3;4,3' \
      --seeds 0,1,2 \
      --max-epochs 1000

Pilot first:

    python scripts/run_alpha_beta_sweep.py \
      --pairs '1,1;1,3;3,1;3,3;4,3' \
      --max-epochs 100 \
      --root sweeps/alpha_beta_pilot

Outputs
-------
<root>/
    a1_b1_seed2/
        train.log
        checkpoints/
        evaluation/summary.json
        evaluation/per_example.npz
    ...
    results_per_run.csv
    results_aggregate.csv
    plots/*.png
    plots/*.pdf
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


# -----------------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------------


def parse_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one value.")
    if any(v <= 0 for v in vals):
        raise ValueError("alpha and beta must be > 0.")
    return vals


def parse_int_list(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one seed.")
    return vals


def parse_pairs(text: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 2:
            raise ValueError(
                f"Bad pair {item!r}. Use --pairs '1,1;1,3;3,1;3,3'."
            )
        a, b = float(parts[0]), float(parts[1])
        if a <= 0 or b <= 0:
            raise ValueError("alpha and beta must be > 0.")
        pairs.append((a, b))
    if not pairs:
        raise ValueError("--pairs did not contain any valid pair.")
    return pairs


def fmt(v: float) -> str:
    """Filesystem-friendly float."""
    return f"{v:g}".replace("-", "m").replace(".", "p")


# -----------------------------------------------------------------------------
# Process helpers
# -----------------------------------------------------------------------------


def run_and_tee(cmd: list[str], cwd: Path, log_path: Path, dry_run: bool) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return

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
        code = proc.wait()

    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def find_checkpoint(run_dir: Path, policy: str) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoint directory: {ckpt_dir}")

    last = ckpt_dir / "last.ckpt"
    if policy == "last":
        if not last.exists():
            raise FileNotFoundError(f"Missing {last}")
        return last.resolve()

    # For 'best', Lightning stores the best_model_path in the callback state
    # inside last.ckpt. Try to recover it without relying on filename parsing.
    if last.exists():
        try:
            import torch

            try:
                payload = torch.load(
                    last,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                payload = torch.load(last, map_location="cpu")

            callbacks = payload.get("callbacks", {}) if isinstance(payload, dict) else {}
            for state in callbacks.values():
                if not isinstance(state, dict):
                    continue
                best_path = state.get("best_model_path")
                if best_path:
                    best_path = Path(str(best_path))
                    if best_path.exists():
                        return best_path.resolve()
                    local = ckpt_dir / best_path.name
                    if local.exists():
                        return local.resolve()
        except Exception as exc:
            print(f"Warning: could not read Lightning best_model_path: {exc}")

    # Fallback: when save_top_k=1 there should be exactly one non-last checkpoint.
    candidates = sorted(p for p in ckpt_dir.glob("*.ckpt") if p.name != "last.ckpt")
    if len(candidates) == 1:
        return candidates[0].resolve()
    if len(candidates) == 0 and last.exists():
        print("Warning: no top-k checkpoint found; using last.ckpt")
        return last.resolve()

    raise RuntimeError(
        f"Could not uniquely identify best checkpoint in {ckpt_dir}. "
        f"Candidates: {[p.name for p in candidates]}"
    )


# -----------------------------------------------------------------------------
# Result handling
# -----------------------------------------------------------------------------


METRICS = [
    "latent_si_sdr_db_mean",
    "source_image_si_sdr_db_mean",
    "nmse_mean",
    "nmae_mean",
    "failure_rate_lt_5db",
    "measurement_residual_mean",
    "mean_nfe",
]


def read_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_per_run_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "alpha",
        "beta",
        "seed",
        "checkpoint",
        "run_dir",
        *METRICS,
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def aggregate_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(float(row["alpha"]), float(row["beta"]))].append(row)

    out: list[dict] = []
    for (alpha, beta), items in sorted(groups.items()):
        r: dict[str, object] = {
            "alpha": alpha,
            "beta": beta,
            "num_seeds": len(items),
        }
        for metric in METRICS:
            vals = [float(x[metric]) for x in items if metric in x]
            if not vals:
                continue
            r[f"{metric}_mean"] = mean(vals)
            r[f"{metric}_std"] = stdev(vals) if len(vals) > 1 else 0.0
        out.append(r)
    return out


def write_aggregate_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields = ["alpha", "beta", "num_seeds"]
    for metric in METRICS:
        fields += [f"{metric}_mean", f"{metric}_std"]

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def make_plots(agg: list[dict], out_dir: Path) -> None:
    if not agg:
        return

    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [rf"$({r['alpha']:g},{r['beta']:g})$" for r in agg]
    x = list(range(len(agg)))

    plot_specs = [
        ("latent_si_sdr_db_mean", "Latent SI-SDR (dB)", "latent_si_sdr"),
        ("source_image_si_sdr_db_mean", "Source-image SI-SDR (dB)", "source_image_si_sdr"),
        ("failure_rate_lt_5db", "Failure rate (<5 dB)", "failure_rate"),
        ("measurement_residual_mean", "Measurement residual", "measurement_residual"),
        ("nmse_mean", "nMSE", "nmse"),
    ]

    for metric, ylabel, filename in plot_specs:
        key_m = f"{metric}_mean"
        key_s = f"{metric}_std"
        if not all(key_m in r for r in agg):
            continue

        y = [float(r[key_m]) for r in agg]
        yerr = [float(r.get(key_s, 0.0)) for r in agg]

        fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * len(agg)), 5.0))
        if any(v > 0 for v in yerr):
            ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3)
        else:
            ax.plot(x, y, marker="o")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_xlabel(r"$(\alpha,\beta)$ used in both training and sampling")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(out_dir / f"{filename}.png", dpi=300, bbox_inches="tight")
        fig.savefig(out_dir / f"{filename}.pdf", bbox_inches="tight")
        plt.close(fig)

    # If this is a complete Cartesian grid, also generate heatmaps.
    alphas = sorted({float(r["alpha"]) for r in agg})
    betas = sorted({float(r["beta"]) for r in agg})
    lookup = {(float(r["alpha"]), float(r["beta"])): r for r in agg}
    complete_grid = len(lookup) == len(alphas) * len(betas)

    if complete_grid and len(alphas) > 1 and len(betas) > 1:
        import numpy as np

        def edges(vals: list[float]) -> np.ndarray:
            a = np.asarray(vals, dtype=float)
            mids = 0.5 * (a[:-1] + a[1:])
            return np.concatenate(
                [[a[0] - 0.5 * (a[1] - a[0])], mids, [a[-1] + 0.5 * (a[-1] - a[-2])]]
            )

        for metric, ylabel, filename in plot_specs:
            key = f"{metric}_mean"
            if not all(key in r for r in agg):
                continue
            z = np.asarray(
                [[float(lookup[(a, b)][key]) for a in alphas] for b in betas],
                dtype=float,
            )
            fig, ax = plt.subplots(figsize=(6.7, 5.3))
            mesh = ax.pcolormesh(edges(alphas), edges(betas), z, shading="flat")
            ax.set_xticks(alphas)
            ax.set_yticks(betas)
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$\beta$")
            ax.set_title(ylabel)
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label(ylabel)
            fig.tight_layout()
            fig.savefig(out_dir / f"heatmap_{filename}.png", dpi=300, bbox_inches="tight")
            fig.savefig(out_dir / f"heatmap_{filename}.pdf", bbox_inches="tight")
            plt.close(fig)


# -----------------------------------------------------------------------------
# Main sweep
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="paper_k2_npz")

    group = ap.add_mutually_exclusive_group()
    group.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="Explicit pairs, e.g. '1,1;1,3;3,1;3,3;4,3'",
    )
    group.add_argument(
        "--grid",
        action="store_true",
        help="Use Cartesian product of --alphas and --betas (default behavior when --pairs is absent).",
    )

    ap.add_argument("--alphas", default="1,3,4")
    ap.add_argument("--betas", default="1,3")
    ap.add_argument("--seeds", default="2")
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--root", type=Path, default=Path("sweeps/alpha_beta"))

    ap.add_argument("--sampler", default="euler", choices=["euler", "heun", "rk4", "em", "pc"])
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument(
        "--checkpoint-policy",
        default="last",
        choices=["last", "best"],
        help=(
            "'last' is recommended for this ablation because all variants get exactly the same "
            "training budget. 'best' uses Lightning's validation-selected checkpoint."
        ),
    )

    ap.add_argument("--sigma-min", type=float, default=None)
    ap.add_argument("--sigma-max", type=float, default=None)
    ap.add_argument("--prior-sampling-prob", type=float, default=None)
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--force-eval", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    root = args.root
    if not root.is_absolute():
        root = (project_root / root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    if args.pairs is not None:
        pairs = parse_pairs(args.pairs)
    else:
        alphas = parse_float_list(args.alphas)
        betas = parse_float_list(args.betas)
        pairs = [(a, b) for a in alphas for b in betas]

    # Preserve order but remove duplicates.
    pairs = list(dict.fromkeys(pairs))
    seeds = parse_int_list(args.seeds)

    print("Project root:", project_root)
    print("Sweep root  :", root)
    print("Pairs       :", pairs)
    print("Seeds       :", seeds)
    print("Epochs      :", args.max_epochs)
    print("Sampler     :", args.sampler, args.num_steps, "steps")
    print("Checkpoint  :", args.checkpoint_policy)

    all_rows: list[dict] = []

    for alpha, beta in pairs:
        for seed in seeds:
            tag = f"a{fmt(alpha)}_b{fmt(beta)}_seed{seed}"
            run_dir = root / tag
            eval_dir = run_dir / "evaluation"
            summary_path = eval_dir / "summary.json"

            train_done = (run_dir / "checkpoints" / "last.ckpt").exists()

            if args.force_train or not train_done:
                run_dir.mkdir(parents=True, exist_ok=True)

                train_cmd = [
                    sys.executable,
                    "train.py",
                    f"experiment={args.experiment}",
                    f"seed={seed}",
                    f"name=alpha_beta_{tag}",
                    f"trainer.max_epochs={args.max_epochs}",
                    "save_top_k=1",
                    "train=true",
                    "test=false",
                    f"model.sde.drift_decay_rate={alpha}",
                    f"model.sde.zero_mode_decay_rate={beta}",
                    f"hydra.run.dir={run_dir}",
                    "hydra.job.chdir=true",
                ]

                if args.sigma_min is not None:
                    train_cmd.append(f"model.sde.sigma_min={args.sigma_min}")
                if args.sigma_max is not None:
                    train_cmd.append(f"model.sde.sigma_max={args.sigma_max}")
                if args.prior_sampling_prob is not None:
                    train_cmd.append(f"model.prior_sampling_prob={args.prior_sampling_prob}")

                run_and_tee(
                    train_cmd,
                    cwd=project_root,
                    log_path=run_dir / "train.log",
                    dry_run=args.dry_run,
                )
            else:
                print(f"\nSkipping training, found {run_dir / 'checkpoints' / 'last.ckpt'}")

            if args.dry_run:
                continue

            ckpt = find_checkpoint(run_dir, args.checkpoint_policy)
            print(f"Using checkpoint: {ckpt}")

            if args.force_eval or not summary_path.exists():
                eval_dir.mkdir(parents=True, exist_ok=True)

                eval_cmd = [
                    sys.executable,
                    "evaluate.py",
                    f"experiment={args.experiment}",
                    f"seed={seed}",
                    f"checkpoint={ckpt}",
                    f"model.sde.drift_decay_rate={alpha}",
                    f"model.sde.zero_mode_decay_rate={beta}",
                    f"model.sampler.method={args.sampler}",
                    f"model.sampler.num_steps={args.num_steps}",
                    f"evaluation_out={eval_dir}",
                    f"hydra.run.dir={run_dir / 'hydra_eval'}",
                    "hydra.job.chdir=false",
                ]

                if args.sigma_min is not None:
                    eval_cmd.append(f"model.sde.sigma_min={args.sigma_min}")
                if args.sigma_max is not None:
                    eval_cmd.append(f"model.sde.sigma_max={args.sigma_max}")
                if args.prior_sampling_prob is not None:
                    eval_cmd.append(f"model.prior_sampling_prob={args.prior_sampling_prob}")

                run_and_tee(
                    eval_cmd,
                    cwd=project_root,
                    log_path=run_dir / "evaluate.log",
                    dry_run=False,
                )
            else:
                print(f"Skipping evaluation, found {summary_path}")

            summary = read_summary(summary_path)
            row = {
                "alpha": alpha,
                "beta": beta,
                "seed": seed,
                "checkpoint": str(ckpt),
                "run_dir": str(run_dir),
            }
            for metric in METRICS:
                if metric in summary:
                    row[metric] = summary[metric]
            all_rows.append(row)

            # Re-write incrementally so interrupted sweeps keep completed results.
            write_per_run_csv(all_rows, root / "results_per_run.csv")
            agg = aggregate_rows(all_rows)
            write_aggregate_csv(agg, root / "results_aggregate.csv")
            make_plots(agg, root / "plots")

    if args.dry_run:
        print("\nDry run complete; no models were trained.")
        return

    agg = aggregate_rows(all_rows)

    print("\n" + "=" * 100)
    print("ALPHA/BETA SWEEP SUMMARY")
    print("=" * 100)
    header = (
        f"{'alpha':>7} {'beta':>7} {'n':>4} "
        f"{'latent SI-SDR':>15} {'image SI-SDR':>15} "
        f"{'failure':>12} {'residual':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in agg:
        print(
            f"{float(r['alpha']):7.3g} {float(r['beta']):7.3g} {int(r['num_seeds']):4d} "
            f"{float(r.get('latent_si_sdr_db_mean_mean', math.nan)):15.4f} "
            f"{float(r.get('source_image_si_sdr_db_mean_mean', math.nan)):15.4f} "
            f"{float(r.get('failure_rate_lt_5db_mean', math.nan)):12.4f} "
            f"{float(r.get('measurement_residual_mean_mean', math.nan)):12.6f}"
        )

    print("\nSaved:")
    print(" ", root / "results_per_run.csv")
    print(" ", root / "results_aggregate.csv")
    print(" ", root / "plots")


if __name__ == "__main__":
    main()

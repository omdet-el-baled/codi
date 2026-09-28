#!/usr/bin/env python3
"""
Evaluate already-trained alpha/beta models with Euler at multiple NFE values.

Purpose
-------
Test whether the performance gap between small and large beta is mainly caused
by numerical integration difficulty / stiffness.

NO retraining is done.

For each existing run such as

    sweeps/alpha_beta/a1_b1_seed2/
    sweeps/alpha_beta/a1_b3_seed2/
    sweeps/alpha_beta/a3_b1_seed2/
    sweeps/alpha_beta/a3_b3_seed2/

the script reuses its checkpoint and evaluates it with

    Euler 50, 100, 200, 500 steps.

Each model is evaluated with the SAME alpha and beta that were used to train it.

Outputs
-------
<root>/nfe_sweep/
    results_per_run.csv
    results_aggregate.csv
    beta_gap.csv
    plots/
        latent_si_sdr_vs_nfe.pdf/png
        source_image_si_sdr_vs_nfe.pdf/png
        measurement_residual_vs_nfe.pdf/png
        beta_gap_latent_vs_nfe.pdf/png
        beta_gap_image_vs_nfe.pdf/png

Interpretation
--------------
If beta=3 catches up to beta=1 as NFE increases, finite-step integration error
is an important contributor.

If the beta=1 vs beta=3 gap stays almost unchanged even at 500 Euler steps,
the dominant explanation is probably not Euler discretization alone.
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

import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    "latent_si_sdr_db_mean",
    "source_image_si_sdr_db_mean",
    "nmse_mean",
    "nmae_mean",
    "failure_rate_lt_5db",
    "measurement_residual_mean",
    "mean_nfe",
]


def parse_float_list(text: str) -> list[float]:
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def parse_pairs(text: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []

    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue

        parts = [v.strip() for v in item.split(",")]

        if len(parts) != 2:
            raise ValueError(
                f"Invalid pair '{item}'. Expected alpha,beta."
            )

        pairs.append(
            (
                float(parts[0]),
                float(parts[1]),
            )
        )

    if not pairs:
        raise ValueError("No alpha/beta pairs were provided.")

    return pairs


def fmt(v: float) -> str:
    """Match run_alpha_beta_sweep.py filesystem formatting."""
    return f"{v:g}".replace("-", "m").replace(".", "p")


def run_and_tee(
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    dry_run: bool,
) -> None:
    print("\n$ " + " ".join(cmd), flush=True)

    if dry_run:
        return

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as logf:

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
            print(
                line,
                end="",
                flush=True,
            )
            logf.write(line)
            logf.flush()

        code = proc.wait()

    if code != 0:
        raise subprocess.CalledProcessError(
            code,
            cmd,
        )


def find_checkpoint(
    run_dir: Path,
    policy: str,
) -> Path:
    ckpt_dir = run_dir / "checkpoints"

    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"No checkpoint directory: {ckpt_dir}"
        )

    last = ckpt_dir / "last.ckpt"

    if policy == "last":
        if not last.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: {last}"
            )
        return last.resolve()

    # Recover Lightning best_model_path from last.ckpt.
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
                payload = torch.load(
                    last,
                    map_location="cpu",
                )

            callbacks = (
                payload.get("callbacks", {})
                if isinstance(payload, dict)
                else {}
            )

            for state in callbacks.values():
                if not isinstance(state, dict):
                    continue

                best_path = state.get(
                    "best_model_path"
                )

                if not best_path:
                    continue

                best_path = Path(
                    str(best_path)
                )

                if best_path.exists():
                    return best_path.resolve()

                local = (
                    ckpt_dir
                    / best_path.name
                )

                if local.exists():
                    return local.resolve()

        except Exception as exc:
            print(
                "Warning: could not recover "
                f"best_model_path: {exc}"
            )

    candidates = sorted(
        p
        for p in ckpt_dir.glob("*.ckpt")
        if p.name != "last.ckpt"
    )

    if len(candidates) == 1:
        return candidates[0].resolve()

    if len(candidates) == 0 and last.exists():
        print(
            "Warning: no unique top-k checkpoint "
            "found; using last.ckpt."
        )
        return last.resolve()

    raise RuntimeError(
        "Could not uniquely determine best "
        f"checkpoint in {ckpt_dir}. "
        f"Candidates={[p.name for p in candidates]}"
    )


def read_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def write_csv(
    rows: list[dict],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    fields: list[str] = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def aggregate_rows(
    rows: list[dict],
) -> list[dict]:
    """
    Aggregate repeated seeds by (alpha,beta,requested_steps).
    """
    grouped: dict[
        tuple[float, float, int],
        list[dict],
    ] = defaultdict(list)

    for row in rows:
        key = (
            float(row["alpha"]),
            float(row["beta"]),
            int(row["requested_steps"]),
        )
        grouped[key].append(row)

    output: list[dict] = []

    for (
        alpha,
        beta,
        requested_steps,
    ), group in sorted(grouped.items()):

        result = {
            "alpha": alpha,
            "beta": beta,
            "requested_steps": requested_steps,
            "n": len(group),
        }

        for metric in METRICS:
            vals = [
                float(r[metric])
                for r in group
                if metric in r
                and r[metric] is not None
                and np.isfinite(
                    float(r[metric])
                )
            ]

            if vals:
                result[f"{metric}_mean"] = float(
                    np.mean(vals)
                )

                result[f"{metric}_std"] = float(
                    np.std(
                        vals,
                        ddof=1,
                    )
                    if len(vals) > 1
                    else 0.0
                )

        output.append(result)

    return output


def build_beta_gap_rows(
    aggregate: list[dict],
    beta_low: float,
    beta_high: float,
) -> list[dict]:
    """
    For every alpha for which both beta_low and beta_high exist,
    compute:

        gap = metric(beta_low) - metric(beta_high)

    Positive SI-SDR gap means beta_low is better.
    """
    index = {
        (
            float(r["alpha"]),
            float(r["beta"]),
            int(r["requested_steps"]),
        ): r
        for r in aggregate
    }

    alphas = sorted(
        {
            float(r["alpha"])
            for r in aggregate
        }
    )

    steps = sorted(
        {
            int(r["requested_steps"])
            for r in aggregate
        }
    )

    rows = []

    for alpha in alphas:
        for step in steps:

            lo = index.get(
                (
                    alpha,
                    beta_low,
                    step,
                )
            )

            hi = index.get(
                (
                    alpha,
                    beta_high,
                    step,
                )
            )

            if lo is None or hi is None:
                continue

            row = {
                "alpha": alpha,
                "beta_low": beta_low,
                "beta_high": beta_high,
                "requested_steps": step,
            }

            for metric in [
                "latent_si_sdr_db_mean",
                "source_image_si_sdr_db_mean",
                "measurement_residual_mean",
                "nmse_mean",
            ]:

                key = f"{metric}_mean"

                if key in lo and key in hi:
                    row[
                        f"{metric}_gap_low_minus_high"
                    ] = (
                        float(lo[key])
                        - float(hi[key])
                    )

            # Use actual measured NFE if present.
            nfe_key = "mean_nfe_mean"

            if (
                nfe_key in lo
                and nfe_key in hi
            ):
                row["mean_nfe"] = (
                    float(lo[nfe_key])
                    + float(hi[nfe_key])
                ) / 2.0
            else:
                row["mean_nfe"] = float(step)

            rows.append(row)

    return rows


def x_value(
    row: dict,
) -> float:
    value = row.get(
        "mean_nfe_mean"
    )

    if value is None:
        value = row.get(
            "requested_steps"
        )

    return float(value)


def make_metric_plot(
    aggregate: list[dict],
    metric: str,
    ylabel: str,
    out: Path,
) -> None:
    key = f"{metric}_mean"

    groups: dict[
        tuple[float, float],
        list[dict],
    ] = defaultdict(list)

    for row in aggregate:
        if key in row:
            groups[
                (
                    float(row["alpha"]),
                    float(row["beta"]),
                )
            ].append(row)

    if not groups:
        return

    fig, ax = plt.subplots(
        figsize=(8.2, 5.5)
    )

    for (alpha, beta), rows in sorted(
        groups.items()
    ):
        rows = sorted(
            rows,
            key=x_value,
        )

        xs = [
            x_value(r)
            for r in rows
        ]

        ys = [
            float(r[key])
            for r in rows
        ]

        yerr_key = f"{metric}_std"

        yerr = [
            float(
                r.get(
                    yerr_key,
                    0.0,
                )
            )
            for r in rows
        ]

        if any(v > 0 for v in yerr):
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                marker="o",
                label=(
                    rf"$\alpha={alpha:g}, "
                    rf"\beta={beta:g}$"
                ),
            )
        else:
            ax.plot(
                xs,
                ys,
                marker="o",
                label=(
                    rf"$\alpha={alpha:g}, "
                    rf"\beta={beta:g}$"
                ),
            )

    ax.set_xlabel(
        "Euler NFE"
    )
    ax.set_ylabel(
        ylabel
    )
    ax.set_title(
        f"{ylabel} vs Euler NFE"
    )
    ax.grid(
        True,
        linestyle="--",
        alpha=0.35,
    )
    ax.legend()
    fig.tight_layout()

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        out.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        out.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)


def make_gap_plot(
    gap_rows: list[dict],
    metric_gap_key: str,
    ylabel: str,
    out: Path,
) -> None:
    groups: dict[
        float,
        list[dict],
    ] = defaultdict(list)

    for row in gap_rows:
        if metric_gap_key in row:
            groups[
                float(row["alpha"])
            ].append(row)

    if not groups:
        return

    fig, ax = plt.subplots(
        figsize=(8.2, 5.5)
    )

    for alpha, rows in sorted(
        groups.items()
    ):
        rows = sorted(
            rows,
            key=lambda r: float(
                r["mean_nfe"]
            ),
        )

        xs = [
            float(r["mean_nfe"])
            for r in rows
        ]

        ys = [
            float(r[metric_gap_key])
            for r in rows
        ]

        ax.plot(
            xs,
            ys,
            marker="o",
            label=rf"$\alpha={alpha:g}$",
        )

    ax.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
    )

    ax.set_xlabel(
        "Euler NFE"
    )
    ax.set_ylabel(
        ylabel
    )
    ax.set_title(
        "Does the beta gap shrink with finer Euler integration?"
    )
    ax.grid(
        True,
        linestyle="--",
        alpha=0.35,
    )
    ax.legend()
    fig.tight_layout()

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        out.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        out.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)


def print_summary(
    aggregate: list[dict],
) -> None:
    print()
    print("=" * 112)
    print("EULER NFE SWEEP SUMMARY")
    print("=" * 112)

    header = (
        f"{'alpha':>7} "
        f"{'beta':>7} "
        f"{'steps':>7} "
        f"{'NFE':>9} "
        f"{'latent SI-SDR':>16} "
        f"{'image SI-SDR':>16} "
        f"{'failure':>11} "
        f"{'residual':>12}"
    )

    print(header)
    print("-" * len(header))

    for r in aggregate:

        print(
            f"{float(r['alpha']):7g} "
            f"{float(r['beta']):7g} "
            f"{int(r['requested_steps']):7d} "
            f"{float(r.get('mean_nfe_mean', math.nan)):9.1f} "
            f"{float(r.get('latent_si_sdr_db_mean_mean', math.nan)):16.4f} "
            f"{float(r.get('source_image_si_sdr_db_mean_mean', math.nan)):16.4f} "
            f"{float(r.get('failure_rate_lt_5db_mean', math.nan)):11.4f} "
            f"{float(r.get('measurement_residual_mean_mean', math.nan)):12.6f}"
        )

    print("=" * 112)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        type=Path,
        default=Path(
            "sweeps/alpha_beta"
        ),
        help=(
            "Existing alpha/beta training sweep root."
        ),
    )

    ap.add_argument(
        "--pairs",
        type=str,
        default=(
            "1,1;1,3;3,1;3,3;4,3"
        ),
    )

    ap.add_argument(
        "--seeds",
        type=str,
        default="2",
    )

    ap.add_argument(
        "--steps",
        type=str,
        default="50,100,200,500",
    )

    ap.add_argument(
        "--experiment",
        type=str,
        default="paper_k2_npz",
    )

    ap.add_argument(
        "--checkpoint-policy",
        choices=[
            "last",
            "best",
        ],
        default="last",
    )

    ap.add_argument(
        "--beta-low",
        type=float,
        default=1.0,
    )

    ap.add_argument(
        "--beta-high",
        type=float,
        default=3.0,
    )

    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-run evaluations even if summary.json already exists.",
    )

    ap.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = ap.parse_args()

    project_root = Path.cwd().resolve()

    if not (
        project_root / "evaluate.py"
    ).exists():
        raise FileNotFoundError(
            "Run this script from the project root "
            "containing evaluate.py."
        )

    root = args.root

    if not root.is_absolute():
        root = (
            project_root
            / root
        ).resolve()

    pairs = list(
        dict.fromkeys(
            parse_pairs(args.pairs)
        )
    )

    seeds = parse_int_list(
        args.seeds
    )

    steps = parse_int_list(
        args.steps
    )

    if any(v <= 0 for v in steps):
        raise ValueError(
            "All Euler step counts must be positive."
        )

    output_root = (
        root
        / "nfe_sweep"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Project root :", project_root)
    print("Training root:", root)
    print("Output root  :", output_root)
    print("Pairs        :", pairs)
    print("Seeds        :", seeds)
    print("Euler steps  :", steps)
    print(
        "Checkpoint   :",
        args.checkpoint_policy,
    )

    all_rows: list[dict] = []

    for alpha, beta in pairs:

        for seed in seeds:

            tag = (
                f"a{fmt(alpha)}_"
                f"b{fmt(beta)}_"
                f"seed{seed}"
            )

            run_dir = (
                root / tag
            )

            if not run_dir.exists():
                print(
                    f"\nWARNING: run does not exist, "
                    f"skipping: {run_dir}"
                )
                continue

            ckpt = find_checkpoint(
                run_dir,
                args.checkpoint_policy,
            )

            print(
                f"\nUsing {tag}: {ckpt}"
            )

            for num_steps in steps:

                eval_dir = (
                    run_dir
                    / "evaluation_nfe"
                    / f"euler_{num_steps}"
                )

                summary_path = (
                    eval_dir
                    / "summary.json"
                )

                if (
                    args.force
                    or not summary_path.exists()
                ):

                    eval_cmd = [
                        sys.executable,
                        "evaluate.py",
                        f"experiment={args.experiment}",
                        f"seed={seed}",
                        f"checkpoint={ckpt}",
                        (
                            "model.sde."
                            f"drift_decay_rate={alpha}"
                        ),
                        (
                            "model.sde."
                            f"zero_mode_decay_rate={beta}"
                        ),
                        "model.sampler.method=euler",
                        (
                            "model.sampler."
                            f"num_steps={num_steps}"
                        ),
                        f"evaluation_out={eval_dir}",
                        (
                            "hydra.run.dir="
                            f"{run_dir / 'hydra_eval_nfe' / f'euler_{num_steps}'}"
                        ),
                        "hydra.job.chdir=false",
                    ]

                    run_and_tee(
                        eval_cmd,
                        cwd=project_root,
                        log_path=(
                            eval_dir
                            / "evaluate.log"
                        ),
                        dry_run=args.dry_run,
                    )

                else:
                    print(
                        "Skipping existing evaluation: "
                        f"{summary_path}"
                    )

                if args.dry_run:
                    continue

                summary = read_json(
                    summary_path
                )

                row = {
                    "alpha": alpha,
                    "beta": beta,
                    "seed": seed,
                    "requested_steps": num_steps,
                    "checkpoint": str(ckpt),
                    "run_dir": str(run_dir),
                    "evaluation_dir": str(
                        eval_dir
                    ),
                }

                for metric in METRICS:
                    if metric in summary:
                        row[metric] = (
                            summary[metric]
                        )

                all_rows.append(row)

                # Save incrementally.
                write_csv(
                    all_rows,
                    output_root
                    / "results_per_run.csv",
                )

                aggregate = aggregate_rows(
                    all_rows
                )

                write_csv(
                    aggregate,
                    output_root
                    / "results_aggregate.csv",
                )

    if args.dry_run:
        print(
            "\nDry run complete. "
            "No evaluations were executed."
        )
        return

    aggregate = aggregate_rows(
        all_rows
    )

    beta_gap = build_beta_gap_rows(
        aggregate,
        beta_low=args.beta_low,
        beta_high=args.beta_high,
    )

    write_csv(
        beta_gap,
        output_root
        / "beta_gap.csv",
    )

    plots = (
        output_root
        / "plots"
    )

    make_metric_plot(
        aggregate,
        metric="latent_si_sdr_db_mean",
        ylabel="Latent SI-SDR (dB)",
        out=(
            plots
            / "latent_si_sdr_vs_nfe"
        ),
    )

    make_metric_plot(
        aggregate,
        metric="source_image_si_sdr_db_mean",
        ylabel="Source-image SI-SDR (dB)",
        out=(
            plots
            / "source_image_si_sdr_vs_nfe"
        ),
    )

    make_metric_plot(
        aggregate,
        metric="measurement_residual_mean",
        ylabel="Measurement residual",
        out=(
            plots
            / "measurement_residual_vs_nfe"
        ),
    )

    make_metric_plot(
        aggregate,
        metric="nmse_mean",
        ylabel="nMSE",
        out=(
            plots
            / "nmse_vs_nfe"
        ),
    )

    make_gap_plot(
        beta_gap,
        metric_gap_key=(
            "latent_si_sdr_db_mean_"
            "gap_low_minus_high"
        ),
        ylabel=(
            rf"SI-SDR gap "
            rf"$\beta={args.beta_low:g}"
            rf" - \beta={args.beta_high:g}$ (dB)"
        ),
        out=(
            plots
            / "beta_gap_latent_vs_nfe"
        ),
    )

    make_gap_plot(
        beta_gap,
        metric_gap_key=(
            "source_image_si_sdr_db_mean_"
            "gap_low_minus_high"
        ),
        ylabel=(
            rf"Image SI-SDR gap "
            rf"$\beta={args.beta_low:g}"
            rf" - \beta={args.beta_high:g}$ (dB)"
        ),
        out=(
            plots
            / "beta_gap_image_vs_nfe"
        ),
    )

    print_summary(
        aggregate
    )

    print()
    print("Saved:")
    print(
        " ",
        output_root
        / "results_per_run.csv",
    )
    print(
        " ",
        output_root
        / "results_aggregate.csv",
    )
    print(
        " ",
        output_root
        / "beta_gap.csv",
    )
    print(
        " ",
        plots,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Launch matched sampler evaluations from one checkpoint."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--experiment", default="paper_k2_npz")
    ap.add_argument("--out", default="evaluation/samplers")
    ap.add_argument("--budget", type=int, default=50, help="Approximate score-evaluation budget")
    args = ap.parse_args()
    # Approximate equal-NFE mapping. PC assumes one corrector step.
    methods = {
        "euler": args.budget,
        "heun": max(1, args.budget // 2),
        "rk4": max(1, args.budget // 4),
        "em": args.budget,
        "pc": max(1, args.budget // 2),
    }
    for method, steps in methods.items():
        cmd = [
            sys.executable, "evaluate.py",
            f"experiment={args.experiment}",
            f"checkpoint={args.checkpoint}",
            f"model.sampler.method={method}",
            f"model.sampler.num_steps={steps}",
            f"evaluation_out={Path(args.out) / method}",
        ]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

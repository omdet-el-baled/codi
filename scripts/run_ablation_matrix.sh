#!/usr/bin/env bash
set -euo pipefail
# Run from repository root. Add extra Hydra overrides at the end if needed.
python train.py experiment=paper_k2_npz "$@"
python train.py experiment=no_mismatch path.datasets.mixdata=data/mechanical_k2_npz path.datasets.filters_path=data/mechanical_k2_npz/filters.npz "$@"
python train.py experiment=no_drift path.datasets.mixdata=data/mechanical_k2_npz path.datasets.filters_path=data/mechanical_k2_npz/filters.npz "$@"
python train.py experiment=no_drift_no_mismatch path.datasets.mixdata=data/mechanical_k2_npz path.datasets.filters_path=data/mechanical_k2_npz/filters.npz "$@"

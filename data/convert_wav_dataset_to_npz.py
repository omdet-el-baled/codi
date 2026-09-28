#!/usr/bin/env python3
"""Convert the original WAV dataset tree to compact split-wise NPZ archives.

The converter is deliberately separate from the original generators: it lets you
reuse an already generated/validated dataset while eliminating thousands of WAV
opens and repeated FFTs during training.

Example
-------
python data/convert_wav_dataset_to_npz.py \
  --input data/sim_2src --output data/sim_2src_npz \
  --filters data/sim_2src/filters/filters.npy --num-sources 2 \
  --sample-rate 2000 --fft-norm ortho --cache-fft
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio

SPLITS = {"tr": "train", "cv": "val", "tt": "test"}


def load_filters(path: Path):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.lib.npyio.NpzFile):
        if "filters" not in obj.files:
            raise KeyError(f"{path} has no 'filters' key")
        filters = obj["filters"]
        lengths = obj["filter_lengths"] if "filter_lengths" in obj.files else None
        if lengths is not None:
            return [np.asarray(filters[i, : int(lengths[i])], np.float32) for i in range(len(lengths))]
        return [np.asarray(x, np.float32) for x in filters]
    return [np.asarray(x, np.float32) for x in obj]


def load_split(root: Path, split: str, k: int, sample_rate: int):
    mix_dir = root / split / "mix"
    names = sorted(p.name for p in mix_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    sources, mixtures = [], []
    for name in names:
        y, sr = torchaudio.load(mix_dir / name)
        if sr != sample_rate:
            raise ValueError(f"{name}: sample rate {sr} != {sample_rate}")
        mixtures.append(y.squeeze(0).numpy().astype(np.float32))
        src = []
        for j in range(k):
            x, sr_x = torchaudio.load(root / split / f"s{j+1}" / name)
            if sr_x != sample_rate:
                raise ValueError(f"{name}/s{j+1}: sample rate {sr_x} != {sample_rate}")
            src.append(x.squeeze(0).numpy().astype(np.float32))
        sources.append(np.stack(src, axis=0))
    return np.stack(sources), np.stack(mixtures), np.asarray(names, dtype="U")


def save_archive(path: Path, payload: dict, compressed: bool):
    fn = np.savez_compressed if compressed else np.savez
    fn(path, **payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--filters", type=Path, required=True)
    ap.add_argument("--num-sources", type=int, required=True)
    ap.add_argument("--sample-rate", type=int, default=2000)
    ap.add_argument("--fft-norm", type=str, default="ortho")
    ap.add_argument("--cache-fft", action="store_true")
    ap.add_argument("--uncompressed", action="store_true", help="Use np.savez instead of np.savez_compressed")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    filters = load_filters(args.filters)
    if len(filters) != args.num_sources:
        raise ValueError(f"Expected {args.num_sources} filters, found {len(filters)}")
    max_filter_len = max(len(h) for h in filters)

    # Keep filter storage rectangular and record true lengths.
    filt = np.zeros((args.num_sources, max_filter_len), dtype=np.float32)
    lengths = np.asarray([len(h) for h in filters], dtype=np.int64)
    for i, h in enumerate(filters):
        filt[i, : len(h)] = h
    np.savez_compressed(
        args.output / "filters.npz",
        filters=filt,
        filter_lengths=lengths,
        sample_rate=np.int64(args.sample_rate),
    )

    summary = {
        "sample_rate": args.sample_rate,
        "num_sources": args.num_sources,
        "fft_norm": args.fft_norm,
        "cache_fft": bool(args.cache_fft),
        "splits": {},
    }

    for code, label in SPLITS.items():
        sources, mixtures, ids = load_split(args.input, code, args.num_sources, args.sample_rate)
        # Match the original training convention: zero pad clean/source-image WAVs
        # to signal_length + max_filter_length - 1 before the one-sided DFT.
        n_fft = int(sources.shape[-1] + max_filter_len - 1)
        payload = {
            "sources": sources.astype(np.float32, copy=False),
            "mixtures": mixtures.astype(np.float32, copy=False),
            "sample_ids": ids,
            "sample_rate": np.int64(args.sample_rate),
            "n_fft": np.int64(n_fft),
            "fft_norm": np.asarray(args.fft_norm),
        }
        if args.cache_fft:
            xs = torch.from_numpy(sources)
            ys = torch.from_numpy(mixtures)
            payload["sources_fft"] = torch.fft.rfft(xs, n=n_fft, dim=-1, norm=args.fft_norm).numpy().astype(np.complex64)
            payload["mixtures_fft"] = torch.fft.rfft(ys, n=n_fft, dim=-1, norm=args.fft_norm).numpy().astype(np.complex64)
        save_archive(args.output / f"{code}.npz", payload, compressed=not args.uncompressed)
        summary["splits"][label] = int(len(sources))

    with open(args.output / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

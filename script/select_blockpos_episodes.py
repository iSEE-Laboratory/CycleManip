#!/usr/bin/env python3
"""Select representative source episodes by an (approx.) block-position proxy.

Background
- We want to pick a subset of source episodes such that the (proxy) block positions
  cover the workspace more uniformly.

Proxy (per user hint)
- Use frame-45 (t=45) `endpose` as an approximation of the block position.
- In this dataset, right-arm slice `endpose[t, 7:14]` has meaningful variation
  across episodes; we use its (x,y) as the proxy by default.

Method
- Farthest Point Sampling (FPS) in 2D over the proxy points.

Outputs
- CSV: episode_id,x,y (for all episodes)
- JSON: selected episode_ids + suggested xlim/ylim (robust quantiles)

Example
```bash
python script/select_blockpos_episodes.py \
  --data-root /abs/path/to/episode_hdf5_dir \
  --frame-idx 45 --arm right --k 50 \
  --out-json /abs/path/to/uniform50.json \
  --out-csv  /abs/path/to/t45_right_xy.csv
```

Notes
- `--data-root` can be under RoboTwin's data folder (or any external path) as long
  as it contains `episode*.hdf5` with an `endpose` dataset.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import h5py  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency 'h5py'. Install it in your env (e.g. pip install h5py)."
    ) from e


@dataclass
class SelectionResult:
    data_root: str
    frame_idx: int
    arm: str
    method: str
    k: int
    episode_ids_sorted: List[int]
    xlim: List[float]
    ylim: List[float]
    quantiles: dict
    stats_all: dict
    stats_selected: dict


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing episode*.hdf5",
    )
    p.add_argument(
        "--glob",
        type=str,
        default="episode*.hdf5",
        help="Glob pattern under data-root.",
    )
    p.add_argument(
        "--frame-idx",
        type=int,
        default=45,
        help="Frame index to take from endpose (0-based).",
    )
    p.add_argument(
        "--arm",
        type=str,
        default="right",
        choices=["right", "left"],
        help="Which arm slice in endpose to use as proxy.",
    )
    p.add_argument(
        "--k",
        type=int,
        default=50,
        help="How many episodes to select.",
    )
    p.add_argument(
        "--method",
        type=str,
        default="fps",
        choices=["fps"],
        help="Selection method. fps=farthest point sampling.",
    )
    p.add_argument(
        "--start",
        type=str,
        default="center",
        choices=["center", "random"],
        help="FPS starting point strategy.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (used when --start=random).",
    )
    p.add_argument(
        "--quantile-lo",
        type=float,
        default=1.0,
        help="Lower quantile (percent) for xlim/ylim, e.g. 1.0",
    )
    p.add_argument(
        "--quantile-hi",
        type=float,
        default=99.0,
        help="Upper quantile (percent) for xlim/ylim, e.g. 99.0",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=0.01,
        help="Margin added to quantile bounds for xlim/ylim.",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        required=True,
        help="Output JSON path.",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    return p.parse_args()


def _episode_id_from_name(path: Path) -> int:
    # episode123.hdf5 -> 123
    s = path.stem
    if not s.startswith("episode"):
        raise ValueError(f"Unexpected episode filename: {path.name}")
    return int(s[len("episode") :])


def _load_xy(data_root: Path, glob_pat: str, frame_idx: int, arm: str) -> Tuple[np.ndarray, np.ndarray]:
    paths = sorted(data_root.glob(glob_pat), key=_episode_id_from_name)
    if not paths:
        raise FileNotFoundError(f"No files matched {data_root}/{glob_pat}")

    ep_ids = np.array([_episode_id_from_name(p) for p in paths], dtype=int)

    xy = np.zeros((len(paths), 2), dtype=np.float64)
    sl = slice(7, 14) if arm == "right" else slice(0, 7)

    for i, p in enumerate(paths):
        with h5py.File(p, "r") as f:
            if "endpose" not in f:
                raise KeyError(f"{p} missing 'endpose'")
            end = f["endpose"][:]
            if end.ndim != 2 or end.shape[1] != 14:
                raise ValueError(f"{p} endpose has unexpected shape {end.shape}")
            if end.shape[0] <= frame_idx:
                raise ValueError(f"{p} has only T={end.shape[0]} frames (<= {frame_idx})")
            pose = end[frame_idx, sl]
            xy[i] = pose[0:2]

    if not np.isfinite(xy).all():
        bad = np.where(~np.isfinite(xy))
        raise ValueError(f"Non-finite values in xy at indices {bad}")

    return ep_ids, xy


def _fps_select(xy: np.ndarray, k: int, start: str, seed: int) -> np.ndarray:
    n = xy.shape[0]
    if not (1 <= k <= n):
        raise ValueError(f"k must be in [1,{n}], got {k}")

    if start == "center":
        c = xy.mean(axis=0)
        start_idx = int(np.argmin(((xy - c) ** 2).sum(axis=1)))
    elif start == "random":
        rng = np.random.default_rng(seed)
        start_idx = int(rng.integers(0, n))
    else:
        raise ValueError(f"Unknown start strategy: {start}")

    selected = np.empty((k,), dtype=int)
    selected[0] = start_idx

    min_d2 = ((xy - xy[start_idx]) ** 2).sum(axis=1)
    min_d2[start_idx] = 0.0

    for t in range(1, k):
        i = int(np.argmax(min_d2))
        selected[t] = i
        d2 = ((xy - xy[i]) ** 2).sum(axis=1)
        min_d2 = np.minimum(min_d2, d2)
        min_d2[i] = 0.0

    if len(set(selected.tolist())) != k:
        raise RuntimeError("FPS produced duplicate indices (unexpected).")

    return selected


def _stats(arr: np.ndarray) -> dict:
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
    }


def main() -> None:
    args = _parse_args()

    ep_ids, xy = _load_xy(args.data_root, args.glob, args.frame_idx, args.arm)

    if args.method == "fps":
        sel_idx = _fps_select(xy, args.k, args.start, args.seed)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    selected_ep_ids = ep_ids[sel_idx]
    selected_ep_ids_sorted = sorted(int(x) for x in selected_ep_ids.tolist())

    q_lo = float(args.quantile_lo)
    q_hi = float(args.quantile_hi)
    if not (0.0 <= q_lo < q_hi <= 100.0):
        raise ValueError("quantile-lo/hi must satisfy 0 <= lo < hi <= 100")

    q = np.percentile(xy, [q_lo, q_hi], axis=0)  # shape (2,2)
    margin = float(args.margin)
    xlim = [float(q[0, 0] - margin), float(q[1, 0] + margin)]
    ylim = [float(q[0, 1] - margin), float(q[1, 1] + margin)]

    out = SelectionResult(
        data_root=str(args.data_root),
        frame_idx=int(args.frame_idx),
        arm=str(args.arm),
        method=str(args.method),
        k=int(args.k),
        episode_ids_sorted=selected_ep_ids_sorted,
        xlim=xlim,
        ylim=ylim,
        quantiles={
            "lo": q_lo,
            "hi": q_hi,
            "bounds_xy": q.tolist(),
            "margin": margin,
        },
        stats_all=_stats(xy),
        stats_selected=_stats(xy[sel_idx]),
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    args.out_json.write_text(
        json.dumps(asdict(out), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rows = []
    for eid, (x, y) in sorted(zip(ep_ids.tolist(), xy.tolist()), key=lambda t: t[0]):
        rows.append((int(eid), float(x), float(y)))

    with args.out_csv.open("w", encoding="utf-8") as f:
        f.write("episode_id,x,y\n")
        for eid, x, y in rows:
            f.write(f"{eid},{x:.10f},{y:.10f}\n")

    print("selected_episode_ids_sorted:")
    print(selected_ep_ids_sorted)
    print("suggested_xlim:", [round(v, 6) for v in xlim])
    print("suggested_ylim:", [round(v, 6) for v in ylim])
    print("wrote:", str(args.out_json))
    print("wrote:", str(args.out_csv))


if __name__ == "__main__":
    main()

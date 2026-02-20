#!/usr/bin/env python3
"""Analyze joint_action/state diversity between a source HDF5 and a directory of generated episodes.

This script is intentionally dependency-light: numpy + h5py.

Example:
  python RoboTwin/script/analyze_joint_action_diversity.py \
	--source /path/to/source/episode0.hdf5 \
	--synth-dir /path/to/generated/data
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
	import h5py
except Exception as e:
	raise RuntimeError("h5py is required. Please `pip install h5py` in your active env.") from e


def _first_existing_dataset(f: h5py.File, candidates: list[str]) -> Optional[np.ndarray]:
	for k in candidates:
		try:
			if k in f:
				return np.asarray(f[k])
		except Exception:
			pass
	return None


def _read_action(f: h5py.File) -> Optional[np.ndarray]:
	# Common layouts:
	#  - joint_action/vector
	#  - action
	#  - joint_action (dataset)
	a = _first_existing_dataset(
		f,
		[
			"joint_action/vector",
			"action",
			"joint_action",
		],
	)
	if a is None:
		return None
	a = np.asarray(a)
	if a.ndim == 1:
		a = a.reshape(-1, a.shape[0])
	return a


def _read_state(f: h5py.File) -> Optional[np.ndarray]:
	# We mainly care about end-effector XYZ (first 3 dims).
	s = _first_existing_dataset(
		f,
		[
			"endpose",
			"state",
			"agent_pos",
			"end_pose",
		],
	)
	if s is None:
		return None
	s = np.asarray(s)
	if s.ndim == 1:
		s = s.reshape(-1, s.shape[0])
	return s


def _stable_hash(arr: np.ndarray, *, round_decimals: int = 6) -> str:
	"""Hash float arrays robustly by rounding, then hashing bytes."""

	if arr is None:
		return "NONE"
	a = np.asarray(arr)
	if a.size == 0:
		return "EMPTY"
	if np.issubdtype(a.dtype, np.floating):
		a = np.round(a.astype(np.float64), round_decimals)
	h = hashlib.md5(a.tobytes()).hexdigest()
	return h


@dataclass
class EpisodeStats:
	path: str
	T: int
	action_dim: int
	state_dim: int
	action_hash: str
	state_hash: str
	action_mean: np.ndarray
	action_std: np.ndarray
	ee_start: Optional[np.ndarray]
	ee_end: Optional[np.ndarray]


def _episode_stats(path: str) -> EpisodeStats:
	with h5py.File(path, "r") as f:
		a = _read_action(f)
		s = _read_state(f)

	if a is None:
		raise KeyError(f"No action dataset found in {path}")

	T = int(a.shape[0])
	action_dim = int(a.shape[1]) if a.ndim >= 2 else 1

	if s is None:
		state_dim = 0
		ee_start = None
		ee_end = None
		state_hash = "NONE"
	else:
		state_dim = int(s.shape[1]) if s.ndim >= 2 else 1
		ee_start = np.asarray(s[0, :3], dtype=np.float64) if s.shape[1] >= 3 else None
		ee_end = np.asarray(s[-1, :3], dtype=np.float64) if s.shape[1] >= 3 else None
		state_hash = _stable_hash(s)

	a64 = np.asarray(a, dtype=np.float64)
	action_mean = np.nanmean(a64, axis=0)
	action_std = np.nanstd(a64, axis=0)

	return EpisodeStats(
		path=path,
		T=T,
		action_dim=action_dim,
		state_dim=state_dim,
		action_hash=_stable_hash(a),
		state_hash=state_hash,
		action_mean=action_mean,
		action_std=action_std,
		ee_start=ee_start,
		ee_end=ee_end,
	)


def _compare_traj(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, int]:
	"""Return (rmse, max_abs, T_used) on first min length."""

	if a is None or b is None:
		return float("nan"), float("nan"), 0
	a = np.asarray(a, dtype=np.float64)
	b = np.asarray(b, dtype=np.float64)
	T = min(int(a.shape[0]), int(b.shape[0]))
	if T <= 0:
		return float("nan"), float("nan"), 0
	aa = a[:T]
	bb = b[:T]
	diff = aa - bb
	rmse = float(np.sqrt(np.mean(diff * diff)))
	max_abs = float(np.max(np.abs(diff)))
	return rmse, max_abs, T


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--source", required=True, help="Source episode HDF5 (e.g., episode0.hdf5)")
	ap.add_argument("--synth-dir", required=True, help="Directory containing generated *.hdf5 episodes")
	ap.add_argument("--pattern", default="episode*.hdf5", help="Glob pattern under synth-dir")
	ap.add_argument("--round-decimals", type=int, default=6, help="Rounding decimals for hash")
	args = ap.parse_args()

	src_path = os.path.expanduser(args.source)
	synth_dir = os.path.expanduser(args.synth_dir)

	if not os.path.exists(src_path):
		raise FileNotFoundError(src_path)
	if not os.path.isdir(synth_dir):
		raise NotADirectoryError(synth_dir)

	# Load source arrays for comparisons
	with h5py.File(src_path, "r") as f:
		src_a = _read_action(f)
		src_s = _read_state(f)

	if src_a is None:
		raise KeyError(f"No action dataset found in source: {src_path}")

	src_a_hash = _stable_hash(src_a, round_decimals=args.round_decimals)
	src_s_hash = _stable_hash(src_s, round_decimals=args.round_decimals) if src_s is not None else "NONE"

	paths = sorted(glob.glob(os.path.join(synth_dir, args.pattern)))
	if not paths:
		raise FileNotFoundError(f"No files matched: {os.path.join(synth_dir, args.pattern)}")

	stats = []
	for p in paths:
		st = _episode_stats(p)
		stats.append(st)

	# Aggregate
	Ts = np.array([s.T for s in stats], dtype=np.int64)
	action_dims = {s.action_dim for s in stats}
	state_dims = {s.state_dim for s in stats}

	action_hashes = [s.action_hash for s in stats]
	state_hashes = [s.state_hash for s in stats]

	uniq_action = len(set(action_hashes))
	uniq_state = len(set(state_hashes))

	# Compare to source (action/state)
	src_action_same_count = sum(1 for h in action_hashes if h == src_a_hash)
	src_state_same_count = sum(1 for h in state_hashes if h == src_s_hash)

	# Trajectory distance to source
	rmse_actions = []
	maxabs_actions = []
	rmse_states = []
	maxabs_states = []

	for p in paths:
		with h5py.File(p, "r") as f:
			a = _read_action(f)
			s = _read_state(f)
		r, m, _ = _compare_traj(a, src_a)
		rmse_actions.append(r)
		maxabs_actions.append(m)
		if src_s is not None and s is not None:
			r2, m2, _ = _compare_traj(s[:, :3], src_s[:, :3])
			rmse_states.append(r2)
			maxabs_states.append(m2)

	def _summ(v: list[float]) -> str:
		if not v:
			return "n/a"
		vv = np.asarray(v, dtype=np.float64)
		return (
			f"min={np.nanmin(vv):.3g}, p50={np.nanmedian(vv):.3g}, "
			f"p95={np.nanpercentile(vv, 95):.3g}, max={np.nanmax(vv):.3g}"
		)

	print("=== Source ===")
	print(f"path: {src_path}")
	print(f"action shape: {tuple(src_a.shape)} hash(round={args.round_decimals}): {src_a_hash}")
	if src_s is None:
		print("state: NONE")
	else:
		print(f"state shape: {tuple(src_s.shape)} hash(round={args.round_decimals}): {src_s_hash}")
		if src_s.shape[1] >= 3:
			print(f"ee_start: {src_s[0, :3]}")
			print(f"ee_end:   {src_s[-1, :3]}")

	print("\n=== Synthetic episodes ===")
	print(f"dir: {synth_dir}")
	print(f"count: {len(paths)}")
	print(f"T: min={int(Ts.min())}, p50={int(np.median(Ts))}, max={int(Ts.max())}")
	print(f"action dims seen: {sorted(action_dims)}")
	print(f"state dims seen: {sorted(state_dims)}")
	print(f"unique action hashes: {uniq_action}/{len(paths)}")
	print(f"unique state hashes: {uniq_state}/{len(paths)}")
	print(f"episodes with action hash == source: {src_action_same_count}/{len(paths)}")
	if src_s is not None:
		print(f"episodes with state  hash == source: {src_state_same_count}/{len(paths)}")

	print("\nDistances to source (first min length):")
	print(f"action rmse:   {_summ(rmse_actions)}")
	print(f"action maxabs: {_summ(maxabs_actions)}")
	if rmse_states:
		print(f"state(xyz) rmse:   {_summ(rmse_states)}")
		print(f"state(xyz) maxabs: {_summ(maxabs_states)}")

	print("\nPer-episode quick view (first 10):")
	for st in stats[:10]:
		rel = os.path.relpath(st.path, synth_dir)
		print(f"- {rel}: T={st.T}, a_hash={st.action_hash[:8]}, s_hash={st.state_hash[:8]}")
		if st.ee_start is not None and st.ee_end is not None:
			print(
				f"  ee_start={np.array2string(st.ee_start, precision=4)} "
				f"ee_end={np.array2string(st.ee_end, precision=4)}"
			)


if __name__ == "__main__":
	main()


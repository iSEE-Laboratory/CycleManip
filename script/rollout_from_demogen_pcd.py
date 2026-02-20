#!/usr/bin/env python3
"""Replay DemoGen-generated pointclouds in RoboTwin to record drive_target as joint_action.

This script maps DemoGen pointclouds back to RoboTwin scene by estimating object poses
from pointcloud colors/geometry, then runs the *same controller* as RoboTwin collection
to record joint_action/vector (drive_target) during rollout.

Example:
  python script/rollout_from_demogen_pcd.py \
    --task_name beat_block_hammer_loop \
    --task_config loop1-8-all \
    --episodes_dir /home/liaohaoran/code/DemoGen/data/datasets/generated/beat_block_hammer_loop_1_test_5_episodes \
    --out_dir /home/liaohaoran/code/RoboTwin/data/beat_block_hammer_loop/loop1-8-all-pcd-replay \
    --max 5 \
    --frame 0

Use --dry_run to only print estimated object poses without running rollout.
"""

from __future__ import annotations

import argparse
import os
import json
import importlib
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import h5py
import yaml


def class_decorator(task_name: str):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def load_task_config(task_config: str) -> Dict[str, Any]:
    config_path = f"./task_config/{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.size == 0:
        return rgb
    vmax = float(np.max(rgb)) if rgb.size else 1.0
    if vmax > 1.5:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def _estimate_block_pose(pcd_xyz: np.ndarray, rgb: np.ndarray) -> Tuple[np.ndarray | None, Dict[str, Any]]:
    # Heuristic: block is red (color=(1,0,0)) in RoboTwin
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    block_mask = (r > 0.6) & (g < 0.3) & (b < 0.3)
    stats = {"block_points": int(np.sum(block_mask))}
    if stats["block_points"] < 20:
        return None, stats
    block_xyz = pcd_xyz[block_mask]
    center = np.mean(block_xyz, axis=0)
    return center, stats


def _estimate_hammer_pose(pcd_xyz: np.ndarray, rgb: np.ndarray, block_mask: np.ndarray) -> Tuple[np.ndarray | None, Dict[str, Any]]:
    # Heuristic: hammer is on negative y side and above table
    y = pcd_xyz[:, 1]
    z = pcd_xyz[:, 2]
    # exclude block points
    hammer_mask = (~block_mask) & (y < -0.02) & (z > 0.72) & (z < 0.95)
    stats = {"hammer_points": int(np.sum(hammer_mask))}
    if stats["hammer_points"] < 30:
        return None, stats
    hammer_xyz = pcd_xyz[hammer_mask]
    center = np.mean(hammer_xyz, axis=0)
    return center, stats


def estimate_object_poses_from_pcd(pcd: np.ndarray) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    pcd = np.asarray(pcd)
    if pcd.ndim != 2 or pcd.shape[1] < 6:
        raise ValueError(f"pointcloud must be (N,6+) got {pcd.shape}")
    xyz = pcd[:, :3].astype(np.float64)
    rgb = _normalize_rgb(pcd[:, 3:6])

    block_center, block_stats = _estimate_block_pose(xyz, rgb)
    if block_center is None:
        block_mask = np.zeros((xyz.shape[0],), dtype=bool)
    else:
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        block_mask = (r > 0.6) & (g < 0.3) & (b < 0.3)

    hammer_center, hammer_stats = _estimate_hammer_pose(xyz, rgb, block_mask)

    poses = {}
    if block_center is not None:
        poses["block"] = {"p": block_center.tolist(), "q": [1, 0, 0, 0]}
    if hammer_center is not None:
        # Keep default hammer orientation; only override position
        poses["hammer"] = {"p": hammer_center.tolist(), "q": [0, 0, 0.995, 0.105]}

    stats = {"block": block_stats, "hammer": hammer_stats}
    return poses, stats


def load_pointcloud_from_h5(h5_path: str, frame: int = 0) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if "pointcloud" in f:
            pcd = f["pointcloud"]
        elif "point_cloud" in f:
            pcd = f["point_cloud"]
        else:
            raise KeyError("HDF5 missing pointcloud/point_cloud")
        if pcd.ndim == 3:
            fi = min(max(int(frame), 0), pcd.shape[0] - 1)
            return np.asarray(pcd[fi])
        return np.asarray(pcd)


def run_rollout(task_name: str, task_config: str, h5_path: str, out_dir: str, ep_idx: int, frame: int, args_cfg: Dict[str, Any]):
    task = class_decorator(task_name)

    # Estimate object poses from pointcloud
    pcd = load_pointcloud_from_h5(h5_path, frame=frame)
    poses, stats = estimate_object_poses_from_pcd(pcd)

    # Prepare args for setup_demo
    args = dict(args_cfg)
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["save_path"] = out_dir
    args["save_data"] = True
    args["need_plan"] = True
    args["render_freq"] = 0
    args["now_ep_num"] = ep_idx
    args["seed"] = ep_idx
    args["override_object_poses"] = poses

    # Run rollout using the same controller
    task.setup_demo(**args)
    loop_cfg = args_cfg.get("loop", {})
    loop_times = int(loop_cfg.get("loop_times", 4))
    task.play_once(loop_times=loop_times)

    if task.plan_success and task.check_success():
        task.merge_pkl_to_hdf5_video()
        task.save_loop_times(loop_times)
        task.remove_data_cache()
    else:
        task.remove_data_cache()
    task.close_env(clear_cache=False)

    return poses, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_name", type=str, required=True)
    ap.add_argument("--task_config", type=str, required=True)
    ap.add_argument("--episodes_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--max", type=int, default=5)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    episodes_dir = os.path.expanduser(args.episodes_dir)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Load task config
    cfg = load_task_config(args.task_config)

    # Find episode files
    eps = sorted([p for p in Path(episodes_dir).glob("episode*.hdf5")])
    if not eps:
        raise FileNotFoundError(f"No episode*.hdf5 under: {episodes_dir}")

    eps = eps[: max(0, int(args.max))]
    report = {}

    for i, p in enumerate(eps):
        ep_idx = i
        if args.dry_run:
            pcd = load_pointcloud_from_h5(str(p), frame=args.frame)
            poses, stats = estimate_object_poses_from_pcd(pcd)
            report[p.name] = {"poses": poses, "stats": stats}
            print(f"{p.name}: {json.dumps(report[p.name], ensure_ascii=False)}")
            continue

        poses, stats = run_rollout(
            task_name=args.task_name,
            task_config=args.task_config,
            h5_path=str(p),
            out_dir=out_dir,
            ep_idx=ep_idx,
            frame=args.frame,
            args_cfg=cfg,
        )
        report[p.name] = {"poses": poses, "stats": stats}
        print(f"{p.name}: {json.dumps(report[p.name], ensure_ascii=False)}")

    with open(os.path.join(out_dir, "pcd_replay_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

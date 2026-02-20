"""Prepare bbhl_demogen dataset folder to match bbhl_demogen/loop1-5 structure.

This script is designed for:
  /home/liaohaoran/code/RoboTwin/data/bbhl_demogen/<setting>

It will:
  - Ensure required folders exist: data/, instructions/, instructions_sim/, instructions_full/, instructions_int/
  - Create loop_times.txt (space-separated numbers, compatible with existing RoboTwin scripts)
  - Create scene_info.json by copying a single source episode entry into N episodes
  - Generate instructions via RoboTwin description generator, then post-process into:
      * instructions      : ", loop N times" style
      * instructions_sim  : same as instructions
      * instructions_full : ", N times" style
      * instructions_int  : numeric-only style (10 seen + 10 unseen, all the same number)

Notes:
  - This script purposefully does NOT touch data/episode*.hdf5.
  - It assumes RoboTwin repo layout and is meant to be run from anywhere.

Example (your current case):
  python data/utils/prepare_bbhl_demogen_loop_dataset.py \
    --setting loop1 \
    --n 100 \
    --loop_value 1 \
    --scene_info_src /home/liaohaoran/code/RoboTwin/data/beat_block_hammer_loop/loop1-8-counter_blhl/scene_info.json \
    --scene_info_src_key episode_0 \
    --language_num 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def _repo_root_from_this_file() -> Path:
    # .../RoboTwin/data/utils/prepare_*.py -> repo root = .../RoboTwin
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_loop_times(path: Path, n: int, loop_value: int) -> None:
    # IMPORTANT: use whitespace-separated numbers to stay compatible with existing scripts
    # (some scripts use .split(" ") rather than .split()).
    content = " ".join([str(loop_value)] * n) + "\n"
    path.write_text(content, encoding="utf-8")


def _ensure_episode_files(data_dir: Path, n: int) -> None:
    missing = []
    for i in range(n):
        p = data_dir / f"episode{i}.hdf5"
        if not p.is_file():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            f"data/ 下缺少 episode 文件（预期 0..{n-1} 共 {n} 个）：\n" + "\n".join(missing[:20])
        )


def _build_scene_info(
    src_scene_info_path: Path,
    src_key: str,
    n: int,
) -> Dict[str, Any]:
    src = _load_json(src_scene_info_path)
    if src_key not in src:
        raise KeyError(f"scene_info 源文件中找不到 key: {src_key} (available sample: {list(src)[:5]})")

    entry = src[src_key]
    out: Dict[str, Any] = {}
    for i in range(n):
        out[f"episode{i}"] = entry
    return out


def _strip_trailing_punct(s: str) -> str:
    return re.sub(r"[\s\.,;:!]+$", "", s)


def _to_loop_style(texts: List[str], loop_value: int, mode: str) -> List[str]:
    """Convert base instructions into desired suffix style.

    mode:
      - "sim" : append ", loop {N} times"
      - "full": append ", {N} times"
    """
    out: List[str] = []
    for t in texts:
        t0 = _strip_trailing_punct(t)
        if mode == "sim":
            out.append(f"{t0}, loop {loop_value} times")
        elif mode == "full":
            out.append(f"{t0}, {loop_value} times")
        else:
            raise ValueError(f"unknown mode: {mode}")
    return out


def _generate_base_instructions(repo_root: Path, task_name: str, setting: str, language_num: int, out_dir: Path) -> None:
    """Run RoboTwin description generator which writes into data/<task>/<setting>/instructions."""

    # We call the python script directly to avoid dependence on bash wrappers.
    script = repo_root / "description" / "utils" / "generate_episode_instructions.py"
    if not script.is_file():
        raise FileNotFoundError(f"找不到脚本：{script}")

    # Ensure a clean base output.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["python", str(script), task_name, setting, str(language_num)]
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def _postprocess_instructions(
    base_dir: Path,
    out_sim_dir: Path,
    out_full_dir: Path,
    out_int_dir: Path,
    n: int,
    loop_value: int,
) -> None:
    out_sim_dir.mkdir(parents=True, exist_ok=True)
    out_full_dir.mkdir(parents=True, exist_ok=True)
    out_int_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        base_path = base_dir / f"episode{i}.json"
        if not base_path.is_file():
            raise FileNotFoundError(f"缺少 base instruction 文件：{base_path}")

        base = _load_json(base_path)
        seen = base.get("seen", [])
        unseen = base.get("unseen", [])
        if not isinstance(seen, list) or not isinstance(unseen, list):
            raise TypeError(f"{base_path} 格式不正确：seen/unseen 应为 list")

        sim = {
            "seen": _to_loop_style(seen, loop_value, mode="sim"),
            "unseen": _to_loop_style(unseen, loop_value, mode="sim"),
        }
        full = {
            "seen": _to_loop_style(seen, loop_value, mode="full"),
            "unseen": _to_loop_style(unseen, loop_value, mode="full"),
        }
        only_num = {
            "seen": [str(loop_value) for _ in range(10)],
            "unseen": [str(loop_value) for _ in range(10)],
        }

        # instructions (base_dir) should become sim style
        _dump_json(base_path, sim)
        _dump_json(out_sim_dir / f"episode{i}.json", sim)
        _dump_json(out_full_dir / f"episode{i}.json", full)
        _dump_json(out_int_dir / f"episode{i}.json", only_num)


def _compare_structure(dest_root: Path, template_root: Path) -> List[str]:
    """Return a list of missing top-level entries compared to template."""
    want = sorted([p.name for p in template_root.iterdir() if not p.name.endswith(".bak")])
    have = sorted([p.name for p in dest_root.iterdir()])

    missing = [x for x in want if x not in have and not x.startswith("scene_info.json.bak")]
    return missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="bbhl_demogen")
    parser.add_argument("--setting", required=True, help="e.g. loop1")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--loop_value", type=int, default=1)

    parser.add_argument(
        "--scene_info_src",
        required=True,
        help="Source scene_info.json path (we will copy one episode entry from it)",
    )
    parser.add_argument(
        "--scene_info_src_key",
        default="episode_0",
        help="Key in source scene_info.json to copy (default: episode_0)",
    )

    parser.add_argument(
        "--language_num",
        type=int,
        default=20,
        help="How many instructions per episode to generate (to match loop1-5, use 20)",
    )

    parser.add_argument(
        "--template_setting",
        default="loop1-5",
        help="Template setting name for structure check (default: loop1-5)",
    )

    args = parser.parse_args()

    repo_root = _repo_root_from_this_file()
    data_root = repo_root / "data" / args.task_name
    dest_root = data_root / args.setting

    data_dir = dest_root / "data"
    instr_dir = dest_root / "instructions"
    instr_sim_dir = dest_root / "instructions_sim"
    instr_full_dir = dest_root / "instructions_full"
    instr_int_dir = dest_root / "instructions_int"

    # 1) Basic dirs
    data_dir.mkdir(parents=True, exist_ok=True)

    # 2) Validate episode files exist
    _ensure_episode_files(data_dir, args.n)

    # 3) loop_times.txt
    _write_loop_times(dest_root / "loop_times.txt", args.n, args.loop_value)

    # 4) scene_info.json
    scene_info = _build_scene_info(Path(args.scene_info_src), args.scene_info_src_key, args.n)
    _dump_json(dest_root / "scene_info.json", scene_info)

    # 5) base instructions generation (writes into instr_dir)
    _generate_base_instructions(repo_root, args.task_name, args.setting, args.language_num, instr_dir)

    # 6) post-process into 4 instruction dirs
    _postprocess_instructions(
        base_dir=instr_dir,
        out_sim_dir=instr_sim_dir,
        out_full_dir=instr_full_dir,
        out_int_dir=instr_int_dir,
        n=args.n,
        loop_value=args.loop_value,
    )

    # 7) Structure check against template
    template_root = data_root / args.template_setting
    if template_root.exists():
        missing = _compare_structure(dest_root, template_root)
        if missing:
            raise RuntimeError(
                "目标目录缺少以下条目（相对模板 %s）：\n%s" % (args.template_setting, "\n".join(missing))
            )

    print("✅ 完成：已生成结构与文件：")
    print(f"   {dest_root}")
    print("   - loop_times.txt / scene_info.json")
    print("   - instructions / instructions_sim / instructions_full / instructions_int")


if __name__ == "__main__":
    main()

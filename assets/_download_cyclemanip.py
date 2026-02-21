"""Download and extract supplemental assets patch (moved into bootstrap directory).

This script downloads CycleManip supplemental assets and extracts them into
the CycleManip repository's `assets/` directory by default. It is placed in
`bootstrap/cyclemanip_patch/` so that patching scripts and usage docs are
co-located.

Usage (from CycleManip repo root):
  python bootstrap/cyclemanip_patch/_download_patch.py

Optional args/env are similar to the original script. Default `assets` dir
is the CycleManip repo `assets/` directory (not the script directory), so the
script behaves consistently when run from different working directories.
"""

from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download


PATCH_REPO_DEFAULT = "lemonhdl/CycleManip-assets-patch"


def _safe_tar_members(tf: tarfile.TarFile) -> list[tarfile.TarInfo]:
    safe: list[tarfile.TarInfo] = []
    for m in tf.getmembers():
        name = m.name
        if name.startswith("/") or name.startswith("\\"):
            continue
        p = Path(name)
        if any(part == ".." for part in p.parts):
            continue
        safe.append(m)
    return safe


def extract_tar(archive_path: Path, assets_dir: Path) -> Path:
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    if not tarfile.is_tarfile(archive_path):
        raise RuntimeError(f"Not a tar archive: {archive_path}")

    repo_root = assets_dir.parent

    with tarfile.open(archive_path, "r:*") as tf:
        members = _safe_tar_members(tf)
        names = [m.name for m in members if m.name]
        dst = repo_root if (names and all(n.startswith("assets/") for n in names)) else assets_dir
        tf.extractall(path=dst, members=members)
        return dst


def download_patch_archive(repo_id: str, assets_dir: Path, revision: str) -> Path:
    def _try(filename: str) -> Optional[Path]:
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=[filename],
                local_dir=str(assets_dir),
                revision=revision,
                resume_download=True,
            )
        except Exception:
            return None

        p = assets_dir / filename
        return p if p.exists() else None

    p = _try("assets.tar.gz")
    if p is not None:
        return p

    p = _try("assets")
    if p is not None:
        return p

    raise FileNotFoundError(
        f"Patch archive not found under {assets_dir}. Expected 'assets.tar.gz' or 'assets' in {repo_id}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download+extract CycleManip supplemental assets patch.")
    parser.add_argument("--repo", default=PATCH_REPO_DEFAULT, help="HF dataset repo id")
    parser.add_argument(
        "--revision",
        default=os.environ.get("HF_REVISION", "main"),
        help="HF revision (branch/tag/commit). Default: $HF_REVISION or main",
    )
    parser.add_argument(
        "--assets-dir",
        default="./assets",
        help="Target assets dir (default: CycleManip repo root /assets)",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Only download the archive and do not extract.",
    )
    args = parser.parse_args()

    # Determine CycleManip repo root relative to this script.
    script_path = Path(__file__).resolve()
    # script_path: .../CycleManip/bootstrap/cyclemanip_patch/_download_patch.py
    # repo_root is two parents up from cyclemanip_patch
    repo_root = script_path.parent.parent.parent

    assets_dir = Path(args.assets_dir).resolve() if args.assets_dir else (repo_root / "assets")
    assets_dir.mkdir(parents=True, exist_ok=True)

    print(f"[assets-patch] Repo     : {args.repo}")
    print(f"[assets-patch] Revision : {args.revision}")
    print(f"[assets-patch] Assets   : {assets_dir}")

    patch_path = download_patch_archive(args.repo, assets_dir, args.revision)
    print(f"[assets-patch] Downloaded: {patch_path}")

    if args.skip_extract:
        print("[assets-patch] --skip-extract set, done.")
        return

    dst = extract_tar(patch_path, assets_dir)
    print(f"[assets-patch] Extracted into: {dst}")

    expected = assets_dir / "objects" / "objaverse" / "list.json"
    if expected.exists():
        print(f"[assets-patch] OK: {expected}")
    else:
        print(f"[assets-patch] Warning: expected file not found: {expected}")
        print("[assets-patch] Please verify archive structure / extraction path.")


if __name__ == "__main__":
    main()

```python
#!/usr/bin/env python3
"""Apply CycleManip overlay patch onto a RoboTwin checkout.

This script pulls CycleManip content (by default from GitHub ref) and copies
selected paths into a RoboTwin checkout, overwriting files. It is intended to
be run from the target RoboTwin repo root or with --local-source pointing to a
local CycleManip checkout.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
import urllib.error
import subprocess
import shutil as _shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class CopyItem:
    src: str
    dst: str


def _now_tag() -> str:
    return time.strftime('%Y%m%d-%H%M%S')


def _load_manifest(manifest_path: Path) -> Tuple[List[CopyItem], List[str], List[str]]:
    data = json.loads(manifest_path.read_text(encoding='utf-8'))
    items = [CopyItem(**it) for it in data.get('copy', [])]
    exclude_globs = list(data.get('exclude_globs', []))
    include_override_globs = list(data.get('include_override_globs', []))
    return items, exclude_globs, include_override_globs


def _match_any(path_posix: str, globs: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path_posix, g) for g in globs)


def _should_copy(rel_posix: str, exclude_globs: List[str], include_override_globs: List[str]) -> bool:
    if _match_any(rel_posix, include_override_globs):
        return True
    if _match_any(rel_posix, exclude_globs):
        return False
    return True


def _download_tarball(owner_repo: str, ref: str, dest: Path, timeout: int = 600) -> Path:
    if '/' not in owner_repo:
        raise ValueError(f'--repo must be like "owner/repo", got: {owner_repo}')
    owner, repo = owner_repo.split('/', 1)
    tar_path = dest / f'{repo}-{ref}.tar.gz'

    token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    if token:
        api_url = f'https://api.github.com/repos/{owner}/{repo}/tarball/{ref}'

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        opener = urllib.request.build_opener(_NoRedirect)
        api_req = urllib.request.Request(
            api_url,
            headers={
                'User-Agent': 'cyclemanip-patch-applier/1.0',
                'Accept': 'application/octet-stream',
                'Authorization': f'Bearer {token}',
                'X-GitHub-Api-Version': '2022-11-28',
            },
            method='GET',
        )

        try:
            with opener.open(api_req, timeout=timeout) as resp:
                data = resp.read()
                tar_path.write_bytes(data)
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                location = e.headers.get('Location')
                if not location:
                    raise RuntimeError('GitHub API tarball redirect missing Location header')
                req = urllib.request.Request(
                    location,
                    headers={
                        'User-Agent': 'cyclemanip-patch-applier/1.0',
                        'Accept': 'application/octet-stream',
                    },
                    method='GET',
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    tar_path.write_bytes(resp.read())
            else:
                detail = ''
                try:
                    detail = e.read().decode('utf-8', errors='ignore')
                except Exception:
                    pass
                raise RuntimeError(f'GitHub API tarball download failed (HTTP {e.code}). {detail}'.strip())
    else:
        url = f'https://codeload.github.com/{owner}/{repo}/tar.gz/{ref}'
        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'cyclemanip-patch-applier/1.0',
                'Accept': 'application/octet-stream',
            },
            method='GET',
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            tar_path.write_bytes(resp.read())

    if tar_path.stat().st_size < 1024:
        raise RuntimeError(f'Downloaded tarball seems too small: {tar_path} ({tar_path.stat().st_size} bytes)')

    return tar_path


def _find_git() -> str:
    if Path('/usr/bin/git').exists():
        return '/usr/bin/git'
    which = _shutil.which('git')
    if which:
        return which
    raise RuntimeError('git executable not found (required for remote clone fallback)')


def _clone_repo(owner_repo: str, ref: str, dest_dir: Path, timeout: int = 600) -> Path:
    if '/' not in owner_repo:
        raise ValueError(f'--repo must be like "owner/repo", got: {owner_repo}')
    owner, repo = owner_repo.split('/', 1)
    url = f'https://github.com/{owner}/{repo}.git'

    git = _find_git()
    repo_dir = dest_dir / f'{repo}-{ref}-clone'

    cmd = [
        git,
        'clone',
        '--depth',
        '1',
        '--branch',
        ref,
        '--single-branch',
        url,
        str(repo_dir),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, text=True)
    return repo_dir


def _extract_tarball(tar_path: Path, dest_dir: Path) -> Path:
    with tarfile.open(tar_path, mode='r:gz') as tf:
        tf.extractall(dest_dir)

    roots = [p for p in dest_dir.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f'Unexpected tarball structure under {dest_dir}: {roots}')
    return roots[0]


def _iter_files(base: Path) -> Iterable[Path]:
    base = base.resolve()
    for root, dirs, files in os.walk(base, topdown=True, followlinks=False):
        root_p = Path(root)
        for name in files:
            p = root_p / name
            if p.is_file() or p.is_symlink():
                yield p


def _copy_file(src: Path, dst: Path, backup_root: Optional[Path]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if backup_root is not None and dst.exists() and dst.is_file():
        backup_path = backup_root / dst.relative_to(backup_root.parent)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dst, backup_path)

    shutil.copy2(src, dst)


def _copy_tree_overlay(
    src_root: Path,
    dst_root: Path,
    exclude_globs: List[str],
    include_override_globs: List[str],
    backup_root: Optional[Path],
    verbose: bool,
) -> None:
    for src_file in _iter_files(src_root):
        rel = src_file.relative_to(src_root)
        rel_posix = rel.as_posix()

        if not _should_copy(rel_posix, exclude_globs, include_override_globs):
            continue

        dst_file = dst_root / rel
        if verbose:
            print(f'[COPY] {src_file} -> {dst_file}')
        _copy_file(src_file, dst_file, backup_root)


def _ensure_export_var_in_shell_script(script_path: Path, var: str, default_value: str) -> bool:
    if not script_path.exists() or not script_path.is_file():
        return False

    try:
        text = script_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        text = script_path.read_text(encoding='latin-1')

    needle = f'export {var}='
    if needle in text:
        return False

    line = f'export {var}=${{{var}:-{default_value}}}\n'
    lines = text.splitlines(keepends=True)

    insert_at = 0
    if lines and lines[0].startswith('#!'):
        insert_at = 1

    lines.insert(insert_at, line)
    new_text = ''.join(lines)

    if new_text == text:
        return False

    script_path.write_text(new_text, encoding='utf-8')
    return True


def _patch_base_task_default_denoiser(base_task_path: Path, default_value: str = 'none') -> bool:
    if not base_task_path.exists() or not base_task_path.is_file():
        return False

    text = base_task_path.read_text(encoding='utf-8')
    old = 'os.environ.get("SAPIEN_RAY_DENOISER", "oidn")'
    new = f'os.environ.get("SAPIEN_RAY_DENOISER", "{default_value}")'
    if old not in text:
        return False
    base_task_path.write_text(text.replace(old, new, 1), encoding='utf-8')
    return True


def apply_patch(
    source_repo_root: Path,
    robottwin_root: Path,
    manifest_items: List[CopyItem],
    exclude_globs: List[str],
    include_override_globs: List[str],
    backup: bool,
    verbose: bool,
) -> None:
    if not (robottwin_root / '.git').exists():
        print(f'[WARN] Target does not look like a git repo: {robottwin_root}')

    backup_root: Optional[Path] = None
    if backup:
        backup_root = robottwin_root.parent / f'.cyclemanip_patch_backup_{_now_tag()}'
        backup_root.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f'[INFO] Backup enabled: {backup_root}')

    for item in manifest_items:
        src_path = source_repo_root / item.src
        dst_path = robottwin_root / item.dst

        if not src_path.exists():
            print(f'[WARN] Source path missing, skip: {src_path}')
            continue

        if src_path.is_dir():
            if verbose:
                print(f'[INFO] Overlay dir: {item.src} -> {item.dst}')
            dst_path.mkdir(parents=True, exist_ok=True)
            _copy_tree_overlay(src_path, dst_path, exclude_globs, include_override_globs, backup_root, verbose)
        else:
            rel_posix = Path(item.src).as_posix()
            if not _should_copy(rel_posix, exclude_globs, include_override_globs):
                if verbose:
                    print(f'[SKIP] excluded by glob: {item.src}')
                continue
            if verbose:
                print(f'[INFO] Copy file: {item.src} -> {item.dst}')
            _copy_file(src_path, dst_path, backup_root)

    modified_any = False
    modified_any |= _ensure_export_var_in_shell_script(robottwin_root / 'collect_data.sh', 'SAPIEN_RAY_DENOISER', 'none')
    modified_any |= _ensure_export_var_in_shell_script(robottwin_root / 'collect_data_loop.sh', 'SAPIEN_RAY_DENOISER', 'none')
    modified_any |= _patch_base_task_default_denoiser(robottwin_root / 'envs' / '_base_task.py', default_value='none')
    if verbose and modified_any:
        print('[INFO] Post-patch: set SAPIEN_RAY_DENOISER default to none')

    print('[DONE] Patch applied (overlay copy).')
    if backup_root is not None:
        print(f'[DONE] Backup saved to: {backup_root}')


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Apply CycleManip overlay patch onto RoboTwin (overwrite files).')
    parser.add_argument('--robottwin', type=Path, default=Path('.'), help='Path to RoboTwin repo root')

    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument('--local-source', type=Path, help='Use a local CycleManip checkout as patch source')

    parser.add_argument('--repo', type=str, default='liaohr9/CycleManip', help='GitHub repo (owner/name)')
    parser.add_argument('--ref', type=str, default='opensource', help='Git ref (branch/tag/commit)')
    parser.add_argument(
        '--github-token-env',
        type=str,
        default='GITHUB_TOKEN',
        help='Environment variable name to read GitHub token from (for private repos). Defaults to GITHUB_TOKEN.',
    )

    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('manifest.json'))
    parser.add_argument('--no-backup', action='store_true', help='Disable backup of overwritten files')
    parser.add_argument('--verbose', action='store_true', help='Print every copied file')

    args = parser.parse_args(argv)
    if args.github_token_env:
        v = os.environ.get(args.github_token_env)
        if v and not os.environ.get('GITHUB_TOKEN'):
            os.environ['GITHUB_TOKEN'] = v

    robottwin_root = args.robottwin.resolve()
    if not robottwin_root.exists():
        print(f'[ERROR] RoboTwin path not found: {robottwin_root}', file=sys.stderr)
        return 2

    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        print(f'[ERROR] Manifest not found: {manifest_path}', file=sys.stderr)
        return 2

    manifest_items, exclude_globs, include_override_globs = _load_manifest(manifest_path)

    if args.local_source is not None:
        source_repo_root = args.local_source.resolve()
        if not source_repo_root.exists():
            print(f'[ERROR] local source not found: {source_repo_root}', file=sys.stderr)
            return 2
    else:
        with tempfile.TemporaryDirectory(prefix='cyclemanip_patch_') as td:
            td_path = Path(td)
            source_repo_root_opt: Optional[Path] = None

            try:
                tar_path = _download_tarball(args.repo, args.ref, td_path)
                source_repo_root_opt = _extract_tarball(tar_path, td_path / 'src')
            except Exception as e:
                print(f'[WARN] GitHub tarball download failed: {e}', file=sys.stderr)
                try:
                    source_repo_root_opt = _clone_repo(args.repo, args.ref, td_path, timeout=600)
                except Exception as e2:
                    print(f'[ERROR] Failed to fetch from GitHub via git clone fallback: {e2}', file=sys.stderr)
                    print('[HINT] If network blocks GitHub, re-run with --local-source ../CycleManip', file=sys.stderr)
                    return 3

            if source_repo_root_opt is None or not source_repo_root_opt.exists():
                print('[ERROR] Failed to obtain source repo root', file=sys.stderr)
                return 4

            source_repo_root = source_repo_root_opt

            apply_patch(
                source_repo_root=source_repo_root,
                robottwin_root=robottwin_root,
                manifest_items=manifest_items,
                exclude_globs=exclude_globs,
                include_override_globs=include_override_globs,
                backup=(not args.no_backup),
                verbose=args.verbose,
            )
            return 0

    apply_patch(
        source_repo_root=source_repo_root,
        robottwin_root=robottwin_root,
        manifest_items=manifest_items,
        exclude_globs=exclude_globs,
        include_override_globs=include_override_globs,
        backup=(not args.no_backup),
        verbose=args.verbose,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

```

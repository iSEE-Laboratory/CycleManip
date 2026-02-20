# CycleManip → RoboTwin Overlay Patch Toolkit

This directory provides an **overlay patch toolkit** for applying CycleManip-provided code and configuration updates onto a target RoboTwin checkout. No files in the target RoboTwin directory are modified until you explicitly execute the patch script.

## Scope and behavior

The workflow is as follows:

- Retrieve CycleManip sources from a Git repository (default: `liaohr9/CycleManip` at ref `opensource`).
- Copy files listed in `manifest.json` into the target RoboTwin checkout (i.e., a direct overlay).
- Back up any replaced files to a timestamped directory located alongside the target repository:
  - `.cyclemanip_patch_backup_YYYYmmdd-HHMMSS/`
- Apply an additional runtime safeguard by disabling OIDN denoising by default (to mitigate `svulkan2` errors such as `OIDN Error: invalid handle`).

## Recommended invocation

Run the patch script from the **RoboTwin repository root** whenever possible.

### Case 1 — GitHub reachable (default)

This fetches `liaohr9/CycleManip@opensource` and applies the overlay:

```bash
bash bootstrap/cyclemanip_patch/apply_cyclemanip_patch.sh
```

### Case 2 — Private repository access (token required)

Set a token via environment variables (to avoid leaving credentials in shell history), then run:

```bash
export GITHUB_TOKEN=...  # or export GH_TOKEN=...
bash bootstrap/cyclemanip_patch/apply_cyclemanip_patch.sh --repo liaohr9/CycleManip --ref opensource
```

### Case 3 — GitHub restricted (use a local CycleManip checkout as the source)

```bash
bash bootstrap/cyclemanip_patch/apply_cyclemanip_patch.sh --local-source ../CycleManip
```

### Apply to a RoboTwin checkout outside the current working directory

Provide the target RoboTwin path as the first positional argument:

```bash
bash bootstrap/cyclemanip_patch/apply_cyclemanip_patch.sh /path/to/RoboTwin --local-source /path/to/CycleManip
```

## Invoking from a standalone CycleManip checkout

In many workflows, this toolkit is invoked from a separate CycleManip repository while the target is a different RoboTwin checkout. For convenience, `apply_cyclemanip_patch.sh` follows these conventions:

- If executed while your current working directory is the RoboTwin root (e.g., `cd /path/to/RoboTwin`), the script treats `$PWD` as the target RoboTwin directory by default.
- If you prefer to explicitly specify the target, pass the RoboTwin path as the first positional argument.

Example (from the RoboTwin repository root):

```bash
cd /home/youruser/code/RoboTwin
/home/youruser/code/CycleManip/bootstrap/cyclemanip_patch/apply_cyclemanip_patch.sh --local-source /home/youruser/code/CycleManip
```

Example (from an arbitrary location, with an explicit target):

```bash
/home/youruser/code/CycleManip/bootstrap/cyclemanip_patch/apply_cyclemanip_patch.sh /home/otheruser/code/RoboTwin --local-source /home/youruser/code/CycleManip
```

## Default overlay contents

See `manifest.json` for the authoritative list. Typical coverage includes:

- `envs/` (e.g., `envs/_base_task.py`, `envs/camera/`, and task definitions)
- `script/` (e.g., `collect_data_loop.py`)
- `task_config/` (loop-related YAML configurations)
- `description/` (instruction generation utilities)
- `data/utils/` (post-processing utilities; does not overlay existing datasets)
- Top-level entry scripts such as `collect_data.sh`, `collect_data_loop.sh`, and `delete_pth.sh`

## Exclusion rules

By default, the overlay process:

- Does **not** overlay `assets/**` or `data/**` (with an explicit exception for `data/utils/**`).
- Skips caches and build artifacts such as `__pycache__` and `*.pyc`.

If you intend to include `assets/` in the overlay, extend `manifest.json` accordingly.

## OIDN denoiser policy (disabled by default)

After applying the overlay, the toolkit enforces a conservative denoiser configuration by:

- Injecting the following at the top of `collect_data.sh` and `collect_data_loop.sh`:
  - `export SAPIEN_RAY_DENOISER=${SAPIEN_RAY_DENOISER:-none}`
- Changing the default value of `SAPIEN_RAY_DENOISER` in `envs/_base_task.py` from `oidn` to `none`.

To enable OIDN for a single run (recommended approach), set an environment variable before launching:

```bash
export SAPIEN_RAY_DENOISER=oidn
```

## Remote fetch implementation notes

The script attempts to download a tarball via `codeload.github.com` first (typically faster). If `codeload.github.com` is unavailable (e.g., returns 404 or is blocked), it falls back to `git clone --depth 1`.

In some Conda environments, `git` may lack HTTPS support (e.g., `git: 'remote-https' is not a git command`). In that case, the script will prefer the system `git` at `/usr/bin/git` as a fallback.

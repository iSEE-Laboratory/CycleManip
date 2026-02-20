推荐从 CycleManip 仓库根目录运行：

```bash
cd /path/to/CycleManip
python3 bootstrap/cyclemanip_patch/_download_patch.py --dest ./assets
```

参数与说明：

- `--repo`：HF 数据集 repo id（默认 `lemonhdl/CycleManip-assets-patch`）
- `--revision`：分支/标签/提交（默认 `main` 或环境变量 `HF_REVISION`）
- `--assets-dir`：目标 assets 目录（默认 `./assets`）
- `--skip-extract`：仅下载不解压

如果下载私有资源或遇到速率限制，请先运行 `huggingface-cli login` 并导出 `HF_TOKEN`。

<h1 align="center">CycleManip</h1>

<p align="center">
	<a href="https://isee-laboratory.github.io/CycleManip/">Project Page</a> ·
	<a href="https://arxiv.org/abs/2512.01022">Paper (arXiv)</a> ·
	<a href="https://arxiv.org/pdf/2512.01022v1">PDF</a>
</p>

## Teaser

![CycleManip teaser (Figure 1 in the paper)](fig.png)

*Teaser figure from the CycleManip paper (local copy extracted from the PDF). If the image fails to load in your Markdown viewer, use the Teaser PDF link below or the full paper PDF above.*

[Teaser PDF](fig.pdf)

## 📌 Overview

**CycleManip** studies *cycle-based manipulation*: robots must execute a cyclic/repetitive action for an **expected number of cycles** (e.g., “shake the bottle three times”) and stop at the correct moment. This setting is challenging for standard imitation policies because cyclic observations across different cycles are often visually similar, and the correct decision depends on long-range history.

The paper proposes a lightweight, end-to-end imitation framework with two core ideas:

- **Effective historical perception** via a *cost-aware sampling strategy*: sparse sampling for high-overhead observations (e.g., RGB / point clouds) and dense long-horizon sampling for low-overhead observations (e.g., proprioception / end-effector pose differences).
- **Effective historical understanding** via **multi-task learning**: jointly train the policy to predict manipulation actions and the cycle progress stage.

## Quickstart (minimal)

This quickstart is the shortest “Mode A” path (run directly in this repo). For detailed steps and troubleshooting, keep reading.

```bash
conda create -n cyclemanip python=3.10 -y
conda activate cyclemanip

git clone git@github.com:iSEE-Laboratory/CycleManip.git
cd CycleManip

bash script/_install.sh
python3 bootstrap/cyclemanip_patch/_download_patch.py --dest ./assets

# Minimal sanity run
bash collect_data_loop.sh beat_block_hammer_loop loop1-8-all 0
```

- Optional: `ffmpeg` (for video-related utilities).
- Optional (for downloading gated/private assets): Hugging Face CLI (`huggingface-cli`).


## Citation

If you use CycleManip in your research, please cite:

```
@article{wei2025cyclemanip,
	title   = {CycleManip: Enabling Cyclic Task Manipulation via Effective Historical Perception and Understanding},
	author  = {Wei, Yi-Lin and Liao, Haoran and Lin, Yuhao and Wang, Pengyue and Liang, Zhizhao and Liu, Guiliang and Zheng, Wei-Shi},
	journal = {arXiv preprint arXiv:2512.01022},
	year    = {2025}
}
```


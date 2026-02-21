<h1 align="center">CycleManip</h1>

<p align="center">
	<a href="https://isee-laboratory.github.io/CycleManip/">Project Page</a> ·
	<a href="https://arxiv.org/abs/2512.01022">Paper (arXiv)</a> ·
	<a href="https://arxiv.org/pdf/2512.01022v1">PDF</a>
</p>


![CycleManip teaser (Figure 1 in the paper)](fig.png)

## 📌 Overview

**CycleManip** studies *cycle-based manipulation*: robots must execute a cyclic/repetitive action for an **expected number of cycles** (e.g., “shake the bottle three times”) and stop at the correct moment. This setting is challenging for standard imitation policies because cyclic observations across different cycles are often visually similar, and the correct decision depends on long-range history.

The paper proposes a lightweight, end-to-end imitation framework with two core ideas:

- **Effective historical perception** via a *cost-aware sampling strategy*: sparse sampling for high-overhead observations (e.g., RGB / point clouds) and dense long-horizon sampling for low-overhead observations (e.g., proprioception / end-effector pose differences).
- **Effective historical understanding** via **multi-task learning**: jointly train the policy to predict manipulation actions and the cycle progress stage.

## INSTALL
### Environment
```bash
conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin

git clone git@github.com:iSEE-Laboratory/CycleManip.git
cd CycleManip

bash script/_install.sh
```

- Optional: `ffmpeg` (for video-related utilities).

### Download Assets

```bash
bash script/_download_assets.sh
```

### Collection Data
```bash
bash collect_data_loop.sh beat_block_hammer_loop loop1-8-all 0
```

## TODO
- [x] Release the data collection code of CycleManip
- [] Release the training code of CycleManip
- [] Release the inference and evaluation code of CycleManip


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


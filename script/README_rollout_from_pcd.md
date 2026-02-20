# DemoGen 点云回放到 RoboTwin

该脚本将 DemoGen 生成的点云映射回 RoboTwin 场景，并使用 **RoboTwin 原生控制器/规划器** 进行 rollout，将执行时的 drive_target 记录为 `joint_action/vector`。

> 说明：当前实现使用启发式点云分割（颜色 + 几何约束）估计物体位置，主要用于 `beat_block_hammer_loop`。

## 入口脚本

- `script/rollout_from_demogen_pcd.py`

## 用法

### 仅估计物体位姿（dry run）

```bash
python script/rollout_from_demogen_pcd.py \
  --task_name beat_block_hammer_loop \
  --task_config loop1-8-all \
  --episodes_dir /home/liaohaoran/code/DemoGen/data/datasets/generated/beat_block_hammer_loop_1_test_5_episodes \
  --out_dir /home/liaohaoran/code/RoboTwin/data/beat_block_hammer_loop/loop1-8-all-pcd-replay \
  --max 5 \
  --frame 0 \
  --dry_run
```

### 真正 rollout 并保存 HDF5

```bash
python script/rollout_from_demogen_pcd.py \
  --task_name beat_block_hammer_loop \
  --task_config loop1-8-all \
  --episodes_dir /home/liaohaoran/code/DemoGen/data/datasets/generated/beat_block_hammer_loop_1_test_5_episodes \
  --out_dir /home/liaohaoran/code/RoboTwin/data/beat_block_hammer_loop/loop1-8-all-pcd-replay \
  --max 5 \
  --frame 0
```

## 输出

- 回放结果保存到 `--out_dir`，并生成 `pcd_replay_report.json`

## 注意事项

- 当前点云映射为启发式估计，依赖颜色分割：红色方块、锤子在负 y 区域。
- 若场景物体颜色或分布变化较大，建议后续改为 **ICP/模板配准** 方式估计物体 pose。

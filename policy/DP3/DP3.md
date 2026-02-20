This is the README documents for dp3 training and evaluation.

1. Prepare the train & test config, for example:
```
config_name: binary_robot_dp3_obs6_multilanguage

train config: 3D-Diffusion-Policy/diffusion_policy_3d/config/${config_name}.yaml

eval train config: deploy_config/deploy_policy_${config_name}.yml
```

2. Prepare DATA, for example:
```
data/${task_name}-${task_name}-${task_config}-${expert_data_num}.zarr

data/shake_bottle_loop-loop1-8-counter-200.zarr
```

3. Train and evluation
```
bash scripts_sh/train_eval.sh ${config_name} ${task_name} ${task_config} ${expert_data_num} ${exp_tag} ${seed} ${gpu_id}

bash scripts_sh/train_eval.sh binary_robot_dp3_obs6_multilanguage shake_bottle_loop loop1-8-counter 200 sim+int_frozen_countertask 0 1
```

4. The training results are in:

experiments dir
```
experiments/${task_name}/${task_name}-${task_config}-${exp_tag}/seed_${seed}
experiments/shake_bottle_loop/shake_bottle_loop-loop1-8-counter-sim+int_frozen_countertask/seed_0
```

ckpt_path
```
${experiments dir}/${wo_colocr_pc}/${epoch}.ckpt
```

log_path
```
${experiments dir}/dp3_train.log
```

config_path
```
${experiments dir}/.hydra/config.yaml
```

5. The evaluation results are in:
```
RoboTwin/eval_result/${task_name}/${policy_name}/${exp_tag}_{epoch}/${time}/
```

6. DATA_PATH
```
/home/liaohaoran/code/RoboTwin/policy/DP3/data/beat_block_hammer_loop-loop1-8-all-200.zarr

/home/liaohaoran/code/RoboTwin/policy/DP3/data/cut_carrot_knife-loop1-8-all-200.zarr

/home/liaohaoran/code/RoboTwin/policy/DP3/data/double_knife_chop-loop1-8-all-200.zarr

/home/liaohaoran/code/RoboTwin/policy/DP3/data/grab_roller_loop-loop1-8-all-200.zarr

/home/liaohaoran/code/RoboTwin/policy/DP3/data/shake_bottle_loop-loop1-8-all-200.zarr
```


7. Add a new sample strategy
```
1. 
create new config: policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/${new_config}.yaml
create new task config: policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/${new_config}.yaml
create new deploy config: policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/deploy_config/${new_config}.yaml

2.
set dataset.sampler_strategy in task config

3. 
add the sampling code in CLASS BinarySequenceSampler in policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/common/binary_sampler.py
add the sampling code in FUCTION get_n_steps_obs in CLASS RobotRunner in policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/env_runner/robot_runner.py
```



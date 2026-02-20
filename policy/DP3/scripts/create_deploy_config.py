import os
import yaml
import argparse

def create_deploy_config(policy_name, config_name, checkpoint_num=1500):
    # 路径
    base_dir = f"policy/{policy_name}/3D-Diffusion-Policy/diffusion_policy_3d"
    src_config_path = f"{base_dir}/config/{config_name}.yaml"
    dst_config_path = f"{base_dir}/deploy_config/deploy_policy_{config_name}.yml"

    # 读取源配置以获取 task_name
    with open(src_config_path, "r") as f:
        src_cfg = yaml.safe_load(f)

    # 从 defaults 中提取 task_name
    task_name = None
    if "defaults" in src_cfg:
        for item in src_cfg["defaults"]:
            if isinstance(item, dict) and "task" in item:
                task_name = item["task"]
                break

    if task_name is None:
        raise ValueError(f"task_name not found in defaults of {src_config_path}")

    # 构建 deploy 配置内容
    deploy_cfg = {
        "policy_name": None,
        "task_name": None,
        "exp_name": None,
        "task_config": None,
        "ckpt_setting": None,
        "seed": None,
        "instruction_type": "unseen",

        "config_name": config_name,
        "checkpoint_num": checkpoint_num,
        "dp3_task": task_name,
        "expert_data_num": None
    }

    # 确保目录存在
    os.makedirs(os.path.dirname(dst_config_path), exist_ok=True)

    # 写入 yml
    with open(dst_config_path, "w") as f:
        yaml.dump(deploy_cfg, f, sort_keys=False)

    print(f"✅ Deploy config generated:\n{dst_config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", type=str, required=True, help="name of the policy folder")
    parser.add_argument("--config_name", type=str, required=True, help="name of config yaml (without .yaml)")
    parser.add_argument("--checkpoint_num", type=int, default=1500, help="checkpoint number")
    args = parser.parse_args()

    create_deploy_config(args.policy_name, args.config_name, args.checkpoint_num)

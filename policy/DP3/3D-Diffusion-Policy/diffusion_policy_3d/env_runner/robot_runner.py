import wandb
import numpy as np
import torch
import tqdm

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from diffusion_policy_3d.common.sampler_strategy import past_k_sample_indices, past_k_sample_step_indices, adaptive_joint_sample_indices, long_short_sample_indices, adaptive_fixthre_endpose_sample_indices

from termcolor import cprint
import pdb
from queue import deque

import time
class RobotRunner(BaseRunner):

    def __init__(
        self,
        output_dir=None,
        eval_episodes=20,
        max_steps=200,
        n_obs_steps=8,
        n_action_steps=8,
        fps=10,
        crf=22,
        render_size=84,
        tqdm_interval_sec=5.0,
        task_name=None,
        use_point_crop=True,
        sampler_strategy=None,
    ):
        super().__init__(output_dir)
        self.task_name = task_name

        steps_per_render = max(10 // fps, 1)

        self.eval_episodes = eval_episodes
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        # self.obs = deque(maxlen=n_obs_steps + 1
        self.obs = []
        self.env = None
        self.sampler_strategy = sampler_strategy

    def stack_last_n_obs(self, all_obs, n_steps):
        assert len(all_obs) > 0
        all_obs = list(all_obs)
        if isinstance(all_obs[0], np.ndarray):
            result = np.zeros((n_steps, ) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = np.array(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        elif isinstance(all_obs[0], torch.Tensor):
            result = torch.zeros((n_steps, ) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = torch.stack(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        else:
            raise RuntimeError(f"Unsupported obs type {type(all_obs[0])}")
        return result

    def reset_obs(self):
        self.obs.clear()

    def update_obs(self, current_obs):
        self.obs.append(current_obs)

    def get_n_steps_obs(self):
        assert len(self.obs) > 0, "no observation is recorded, please update obs first"

        result = dict()
        for key in self.obs[0].keys():
            result[key] = self.stack_last_n_obs([obs[key] for obs in self.obs], self.n_obs_steps)

        return result
    
    def get_n_steps_obs_binary(self):

        assert len(self.obs) > 0, "no observation is recorded, please update obs first"
        
        result = dict()

        history_state = np.stack([obs["agent_pos"] for obs in self.obs])
        short_state_indices = None
        if "endpose" in self.obs[0]:
            history_endpose = np.stack([obs["endpose"] for obs in self.obs])
        else:
            history_endpose = None

        if self.sampler_strategy == "past":
            input_indices = None
            pc_indices = past_k_sample_indices(0, len(self.obs), self.n_obs_steps)
            joint_indices = pc_indices

        elif self.sampler_strategy == "past_all":
            input_indices = None
            pc_indices = past_k_sample_indices(0, len(self.obs), self.n_obs_steps)
            joint_indices = range(0, len(self.obs))

        elif self.sampler_strategy == "binary":
            input_indices = None
            pc_indices = long_short_sample_indices(0, len(self.obs), self.n_obs_steps)
            joint_indices = range(0, len(self.obs))
            
        elif self.sampler_strategy == "joint_adaptive-pc_adaptive_binary":
            input_indices = adaptive_joint_sample_indices(0, len(self.obs), history_state)
            pc_indices = long_short_sample_indices(0, len(input_indices), self.n_obs_steps)
            joint_indices = range(0, len(input_indices))
 
        elif self.sampler_strategy == "joint_adaptive-pc_binary":
            input_indices = None
            joint_indices = adaptive_joint_sample_indices(0, len(self.obs), history_state)
            pc_indices = long_short_sample_indices(0, len(self.obs), self.n_obs_steps)

        elif self.sampler_strategy == "joint_adaptive_fixthre_endpose-pc_adaptive_binary":
            assert history_endpose is not None
            input_indices = adaptive_fixthre_endpose_sample_indices(0, len(self.obs), history_endpose)
            pc_indices = long_short_sample_indices(0, len(input_indices), self.n_obs_steps)
            joint_indices = range(0, len(input_indices))

        elif self.sampler_strategy == "joint_adaptive-pc_past":
            input_indices = None
            joint_indices = adaptive_joint_sample_indices(0, len(self.obs), history_state)
            pc_indices = past_k_sample_indices(0, len(self.obs), self.n_obs_steps)

        elif self.sampler_strategy == "joint_all_pc_binary_short_state":
            input_indices = None
            pc_indices = long_short_sample_indices(0, len(self.obs), self.n_obs_steps)
            joint_indices = range(0, len(self.obs))
            short_state_indices = past_k_sample_indices(0, len(self.obs), self.n_obs_steps)

        elif self.sampler_strategy == "adaptive_joint_all_pc_binary_short_state":
            input_indices = adaptive_fixthre_endpose_sample_indices(0, len(self.obs), history_state)
            pc_indices = long_short_sample_indices(0, len(input_indices), self.n_obs_steps)
            joint_indices = range(0, len(input_indices))
            short_state_indices = past_k_sample_indices(0, len(self.obs), self.n_obs_steps)
        elif self.sampler_strategy == "pc_uniform":
            input_indices = None
            pc_indices = np.linspace(0, len(self.obs)-1, self.n_obs_steps, dtype=np.int64)
            joint_indices = range(0, len(self.obs))
        elif self.sampler_strategy == "pc_past_k_step":
            input_indices = None
            pc_indices = past_k_sample_step_indices(0, len(self.obs), self.n_obs_steps, 4)
            joint_indices = range(0, len(self.obs))
        # elif self.sampler_strategy == "joint_fixextra-pc_binary":
        #     input_indices = None
        #     input_indices = adaptive_fixthre_endpose_sample_indices(0, len(self.obs), history_endpose)
        #     pc_indices = long_short_sample_indices(0, len(input_indices), self.n_obs_steps)
        #     joint_indices = range(0, len(input_indices))
            
        else:
            raise Exception(f"{self.sampler_strategy}")

        if input_indices is not None:
            cur_obs = [self.obs[i] for i in input_indices]
        else:
            cur_obs = [obs for obs in self.obs]

        for key in self.obs[0].keys():
            if isinstance(self.obs[0][key], str):
                sampled_obs = [self.obs[-1][key]]
            else:
                if key in ['agent_pos', 'endpose']:
                    sampled_obs = [cur_obs[i][key] for i in joint_indices]
                else:
                    sampled_obs = [cur_obs[i][key] for i in pc_indices]

            if isinstance(sampled_obs[0], np.ndarray):
                try:
                    result[key] = np.stack(sampled_obs)
                except:
                    print(key, len(sampled_obs))
                    for xxx in sampled_obs:
                        print(xxx.shape)
                    print(sampled_obs)
            elif isinstance(sampled_obs[0], torch.Tensor):
                try:
                    result[key] = torch.stack(sampled_obs)
                except:
                    print(key, len(sampled_obs))
                    for xxx in sampled_obs:
                        print(xxx.shape)
                    print(sampled_obs)
            else:
                result[key] = sampled_obs

        if short_state_indices is not None:
            sampled_obs = [cur_obs[i]['agent_pos'] for i in short_state_indices]
            if isinstance(sampled_obs[0], np.ndarray):
                result["short_state"] = np.stack(sampled_obs)
            elif isinstance(sampled_obs[0], torch.Tensor):
                result["short_state"] = torch.stack(sampled_obs)
            else:
                raise Exception(f"Unsupported data type {type(sampled_obs[0])} for key {key}")

        return result

    def get_action(self, policy: BasePolicy, observaton=None) -> bool:
        device, dtype = policy.device, policy.dtype
        if observaton is not None:
            self.obs.append(observaton)  # update
        # obs = self.get_n_steps_obs() # 这里就是获取了n_steps的obs
        # print(f"obs length: {len(self.obs)}")

        obs = self.get_n_steps_obs_binary() 

        # create obs dict
        np_obs_dict = dict(obs)

        np_obs_dict = {}
        obs_dict_input = {}  # flush unused keys
        # 遍历原字典，筛选值为list类型的键值对
        for key, value in obs.items():
            if isinstance(value, list):
                obs_dict_input[key] = value
            else:
                np_obs_dict[key] = value

        # device transfer
        obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
        # run policy
        with torch.no_grad():
            obs_dict_input["point_cloud"] = obs_dict["point_cloud"].unsqueeze(0)
            obs_dict_input["agent_pos"] = [obs_dict["agent_pos"]]
            if "endpose" in obs_dict:
                obs_dict_input["endpose"] = [obs_dict["endpose"]]
            if "short_state" in obs_dict:
                obs_dict_input["short_state"] = obs_dict["short_state"].unsqueeze(0)
            action_dict = policy.predict_action(obs_dict_input)
        # device_transfer
        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to("cpu").numpy())
        action = np_action_dict["action"].squeeze(0)
        return action

    def run(self, policy: BasePolicy):
        pass


if __name__ == "__main__":
    test = RobotRunner("./")
    print("ready")

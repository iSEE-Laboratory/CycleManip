from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import time
import pdb

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.model.vision.pointnet_extractor import DP3Encoder, create_mlp
from diffusion_policy_3d.model.language import LanguageEncoderFinetune, LanguageEncoderFrozen, LanguageEncoderMulti, LanguageEncoderOnehot

class BinaryDP3(BasePolicy):
    """
    二分观测采样的DP3模型
    - 观测：二分采样的长历史序列
    - 动作：未来的连续动作序列
    """

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        action_horizon,
        obs_horizon,
        n_action_steps,
        action_task_horizon=0,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        condition_type="film",
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        encoder_output_dim=256,
        crop_shape=None,
        use_pc_color=False,
        pointnet_type="pointnet",
        pointcloud_encoder_cfg=None,
        multi_task_times=False,
        multi_task_counter=False,
        multi_selftask=False,
        multi_progresstask=False,
        multi_selfjointtask=False,
        language_encoder_type="Frozen",
        language_keys=None,
        new_multitask=False,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        self.condition_type = condition_type

        self.multi_task_times = multi_task_times
        self.multi_task_counter = multi_task_counter
        self.multi_selftask = multi_selftask
        self.multi_progresstask = multi_progresstask
        self.new_multitask = new_multitask
        self.multi_selfjointtask = multi_selfjointtask

        # parse shape_meta
        action_shape = shape_meta["action"]["shape"]
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:  # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_shape_meta = shape_meta["obs"]
        obs_dict = dict_apply(obs_shape_meta, lambda x: x["shape"])

        obs_encoder = DP3Encoder(
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
            obs_horizon=obs_horizon
        )
        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
    

        language_encoder_classes = {
            "LanguageEncoderFinetune": LanguageEncoderFinetune,
            "LanguageEncoderFrozen": LanguageEncoderFrozen,
            "LanguageEncoderMulti": LanguageEncoderMulti,
            "LanguageEncoderOnehot": LanguageEncoderOnehot
        }
        if language_encoder_type in language_encoder_classes:

            language_encoder = language_encoder_classes[language_encoder_type](language_embed_dim=512, output_projection_dim=obs_feature_dim, language_keys=language_keys)
            language_feature_dim = language_encoder.output_shape()
        else:
            language_encoder = None
            language_feature_dim = 0

        # 关键修改：输入维度只包含动作，观测完全作为条件
        input_dim = action_dim  # 只对动作进行扩散
        global_cond_dim = None
        
        if obs_as_global_cond:
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim  # 序列形式的条件
            else:
                if language_encoder_type in language_encoder_classes:
                    global_cond_dim = obs_feature_dim * (obs_horizon + 1)  # 展平的条件
                else:
                    global_cond_dim = obs_feature_dim * obs_horizon

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(
            f"[BinaryDP3] use_pc_color: {self.use_pc_color}",
            "yellow",
        )
        cprint(
            f"[BinaryDP3] pointnet_type: {self.pointnet_type}",
            "yellow",
        )
        cprint(
            f"[BinaryDP3] action_horizon: {action_horizon}, obs_horizon: {obs_horizon}",
            "yellow",
        )

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.language_encoder = language_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler

        if self.new_multitask:
            pc_joint_dim = obs_feature_dim * obs_horizon

            if self.multi_task_counter or self.multi_selftask:
                self.pc_joint_fusion = nn.Sequential(*create_mlp(pc_joint_dim, pc_joint_dim, [pc_joint_dim//2], activation_fn=nn.ReLU))
            else:
                self.pc_joint_fusion = None
                
            if self.multi_progresstask:
                self.pc_joint_lan_fusion = nn.Sequential(*create_mlp(global_cond_dim, global_cond_dim, [global_cond_dim//2], activation_fn=nn.ReLU))
            
            if self.multi_selfjointtask:
                self.joint_mlp = nn.Sequential(*create_mlp(global_cond_dim, global_cond_dim, [global_cond_dim//2], activation_fn=nn.ReLU))

            if self.multi_task_times:
                self.times_decoder = nn.Sequential(*create_mlp(language_feature_dim, 10, [language_feature_dim//2], activation_fn=nn.ReLU))
                self.times_criterion = nn.CrossEntropyLoss()

            if self.multi_task_counter:
                self.counter_decoder = nn.Sequential(*create_mlp(pc_joint_dim, 10, [], activation_fn=nn.ReLU))
                self.counter_criterion = nn.CrossEntropyLoss()
                
            if self.multi_selftask:
                self.selfcounter_decoder = nn.Sequential(*create_mlp(pc_joint_dim, 15, [], activation_fn=nn.ReLU))
                self.selfcounter_criterion = nn.CrossEntropyLoss()

            if self.multi_progresstask:
                self.progresscounter_decoder = nn.Sequential(*create_mlp(global_cond_dim, 10, [], activation_fn=nn.ReLU))
                self.progresscounter_criterion = nn.CrossEntropyLoss()

        else:
            if self.multi_task_times:
                self.times_decoder = nn.Sequential(*create_mlp(language_feature_dim, 10, [language_feature_dim//2], activation_fn=nn.ReLU))
                self.times_criterion = nn.CrossEntropyLoss()

            if self.multi_task_counter:
                self.counter_decoder = nn.Sequential(*create_mlp(obs_feature_dim, 10, [obs_feature_dim//2], activation_fn=nn.ReLU))
                self.counter_criterion = nn.CrossEntropyLoss()
                
            if self.multi_selftask:
                self.selfcounter_decoder = nn.Sequential(*create_mlp(obs_feature_dim, 15, [obs_feature_dim//2], activation_fn=nn.ReLU))
                self.selfcounter_criterion = nn.CrossEntropyLoss()

        if not hasattr(self, 'pc_joint_fusion'):
            self.pc_joint_fusion = None

        if not hasattr(self, 'pc_joint_lan_fusion'):
            self.pc_joint_lan_fusion = None

        # 不需要mask_generator，因为不对观测进行扩散
        self.normalizer = LinearNormalizer()
        self.action_horizon = action_horizon + action_task_horizon
        self.action_task_horizon = action_task_horizon
        self.obs_horizon = obs_horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print_params(self)

    # ========= inference  ============
    def conditional_sample(
        self,
        global_cond=None,
        local_cond=None,
        generator=None,
        **kwargs,
    ):
        model = self.model
        scheduler = self.noise_scheduler

        # 初始化纯噪声作为动作序列
        trajectory = torch.randn(
            size=(global_cond.shape[0], self.action_horizon, self.action_dim),
            dtype=global_cond.dtype,
            device=global_cond.device,
            generator=generator
        )

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            model_output = model(
                sample=trajectory,
                timestep=t,
                local_cond=local_cond,
                global_cond=global_cond,
            )

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output,
                t,
                trajectory,
            ).prev_sample

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # x = time.time()
        obs_dict_for_norm = {k: v for k, v in obs_dict.items() if k in self.normalizer.params_dict}

        nobs = self.normalizer.normalize(obs_dict_for_norm)

        for k, v in obs_dict.items():
            if k not in self.normalizer.params_dict:
                nobs[k] = v
        if not self.use_pc_color and "point_cloud" in nobs:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]

        batch_size = len(nobs["agent_pos"])
        
        nobs_for_reshape = {k: v for k, v in nobs.items() if "instruction" not in k}
        this_nobs = dict_apply(nobs_for_reshape, lambda x: x.reshape(-1, *x.shape[2:]))

        for key in nobs:
            if "instruction" in key:
                this_nobs[key] = nobs[key]

        nobs_features = self.obs_encoder(this_nobs)

        if self.language_encoder is not None:
            language_features = self.language_encoder(this_nobs)   

        if "cross_attention" in self.condition_type:
            # 保持序列形式：[batch_size, obs_horizon, feature_dim]
            # global_cond = nobs_features.reshape(batch_size, self.obs_horizon, -1)
            # if self.language_encoder is not None:
            #     global_cond = torch.cat([global_cond, language_features.unsqueeze(1)], dim=1)
            raise Exception("1")
        else:
            # 展平：[batch_size, obs_horizon * feature_dim]
            pc_joint_cond = nobs_features.reshape(batch_size, -1)
            if self.pc_joint_fusion is not None:
                pc_joint_cond = self.pc_joint_fusion(pc_joint_cond)
            if self.language_encoder is not None:
                global_cond = torch.cat([pc_joint_cond, language_features], dim=-1)
                if self.pc_joint_lan_fusion is not None:
                    global_cond = self.pc_joint_lan_fusion(global_cond)
            else:
                global_cond = pc_joint_cond

            if self.new_multitask:
                if self.multi_selftask:
                    selfcounter_from_now_pred = self.selfcounter_decoder(pc_joint_cond)
                    pred_classes = torch.argmax(selfcounter_from_now_pred, dim=1) 
                    print(pred_classes.cpu().numpy())

        # print("global_cond.shape", global_cond.shape)
        # 条件采样生成动作
        nsample = self.conditional_sample(
            global_cond=global_cond,
            **self.kwargs,
        )
        # print("global_cond.shape", global_cond.shape)
        # print("diffusion", time.time()-x)
        # x = time.time()
        # unnormalize prediction
        action_pred = self.normalizer["action"].unnormalize(nsample)

        # 执行前n_action_steps帧动作
        start = 0
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {
            "action": action,
            "action_pred": action_pred,
        }

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        """
        计算二分采样数据的损失
        batch["obs"]: [B, obs_horizon, ...] - 二分采样的观测
        batch["action"]: [B, action_horizon, action_dim] - 未来的动作序列
        """
        # x = time.time()
        
        # normalize input
        obs_dict = batch["obs"]
        obs_dict_for_norm = {k: v for k, v in obs_dict.items() if k in self.normalizer.params_dict}

        # 执行标准化
        nobs = self.normalizer.normalize(obs_dict_for_norm)

        # 将不在params_dict中的数据直接添加到nobs中
        for k, v in obs_dict.items():
            if k not in self.normalizer.params_dict:
                nobs[k] = v

        nactions = self.normalizer["action"].normalize(batch["action"])

        if not self.use_pc_color and "point_cloud" in nobs:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]

        batch_size = nactions.shape[0]
        action_horizon = nactions.shape[1]

        # 处理观测条件
        if self.obs_as_global_cond:
            if isinstance(nobs["agent_pos"], torch.Tensor):
                nobs_for_reshape = {k: v for k, v in nobs.items() if "instruction" not in k and "agent_pos" not in k and "endpose" not in k}
                this_nobs = dict_apply(nobs_for_reshape, lambda x: x.reshape(-1, *x.shape[2:]))
                this_nobs["agent_pos"] = nobs["agent_pos"]
                if "endpose" in nobs:
                    this_nobs["endpose"] = nobs["endpose"]
            else:
                nobs_for_reshape = {k: v for k, v in nobs.items() if "instruction" not in k}
                this_nobs = dict_apply(nobs_for_reshape, lambda x: x.reshape(-1, *x.shape[2:]))

            for key in nobs:
                if "instruction" in key:
                    this_nobs[key] = nobs[key]

            nobs_features = self.obs_encoder(this_nobs)
            if self.language_encoder is not None:
                language_features = self.language_encoder(this_nobs)   

            if "cross_attention" in self.condition_type:
                # 保持序列形式：[batch_size, obs_horizon, feature_dim]
                # global_cond = nobs_features.reshape(batch_size, self.obs_horizon, -1)
                # if self.language_encoder is not None:
                #     global_cond = torch.cat([global_cond, language_features.unsqueeze(1)], dim=1)
                raise Exception("1")
            else:
                # 展平：[batch_size, obs_horizon * feature_dim]
                pc_joint_cond = nobs_features.reshape(batch_size, -1)
                if self.pc_joint_fusion is not None:
                    pc_joint_cond = self.pc_joint_fusion(pc_joint_cond)
                if self.language_encoder is not None:
                    global_cond = torch.cat([pc_joint_cond, language_features], dim=-1)
                    if self.pc_joint_lan_fusion is not None:
                        global_cond = self.pc_joint_lan_fusion(global_cond)
                else:
                    global_cond = pc_joint_cond
                    
        else:
            raise NotImplementedError("只支持obs_as_global_cond=True模式")

        # 关键修改：只对动作序列进行扩散，无需mask
        trajectory = nactions  # [B, action_horizon, action_dim]

        # Sample noise that we'll add to the actions
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        bsz = trajectory.shape[0]
        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()

        # Add noise to the clean actions
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # Predict the noise residual or clean action
        pred = self.model(
            sample=noisy_trajectory,
            timestep=timesteps,
            local_cond=None,
            global_cond=global_cond,
        )
        # print("diffusion", time.time()-x)
        # x = time.time()
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        elif pred_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(trajectory, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')

        # 创建一个权重矩阵，后半部分 (B, 5) 的权重是 0.1，其余部分是 1

        if self.action_task_horizon != 0:
            weights = torch.ones_like(pred).to(pred.device)  
            weights[:, -self.action_task_horizon:, :] = 0.5  
            loss = loss * weights

        loss = loss.mean()

        # loss = F.mse_loss(pred, target, reduction="mean")
        
        loss_dict = {
            "mse_loss": loss.item(),
        }

        if self.new_multitask:
            if self.multi_task_times:
                times_from_language_pred = self.times_decoder(language_features)
                gt = obs_dict["loop_times"]
                time_loss = self.times_criterion(times_from_language_pred, gt) * 0.1
                loss_dict["times_loss"] = time_loss.item()
                loss += time_loss

            if self.multi_task_counter:
                counter_from_now_pred = self.counter_decoder(pc_joint_cond)
                gt = obs_dict["loop_counter"]
                counter_loss = self.counter_criterion(counter_from_now_pred, gt) * 0.1
                loss_dict["counter_loss"] = counter_loss.item()
                loss += counter_loss

            if self.multi_selftask:
                selfcounter_from_now_pred = self.selfcounter_decoder(pc_joint_cond)
                gt = obs_dict["loop_curlen"] // 20
                gt = torch.clamp(gt, min=0, max=14)
                selfcounter_loss = self.selfcounter_criterion(selfcounter_from_now_pred, gt) * 0.1
                loss_dict["selfcounter_loss"] = selfcounter_loss.item()
                loss += selfcounter_loss

            if self.multi_progresstask:
                progresscounter_from_now_pred = self.progresscounter_decoder(global_cond)
                gt = (10 * obs_dict["loop_curlen"]) // obs_dict["loop_length"]
                gt = torch.clamp(gt, min=0, max=9)
                progresscounter_loss = self.progresscounter_criterion(progresscounter_from_now_pred, gt) * 0.1
                loss_dict["progresscounter_loss"] = progresscounter_loss.item()
                loss += progresscounter_loss
        else:
            if self.multi_task_times:
                times_from_language_pred = self.times_decoder(language_features)
                gt = obs_dict["loop_times"]
                time_loss = self.times_criterion(times_from_language_pred, gt) * 0.1
                loss_dict["times_loss"] = time_loss.item()
                loss += time_loss

            if self.multi_task_counter:
                counter_from_now_pred = self.counter_decoder(torch.mean(nobs_features.reshape(batch_size, self.obs_horizon, -1), dim=1))
                gt = obs_dict["loop_counter"]
                counter_loss = self.counter_criterion(counter_from_now_pred, gt) * 0.1
                loss_dict["counter_loss"] = counter_loss.item()
                loss += counter_loss

            if self.multi_selftask:
                selfcounter_from_now_pred = self.selfcounter_decoder(torch.mean(nobs_features.reshape(batch_size, self.obs_horizon, -1), dim=1))
                gt = obs_dict["loop_curlen"] // 20
                gt = torch.clamp(gt, min=0, max=14)
                selfcounter_loss = self.selfcounter_criterion(selfcounter_from_now_pred, gt) * 0.1
                loss_dict["selfcounter_loss"] = selfcounter_loss.item()
                loss += selfcounter_loss


        loss_dict["loss"] = loss.item()

        return loss, loss_dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
import clip

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
import pdb

from diffusion_policy_3d.model.vision.state_mamba import StateFusionMamba
from diffusion_policy_3d.model.vision.state_attention import StateFusionAttention

def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1024,
        use_layernorm: bool = False,
        final_norm: str = "none",
        use_projection: bool = True,
        **kwargs,
    ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), "cyan")
        cprint("pointnet use_final_norm: {}".format(final_norm), "cyan")

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(nn.Linear(block_channel[-1], out_channels),
                                                  nn.LayerNorm(out_channels))
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x


class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud"""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1024,
        use_layernorm: bool = False,
        final_norm: str = "none",
        use_projection: bool = True,
        **kwargs,
    ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), "cyan")
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), "cyan")

        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(nn.Linear(block_channel[-1], out_channels),
                                                  nn.LayerNorm(out_channels))
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")

        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)

    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x

    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()

    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()


class DP3EncoderPlus(nn.Module):

    def __init__(
        self,
        observation_space: Dict,
        img_crop_shape=None,
        out_channel=256,
        state_mlp_size=(64, 64),
        state_mlp_activation_fn=nn.ReLU,
        pointcloud_encoder_cfg=None,
        use_pc_color=False,
        pointnet_type="pointnet",
        obs_horizon=None,
    ):
        super().__init__()
        self.imagination_key = "imagin_robot"
        self.state_key = getattr(pointcloud_encoder_cfg, "state_key", "agent_pos")
        self.short_state_key = "short_state"
        self.point_cloud_key = "point_cloud"
        self.rgb_image_key = "image"
        self.language_key = "instruction"
        self.n_output_channels = out_channel

        use_diff = getattr(pointcloud_encoder_cfg, "use_diff", False)
        posi_type = getattr(pointcloud_encoder_cfg, "posi_type", "learning")
        flip = getattr(pointcloud_encoder_cfg, "flip", True)

        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]

        self.short_state_shape = observation_space["agent_pos"]

        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None

        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        elif pointnet_type == "no_pc":
            self.extractor = None
            self.n_output_channels = 0 
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]       

        if self.pointnet_type == "no_pc":
            self.n_output_channels = 0
        else:
            self.n_output_channels = out_channel * 6

        if pointcloud_encoder_cfg.state_encode_type == "all_joint_attention_bicls":
            self.state_dim = output_dim*4
            self.short_joint_dim = output_dim
            self.n_output_channels += output_dim * (2*2+1)
            self.state_attention = StateFusionAttention(d_in=self.state_shape[0], d_out=output_dim*2, d_model=output_dim, use_final_cls=True, use_diff=use_diff, posi_type=posi_type, flip=flip)
            self.state_mlp = nn.Sequential(*create_mlp(self.short_state_shape[0], output_dim, net_arch, state_mlp_activation_fn))            
        elif pointcloud_encoder_cfg.state_encode_type == "all_joint_attention":
            self.state_dim = output_dim*2
            self.short_joint_dim = output_dim 
            self.n_output_channels += output_dim * (2*1+1)
            self.state_attention = StateFusionAttention(d_in=self.state_shape[0], d_out=output_dim*2, d_model=output_dim, use_diff=use_diff, posi_type=posi_type, flip=flip)
            self.state_mlp = nn.Sequential(*create_mlp(self.short_state_shape[0], output_dim, net_arch, state_mlp_activation_fn))            
        elif pointcloud_encoder_cfg.state_encode_type == "past":
            self.state_dim = output_dim*2
            self.short_joint_dim = output_dim 
            self.n_output_channels += output_dim * 1
            self.state_mlp = nn.Sequential(*create_mlp(self.short_state_shape[0], output_dim, net_arch, state_mlp_activation_fn)) 
        else:
            raise Exception("1")

        self.state_encode_type = pointcloud_encoder_cfg.state_encode_type

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")

    def forward(self, observations: Dict) -> torch.Tensor:

        state = observations[self.state_key]
        short_state = observations[self.short_state_key]

        batch_size = len(state)

        if self.pointnet_type != "no_pc":
            points = observations[self.point_cloud_key]
            assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
            if self.use_imagined_robot:
                img_points = observations[self.imagination_key][..., :points.shape[-1]]  # align the last dim
                points = torch.concat([points, img_points], dim=1)
            pn_feat = self.extractor(points)  # (B*L) * out_channel
            pn_feat = pn_feat.reshape(batch_size, -1)
        else:
            pn_feat = None


        if self.state_encode_type == "all_joint_attention":
            if isinstance(short_state, List):
                # ONLY FOR INFERENCE (BATCH WILL STACK AS HORIZEN IS SAME IN TRAINING)
                short_state = torch.cat(short_state, dim=0)
            elif short_state.ndim == 3:
                short_state = short_state.reshape(-1, *short_state.shape[2:])
                
            short_state_feat = self.state_mlp(short_state)
            short_state_feat = short_state_feat.reshape(batch_size, -1, short_state_feat.shape[-1])
            max_val, _ = torch.max(short_state_feat, dim=1) 
            # 计算平均值
            mean_val = torch.mean(short_state_feat, dim=1)  

            # 将最大值和平均值相加
            short_state_feat = max_val + mean_val
            state_cls_feat = self.state_attention(state)
            state_feat = state_cls_feat.reshape(batch_size, -1)
            if self.pointnet_type != "no_pc":
                features_to_concat = [pn_feat, state_feat, short_state_feat]  
            else:
                features_to_concat = [state_feat, short_state_feat]  
            final_feat = torch.cat(features_to_concat, dim=-1)
        elif self.state_encode_type == "all_joint_attention_bicls":
            if isinstance(short_state, List):
                # ONLY FOR INFERENCE (BATCH WILL STACK AS HORIZEN IS SAME IN TRAINING)
                short_state = torch.cat(short_state, dim=0)
            elif short_state.ndim == 3:
                short_state = short_state.reshape(-1, *short_state.shape[2:])
                
            short_state_feat = self.state_mlp(short_state)
            short_state_feat = short_state_feat.reshape(batch_size, -1, short_state_feat.shape[-1])

            max_val, _ = torch.max(short_state_feat, dim=1) 
            # 计算平均值
            mean_val = torch.mean(short_state_feat, dim=1)  

            # 将最大值和平均值相加
            short_state_feat = max_val + mean_val
            state_cls_feat, _, state_cls_feat2 = self.state_attention(state, verbos=True)
            state_feat = torch.cat([state_cls_feat, state_cls_feat2], dim=-1).reshape(batch_size, -1)

            if self.pointnet_type != "no_pc":
                features_to_concat = [pn_feat, state_feat, short_state_feat]  
            else:
                features_to_concat = [state_feat, short_state_feat] 
            final_feat = torch.cat(features_to_concat, dim=-1)
        elif self.state_encode_type == "past":
            if isinstance(short_state, List):
                # ONLY FOR INFERENCE (BATCH WILL STACK AS HORIZEN IS SAME IN TRAINING)
                short_state = torch.cat(short_state, dim=0)
            elif short_state.ndim == 3:
                short_state = short_state.reshape(-1, *short_state.shape[2:])
                
            short_state_feat = self.state_mlp(short_state)
            short_state_feat = short_state_feat.reshape(batch_size, -1, short_state_feat.shape[-1])

            max_val, _ = torch.max(short_state_feat, dim=1) 
            # 计算平均值
            mean_val = torch.mean(short_state_feat, dim=1)  

            # 将最大值和平均值相加
            short_state_feat = max_val + mean_val
            if self.pointnet_type != "no_pc":
                features_to_concat = [pn_feat, short_state_feat]  
            else:
                features_to_concat = [short_state_feat] 
            final_feat = torch.cat(features_to_concat, dim=-1)
            state_feat = None

        else:
            raise Exception("1")
        
        return final_feat, pn_feat, short_state_feat, state_feat


    def output_shape(self):
        return self.n_output_channels
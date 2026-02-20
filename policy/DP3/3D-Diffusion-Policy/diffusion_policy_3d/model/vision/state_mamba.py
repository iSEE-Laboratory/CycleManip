import math
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from diffusion_policy_3d.model.diffusion.positional_embedding import SinusoidalPosEmb


class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=1024):
        super().__init__()

        # 预计算位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        """输入x: [B, L, d_model]"""
        L = x.size(1)
        return self.pe[:, :L, :]
    
class StateFusionMamba(nn.Module):
    def __init__(self, d_in, d_out, d_model=None, n_layers=1, use_cls=True, use_pos_emb=False):
        """
        Args:
            d_in: 输入维度
            d_out: 输出维度
            d_model: Mamba模型维度（默认等于d_out）
            n_layers: 堆叠层数
            use_cls: 是否使用可学习CLS token来聚合序列
        """
        super().__init__()
        d_model = d_model or d_out

        # Step 1: 映射输入维度
        self.state_mlp = nn.Linear(d_in, d_model)

        # Step 2: 堆叠多层 Mamba 模块
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=16,     # 状态维度，可调
                d_conv=4,       # 卷积核宽度，可调
                expand=2        # 扩展比例，可调
            ) for _ in range(n_layers)
        ])

        # Step 3: 是否使用CLS token（更稳定）
        self.use_cls = use_cls
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)

        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_out)

        if use_pos_emb:
            self.pos_encoder = SinusoidalPositionalEncoding(d_model)
        self.use_pos_emb = use_pos_emb

    def forward(self, state_list):
        """
        state_list: list of [K_b, d_in] tensors, len = B
        returns: Tensor [B, d_out]
        """
        B = len(state_list)
        device = state_list[0].device
        dtype = state_list[0].dtype
        K_max = max(s.shape[0] for s in state_list)

        # print(len(state_list))
        # === 1️⃣ Padding + Mask ===
        padded = torch.zeros(B, K_max, state_list[0].shape[-1], device=device, dtype=dtype)
        mask = torch.ones(B, K_max, dtype=torch.bool, device=device)
        for i, s in enumerate(state_list):
            k = s.shape[0]
            padded[i, :k] = s
            mask[i, :k] = False  # 有效部分

        # === 2️⃣ 输入投影 ===
        x = self.state_mlp(padded)  # [B, K_max, d_model]

        # === 3️⃣ CLS token optional ===
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
            x = torch.cat([cls, x], dim=1)
            # mask: prepend False for cls
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
            mask = torch.cat([cls_mask, mask], dim=1)  # [B, K_max+1]

        # === 添加正余弦位置编码 ===
        if self.use_pos_emb:
            pos_emb = self.pos_encoder(x)  # [B, L, D]
            x = x + pos_emb * (~mask).unsqueeze(-1)  # mask掉padding部分

        # === 4️⃣ Mamba 编码 ===
        for layer in self.mamba_layers:
            x = layer(x)  # (B, L, d_model)

        x = self.norm(x)

        # === 5️⃣ 输出聚合 ===
        if self.use_cls:
            state_feat = x[:, 0, :]  # CLS token输出
        else:
            valid_mask = (~mask).unsqueeze(-1)
            valid_len = valid_mask.sum(dim=1)
            state_feat = (x * valid_mask).sum(dim=1) / valid_len  # mean pooling

        state_feat = self.output_proj(state_feat)  # [B, d_out]
        return state_feat

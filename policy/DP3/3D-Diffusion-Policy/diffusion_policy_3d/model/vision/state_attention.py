import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def sinusoidal_position_encoding(max_len, dim):
    pe = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

class NoPaddingConvTokenizer(nn.Module):
    """无补零的卷积tokenizer"""
    def __init__(self, in_channels, out_channels, kernel_size=5, use_final_cls=False, posi_type="learning"):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size  # 步长=5，输出长度由卷积无padding计算决定

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=0  # 无补零
        )
        self.norm = nn.LayerNorm(out_channels)
        # 位置编码：仅用于原始序列位置
        if posi_type == "learning":
            self.pos_embedding = nn.Parameter(torch.randn(1, 1024, out_channels))  # 原始序列位置编码
            nn.init.normal_(self.pos_embedding, std=0.02)
        elif posi_type == "cos_sim":
            max_len = 1024
            dim = out_channels
            pe = torch.zeros(max_len, dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.pos_embedding =  pe.unsqueeze(0) 
        else:
            raise Exception("1")
            
        self.cls_token = nn.Parameter(torch.zeros(1, 1, out_channels))
        nn.init.normal_(self.cls_token, std=0.02)
        if use_final_cls:
            self.cls_token2 = nn.Parameter(torch.zeros(1, 1, out_channels))
            nn.init.normal_(self.cls_token2, std=0.02)
        self.use_final_cls = use_final_cls

    def forward(self, x):
        """
        x: 输入序列 (B, N, C)，B=批量，N=序列长度，C=输入通道
        return: 输出token序列 (B, L, C2)，L为卷积无padding输出长度，带位置编码
        """
        B, N, C = x.shape
        
        # 1. 无需补零，直接调整维度适应Conv1d：(B, N, C) → (B, C, N)
        x = x.permute(0, 2, 1)
        
        # 2. 卷积采样（无补零）：输出长度 L = floor((N - kernel_size) / stride) + 1
        x = self.conv(x)  # (B, C2, L)
        L = x.shape[2]
        
        # 3. 调整回 (B, L, C2)，适配注意力输入
        x = x.permute(0, 2, 1)
        
        # 4. 层归一化
        x = self.norm(x)
        
        if self.use_final_cls:
            # 5. 生成位置编码（长度与输出token一致）
            pos_emb = self.pos_embedding[:, :L+2, :].expand(B, -1, -1)  # 取前L个位置编码
            x = torch.cat([self.cls_token.expand(B, -1, -1), x, self.cls_token2.expand(B, -1, -1)], dim=1)
            # 加入位置编码
            x = x + pos_emb
        else:
            # 5. 生成位置编码（长度与输出token一致）
            pos_emb = self.pos_embedding[:, :L+1, :].expand(B, -1, -1)  # 取前L个位置编码
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
            # 加入位置编码
            x = x + pos_emb
            
        return x


class StateFusionAttention(nn.Module):
    def __init__(self, d_in, d_out, d_model=None, n_layers=2, n_heads=4, dropout=0.1, use_final_cls=False, use_diff=False, posi_type="learning", flip=True):
        """
        Args:
            d_in: 输入维度
            d_out: 输出维度
            d_model: Attention模型维度（默认等于d_out）
            n_layers: Transformer层数
            n_heads: 注意力头数
            use_cls: 是否使用可学习CLS token
            use_pos_emb: 是否使用正余弦位置编码
            dropout: dropout比例
        """
        super().__init__()
        d_model = d_model or d_out
        self.stride = 5
        self.use_final_cls = use_final_cls
        # === 输入线性映射 ===
        self.state_tokenize = NoPaddingConvTokenizer(d_in, d_model, kernel_size=self.stride, use_final_cls=use_final_cls, posi_type=posi_type)

        # === Transformer编码层 ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # 使用[B, L, D]格式
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # === 输出层 ===
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_out)
        self.use_diff = use_diff
        self.flip = flip

    def forward(self, state_list, verbos=False):
        """
        state_list: list of [K_b, d_in] tensors, len = B
        returns: Tensor [B, d_out]
        """
        B = len(state_list)
        device = state_list[0].device
        dtype = state_list[0].dtype
        K_max = max(s.shape[0] for s in state_list) 
        padded_right = self.stride - K_max % self.stride

        # === Padding + Mask ===
        padded = torch.zeros(B, K_max+padded_right, state_list[0].shape[1], device=device, dtype=dtype)
        if self.use_final_cls:
            mask = torch.ones(B, (K_max+padded_right)//self.stride+2, dtype=torch.bool, device=device)
        else:
            mask = torch.ones(B, (K_max+padded_right)//self.stride+1, dtype=torch.bool, device=device)
        for i, s in enumerate(state_list):
            if self.use_diff:
                diff_s = torch.diff(s, dim=0)
                diff_s = torch.cat([torch.zeros(1, diff_s.size(1)).to(diff_s.device), diff_s], dim=0)
                if self.flip:
                    s_reversed = diff_s.flip(dims=[0])
                else:
                    s_reversed = diff_s
            else:
                if self.flip:
                    s_reversed = s.flip(dims=[0])  # 仅对当前序列反向
                else:
                    s_reversed = s
            k = s_reversed.shape[0]
            padded[i, :k] = s_reversed  # 反向后的序列放左侧，右侧补零
            mask_token = max(0, k // self.stride)
            mask[i, :mask_token+1] = False  # 标记有效部分, 有cls token

        # === 输入映射 ===
        x = self.state_tokenize(padded)  # [B, (K_max+pad_left)//self.stride, d_model]

        assert x.shape[1] == mask.shape[1]

        # === Transformer 编码 ===
        # 注意: TransformerEncoder的src_key_padding_mask中 True 表示"需要被mask掉"
        x = self.encoder(x, src_key_padding_mask=mask)

        if verbos:
            state_feat = self.norm(x) 
            state_feat = self.output_proj(state_feat)  # [B, L, d_out]
            return state_feat[:, 0, :], state_feat[:, 1:-1, :], state_feat[:, -1, :]
        else:
            state_feat = self.norm(x[:, 0, :]) # 使用CLS token
            state_feat = self.output_proj(state_feat)  # [B, d_out]
            return state_feat

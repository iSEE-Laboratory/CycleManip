import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads=4, num_layers=2, dropout=0.1, max_len=10):
        super(TransformerEncoder, self).__init__()
        
        # 创建可学习的位置编码（不包括CLS token）
        self.positional_encoding = nn.Embedding(max_len, embed_dim)
        
        # 创建一个CLS token的可学习参数
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 形状 (1, 1, C)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # 输入为 [B, N, C]
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # LayerNorm
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, N, C)
        
        Returns:
            cls_feature: Tensor of shape (B, C)
        """
        B, N, C = x.shape
        
        # 获取位置编码
        positions = torch.arange(0, N, device=x.device).unsqueeze(0)  # (1, N)
        pos_enc = self.positional_encoding(positions)  # (1, N, C)
        x = x + pos_enc  # 加入位置编码
        
        # 扩展CLS token到batch维度
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, C)
        
        # 将CLS token拼接到序列最前面
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, C)
        
        # Transformer Encoder expects (B, N, C) since batch_first=True
        x = self.encoder(x)  # (B, N+1, C)
        
        # 归一化
        x = self.norm(x)
        
        # 输出CLS token特征
        cls_feature = x[:, 0, :]  # (B, C)
        return cls_feature

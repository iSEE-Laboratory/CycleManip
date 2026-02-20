import torch
import torch.nn as nn
import torch.nn.functional as F

class LanguageEncoderOnehot(nn.Module):
    """
    One-hot + MLP 编码器
    输入: 一个包含K个字符串（'1'~'8'）的list
    输出: Tensor, shape = [K, C]
    """
    def __init__(
                self, 
                language_embed_dim=None, 
                num_classes=10, 
                hidden_dim=64, 
                output_projection_dim=128, 
                language_keys = "instruction",
                device="cuda"
                ):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.n_output_channels = output_projection_dim
        self.language_key = language_keys
        # 定义MLP: 输入为 one-hot (num_classes)，输出为 C维
        self.mlp = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_projection_dim)
        )

    def forward(self, observations):
        """
        参数:
            str_list: list[str], 每个元素是 '1'~'8'
        返回:
            encoded: Tensor, shape = [K, C]
        """
        # step1: 将字符串转为整数索引 (0~7)
        assert self.language_key in observations
        instruction_text = observations[self.language_key]

        idx = torch.tensor([int(s) - 1 for s in instruction_text], dtype=torch.long, device=self.device)

        # step2: 转为one-hot编码 [K, num_classes]，并放到CUDA上
        onehot = F.one_hot(idx, num_classes=self.num_classes).float().to(self.device)

        # step3: 送入MLP编码
        encoded = self.mlp(onehot)
        return encoded
    
    def output_shape(self):
        return self.n_output_channels
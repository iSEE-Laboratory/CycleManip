import torch
import torch.nn as nn

import clip
from termcolor import cprint
import time

class LanguageEncoderFrozen(nn.Module):

    def __init__(self,
                language_embed_dim=512,
                output_projection_dim=128,
                language_keys = "instruction",
                device="cuda"):
        super().__init__()
        self.language_key = language_keys
        # Load CLIP model (using ViT-B/32 which has 512-dim text embeddings)
        cprint("[LanguaeEncoder] Loading CLIP model for language encoding...", "cyan")
        self.clip_model, _ = clip.load("ViT-B/32", device="cpu")
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()
        self.clip_model = self.clip_model.to(device)
        self.device = device
        
        # Add a projection MLP to align CLIP embedding with our feature space
        # Input: language_embed_dim (512 from CLIP), Output: configurable
        language_projection_dim = output_projection_dim  # Hard-coded projection dimension
        self.language_projection = nn.Sequential(
            nn.Linear(language_embed_dim, language_projection_dim),
            nn.ReLU(),
            nn.Linear(language_projection_dim, language_projection_dim)
        )
        self.n_output_channels = language_projection_dim
        cprint(f"[DP3Encoder] Language projection dim: {language_projection_dim}", "cyan")

    def forward(self, observations):
        assert self.language_key in observations
        instruction_text = observations[self.language_key]
        
        # Encode text using CLIP    
        with torch.no_grad():
            # Tokenize text
            text_tokens = clip.tokenize(instruction_text, truncate=True).to(self.device)
            # Encode text to get embeddings
            text_features = self.clip_model.encode_text(text_tokens).float()  # B * 512
        # Project language embeddings
        language_feat = self.language_projection(text_features)  # B * language_projection_dim: 128 here
        return language_feat
    
    def output_shape(self):
        return self.n_output_channels

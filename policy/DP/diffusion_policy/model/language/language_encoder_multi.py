import torch
import torch.nn as nn
import clip
from termcolor import cprint

class LanguageEncoderMulti(nn.Module):

    def __init__(self,
                language_embed_dim=512,
                output_projection_dim=128,
                language_keys=None,
                device="cuda"):
        super().__init__()

        # Default to a single key if no keys are specified
        if language_keys is None:
            language_keys = ["instruction"]
        
        self.language_keys = language_keys

        # Load CLIP model (using ViT-B/32 which has 512-dim text embeddings)
        cprint("[MultiLanguageEncoder] Loading CLIP model for language encoding...", "cyan")
        self.clip_model, _ = clip.load("ViT-B/32", device="cpu")
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()
        self.clip_model = self.clip_model.to(device)
        self.device = device
        
        # Add a projection MLP to align CLIP embedding with our feature space
        # Input: language_embed_dim (512 from CLIP), Output: configurable
        self.language_projection = nn.Sequential(
            nn.Linear(language_embed_dim, output_projection_dim),
            nn.ReLU(),
            nn.Linear(output_projection_dim, output_projection_dim)
        )
        
        self.n_output_channels = output_projection_dim

        self.head = nn.Linear(len(self.language_keys) * output_projection_dim, output_projection_dim)

        cprint(f"[MultiLanguageEncoder] Language projection dim: {output_projection_dim}", "cyan")

    def forward(self, observations):
        """
        Process multiple language inputs from the observation dict and encode them.
        Each language key (e.g., 'instruction') will be processed separately.
        """
        language_feats = []
        # Iterate over all language keys
        for key in self.language_keys:
            assert key in observations, f"Key '{key}' not found in observations"
            instruction_text = observations[key]
            # Tokenize text
            with torch.no_grad():
                text_tokens = clip.tokenize(instruction_text, truncate=True).to(self.device)
                # Encode text to get embeddings
                text_features = self.clip_model.encode_text(text_tokens).float()  # B * 512
            
            # Project the language features to the output dimension
            language_feat = self.language_projection(text_features)  # B * output_projection_dim
            language_feats.append(language_feat)
        combined_feats = torch.cat(language_feats, dim=-1)  # B * (len(language_keys) * output_projection_dim)
        final_output = self.head(combined_feats)
        return final_output

    def output_shape(self):
        # Return the final shape after concatenation
        return self.n_output_channels

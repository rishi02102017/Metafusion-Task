"""
Projection Layer Module
Maps vision features to text decoder's input space
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class ProjectionLayer(nn.Module):
    """
    Projects vision encoder output to text decoder's embedding space.
    
    This is a lightweight MLP that bridges the vision and language modalities.
    It converts a single vision embedding into a sequence of tokens that
    can be fed to the text decoder.
    """
    
    def __init__(
        self,
        vision_dim: int = 256,
        text_dim: int = 256,
        num_query_tokens: int = 8,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            vision_dim: Dimension of vision encoder output
            text_dim: Dimension expected by text decoder
            num_query_tokens: Number of "visual tokens" to generate for decoder
            hidden_dim: Hidden layer dimension (default: 2 * text_dim)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.num_query_tokens = num_query_tokens
        
        hidden_dim = hidden_dim or (2 * text_dim)
        
        # MLP to project and expand vision features
        self.projection = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_query_tokens * text_dim),
            nn.Dropout(dropout),
        )
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(text_dim)
        
        # Learnable position embeddings for visual tokens
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_query_tokens, text_dim) * 0.02
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Project vision features to decoder input space.
        
        Args:
            vision_features: (B, vision_dim) from vision encoder
            
        Returns:
            visual_tokens: (B, num_query_tokens, text_dim) for decoder
        """
        batch_size = vision_features.shape[0]
        
        # Project and reshape: (B, vision_dim) -> (B, num_query_tokens * text_dim)
        projected = self.projection(vision_features)
        
        # Reshape to sequence: (B, num_query_tokens, text_dim)
        visual_tokens = projected.view(batch_size, self.num_query_tokens, self.text_dim)
        
        # Add positional information and normalize
        visual_tokens = visual_tokens + self.position_embeddings
        visual_tokens = self.layer_norm(visual_tokens)
        
        return visual_tokens
    
    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class CrossAttentionProjection(nn.Module):
    """
    Alternative projection using cross-attention (Q-Former style).
    More expressive but slightly more parameters.
    """
    
    def __init__(
        self,
        vision_dim: int = 256,
        text_dim: int = 256,
        num_query_tokens: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_query_tokens = num_query_tokens
        self.text_dim = text_dim
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, text_dim) * 0.02
        )
        
        # Project vision features to same dimension
        self.vision_proj = nn.Linear(vision_dim, text_dim)
        
        # Cross-attention: queries attend to vision features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # FFN after attention
        self.ffn = nn.Sequential(
            nn.Linear(text_dim, text_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim * 2, text_dim),
            nn.Dropout(dropout),
        )
        
        self.norm1 = nn.LayerNorm(text_dim)
        self.norm2 = nn.LayerNorm(text_dim)
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (B, vision_dim) from vision encoder
            
        Returns:
            visual_tokens: (B, num_query_tokens, text_dim)
        """
        batch_size = vision_features.shape[0]
        
        # Expand queries for batch: (1, N, D) -> (B, N, D)
        queries = self.query_tokens.expand(batch_size, -1, -1)
        
        # Project vision features and add sequence dim: (B, D) -> (B, 1, D)
        vision_kv = self.vision_proj(vision_features).unsqueeze(1)
        
        # Cross-attention
        attended, _ = self.cross_attention(
            query=queries,
            key=vision_kv,
            value=vision_kv,
        )
        queries = self.norm1(queries + attended)
        
        # FFN
        queries = self.norm2(queries + self.ffn(queries))
        
        return queries


if __name__ == "__main__":
    # Test both projection types
    batch_size = 4
    vision_dim = 256
    text_dim = 256
    num_tokens = 8
    
    vision_features = torch.randn(batch_size, vision_dim)
    
    # Test MLP projection
    mlp_proj = ProjectionLayer(vision_dim, text_dim, num_tokens)
    mlp_out = mlp_proj(vision_features)
    print(f"MLP Projection:")
    print(f"  Input: {vision_features.shape}")
    print(f"  Output: {mlp_out.shape}")
    print(f"  Params: {mlp_proj.get_num_params():,}")
    
    # Test Cross-attention projection
    ca_proj = CrossAttentionProjection(vision_dim, text_dim, num_tokens)
    ca_out = ca_proj(vision_features)
    print(f"\nCross-Attention Projection:")
    print(f"  Input: {vision_features.shape}")
    print(f"  Output: {ca_out.shape}")
    print(f"  Params: {sum(p.numel() for p in ca_proj.parameters()):,}")

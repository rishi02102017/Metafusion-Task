"""
Text Decoder Module
Lightweight transformer decoder for generating structured person descriptions
Target: ~20-30M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class DecoderConfig:
    """Configuration for the text decoder."""
    vocab_size: int = 1000  # Small controlled vocabulary
    max_seq_length: int = 64  # Short descriptions only
    embed_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    def get_param_estimate(self) -> int:
        """Estimate total parameters."""
        # Embeddings
        embed_params = self.vocab_size * self.embed_dim
        pos_params = self.max_seq_length * self.embed_dim
        
        # Per layer: self-attn + cross-attn + ffn
        attn_params = 4 * self.embed_dim * self.embed_dim  # Q, K, V, O
        ffn_params = 2 * self.embed_dim * self.ffn_dim
        layer_params = 2 * attn_params + ffn_params + 4 * self.embed_dim  # + layer norms
        
        # Output head
        head_params = self.embed_dim * self.vocab_size
        
        total = embed_params + pos_params + self.num_layers * layer_params + head_params
        return total


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional masking."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, Tq, D)
            key: (B, Tk, D)
            value: (B, Tk, D)
            mask: Optional attention mask
            is_causal: Whether to apply causal masking
        """
        B, Tq, _ = query.shape
        Tk = key.shape[1]
        
        # Project and reshape for multi-head
        q = self.q_proj(query).view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask if needed
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(Tq, Tk, dtype=torch.bool, device=query.device), diagonal=1
            )
            attn = attn.masked_fill(causal_mask, float('-inf'))
        
        # Apply additional mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.embed_dim)
        out = self.out_proj(out)
        
        return out


class DecoderLayer(nn.Module):
    """Single transformer decoder layer with self-attention and cross-attention."""
    
    def __init__(self, config: DecoderConfig):
        super().__init__()
        
        # Self-attention (causal)
        self.self_attn = MultiHeadAttention(
            config.embed_dim, config.num_heads, config.dropout
        )
        self.self_attn_norm = nn.LayerNorm(config.embed_dim)
        
        # Cross-attention to visual tokens
        self.cross_attn = MultiHeadAttention(
            config.embed_dim, config.num_heads, config.dropout
        )
        self.cross_attn_norm = nn.LayerNorm(config.embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )
        self.ffn_norm = nn.LayerNorm(config.embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        visual_tokens: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) text embeddings
            visual_tokens: (B, Nv, D) visual tokens from projection
            self_attn_mask: Optional mask for self-attention
        """
        # Self-attention with residual
        residual = x
        x = self.self_attn_norm(x)
        x = residual + self.self_attn(x, x, x, is_causal=True)
        
        # Cross-attention with residual
        residual = x
        x = self.cross_attn_norm(x)
        x = residual + self.cross_attn(x, visual_tokens, visual_tokens)
        
        # FFN with residual
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn(x)
        
        return x


class TextDecoder(nn.Module):
    """
    Lightweight transformer decoder for generating person descriptions.
    
    Architecture:
    - Token + Position embeddings
    - N decoder layers (self-attn + cross-attn + FFN)
    - Output projection to vocabulary
    
    Target: ~20-30M parameters with default config
    """
    
    def __init__(self, config: Optional[DecoderConfig] = None):
        super().__init__()
        
        self.config = config or DecoderConfig()
        
        # Token embeddings
        self.token_embedding = nn.Embedding(
            self.config.vocab_size, self.config.embed_dim, padding_idx=self.config.pad_token_id
        )
        
        # Positional embeddings (learnable)
        self.position_embedding = nn.Embedding(
            self.config.max_seq_length, self.config.embed_dim
        )
        
        # Dropout after embeddings
        self.embed_dropout = nn.Dropout(self.config.dropout)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(self.config) for _ in range(self.config.num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(self.config.embed_dim)
        
        # Output projection (tied with token embeddings optionally)
        self.output_proj = nn.Linear(self.config.embed_dim, self.config.vocab_size, bias=False)
        
        # Tie weights
        self.output_proj.weight = self.token_embedding.weight
        
        self._init_weights()
        
        # Print param count
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Text Decoder initialized with {total_params:,} parameters")
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        visual_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training with teacher forcing.
        
        Args:
            input_ids: (B, T) token indices
            visual_tokens: (B, Nv, D) from projection layer
            attention_mask: Optional padding mask
            
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.shape
        
        # Get embeddings
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embed_dropout(x)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, visual_tokens)
        
        # Final norm and output projection
        x = self.final_norm(x)
        logits = self.output_proj(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        visual_tokens: torch.Tensor,
        max_length: int = 32,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            visual_tokens: (B, Nv, D) visual tokens
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (1.0 = neutral)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            
        Returns:
            generated_ids: (B, L) generated token indices
        """
        B = visual_tokens.shape[0]
        device = visual_tokens.device
        
        # Start with BOS token
        generated = torch.full(
            (B, 1), self.config.bos_token_id, dtype=torch.long, device=device
        )
        
        for _ in range(max_length - 1):
            # Get logits for next token
            logits = self.forward(generated, visual_tokens)
            next_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][:, -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences have EOS
            if (next_token == self.config.eos_token_id).all():
                break
        
        return generated
    
    def get_num_params(self) -> Tuple[int, int]:
        """Return (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def get_decoder_config(size: str = "small") -> DecoderConfig:
    """
    Get predefined decoder configurations.
    
    Sizes:
    - tiny: ~5M params (very fast, limited capacity)
    - small: ~15M params (balanced)
    - medium: ~25M params (better quality)
    """
    configs = {
        "tiny": DecoderConfig(
            vocab_size=500,
            embed_dim=192,
            num_heads=4,
            num_layers=3,
            ffn_dim=384,
        ),
        "small": DecoderConfig(
            vocab_size=1000,
            embed_dim=256,
            num_heads=4,
            num_layers=4,
            ffn_dim=512,
        ),
        "medium": DecoderConfig(
            vocab_size=1500,
            embed_dim=320,
            num_heads=8,
            num_layers=5,
            ffn_dim=640,
        ),
    }
    
    if size not in configs:
        raise ValueError(f"Size must be one of: {list(configs.keys())}")
    
    config = configs[size]
    print(f"Decoder config '{size}': ~{config.get_param_estimate():,} params (estimate)")
    return config


if __name__ == "__main__":
    # Test decoder
    config = get_decoder_config("small")
    decoder = TextDecoder(config)
    
    # Test forward pass
    batch_size = 4
    seq_length = 20
    num_visual_tokens = 8
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    visual_tokens = torch.randn(batch_size, num_visual_tokens, config.embed_dim)
    
    logits = decoder(input_ids, visual_tokens)
    print(f"\nForward pass:")
    print(f"  Input IDs: {input_ids.shape}")
    print(f"  Visual tokens: {visual_tokens.shape}")
    print(f"  Output logits: {logits.shape}")
    
    # Test generation
    generated = decoder.generate(visual_tokens, max_length=16)
    print(f"\nGeneration:")
    print(f"  Generated shape: {generated.shape}")
    print(f"  Sample: {generated[0].tolist()}")

"""
PersonVLM - Complete Vision-Language Model for Person Description
Combines: Vision Encoder + Projection + Text Decoder
Total target: ≤100M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

from .vision_encoder import VisionEncoder, get_vision_encoder
from .projection import ProjectionLayer
from .text_decoder import TextDecoder, DecoderConfig, get_decoder_config


@dataclass
class PersonVLMConfig:
    """Full model configuration."""
    # Vision encoder
    vision_backbone: str = "mobilevit_xs"
    vision_pretrained: bool = True
    vision_freeze_ratio: float = 0.9
    
    # Projection
    num_visual_tokens: int = 8
    projection_hidden_dim: int = 512
    
    # Text decoder
    decoder_size: str = "small"  # tiny, small, medium
    vocab_size: int = 3179  # Vocabulary size (default from MSP60k corpus)
    max_seq_length: int = 256  # Maximum sequence length (MSP60k has long captions)
    
    # Special token IDs (should match vocabulary)
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # Shared dimensions
    hidden_dim: int = 256
    
    # Training
    dropout: float = 0.1
    label_smoothing: float = 0.1


class PersonVLM(nn.Module):
    """
    Lightweight Vision-Language Model for Person Blob Description.
    
    Architecture:
        Person Image -> Vision Encoder (frozen) -> Projection -> Text Decoder -> Description
    
    Designed to generate short, structured descriptions like:
        "person wearing blue shirt and black pants, holding a phone, walking"
    """
    
    def __init__(self, config: Optional[PersonVLMConfig] = None, tokenizer=None):
        super().__init__()
        
        self.config = config or PersonVLMConfig()
        self.tokenizer = tokenizer
        
        print("=" * 60)
        print("Initializing PersonVLM")
        print("=" * 60)
        
        # 1. Vision Encoder (mostly frozen)
        self.vision_encoder = get_vision_encoder(
            backbone=self.config.vision_backbone,
            output_dim=self.config.hidden_dim,
            freeze_ratio=self.config.vision_freeze_ratio,
            pretrained=self.config.vision_pretrained,
        )
        
        # 2. Projection Layer
        self.projection = ProjectionLayer(
            vision_dim=self.config.hidden_dim,
            text_dim=self.config.hidden_dim,
            num_query_tokens=self.config.num_visual_tokens,
            hidden_dim=self.config.projection_hidden_dim,
            dropout=self.config.dropout,
        )
        
        # 3. Text Decoder
        decoder_config = get_decoder_config(self.config.decoder_size)
        decoder_config.embed_dim = self.config.hidden_dim
        decoder_config.vocab_size = self.config.vocab_size
        decoder_config.max_seq_length = self.config.max_seq_length
        decoder_config.pad_token_id = self.config.pad_token_id
        decoder_config.bos_token_id = self.config.bos_token_id
        decoder_config.eos_token_id = self.config.eos_token_id
        decoder_config.dropout = self.config.dropout
        self.decoder = TextDecoder(decoder_config)
        self.decoder_config = decoder_config
        
        # Print total parameters
        self._print_param_summary()
    
    def _print_param_summary(self):
        """Print parameter breakdown."""
        vision_total, vision_train = self.vision_encoder.get_num_params()
        proj_params = self.projection.get_num_params()
        decoder_total, decoder_train = self.decoder.get_num_params()
        
        total = vision_total + proj_params + decoder_total
        trainable = vision_train + proj_params + decoder_train
        
        print("\n" + "=" * 60)
        print("Parameter Summary")
        print("=" * 60)
        print(f"Vision Encoder:  {vision_total:>10,} total | {vision_train:>10,} trainable")
        print(f"Projection:      {proj_params:>10,} total | {proj_params:>10,} trainable")
        print(f"Text Decoder:    {decoder_total:>10,} total | {decoder_train:>10,} trainable")
        print("-" * 60)
        print(f"TOTAL:           {total:>10,} total | {trainable:>10,} trainable")
        print(f"Model Size:      {total / 1e6:.2f}M parameters")
        print("=" * 60 + "\n")
        
        if total > 100_000_000:
            print("WARNING: Model exceeds 100M parameter budget!")
        else:
            print(f"Model is within 100M budget ({total/1e6:.1f}M / 100M)")
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to visual tokens.
        
        Args:
            images: (B, 3, H, W) normalized images
            
        Returns:
            visual_tokens: (B, num_visual_tokens, hidden_dim)
        """
        # Vision encoding
        vision_features = self.vision_encoder(images)
        
        # Project to visual tokens
        visual_tokens = self.projection(vision_features)
        
        return visual_tokens
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            images: (B, 3, H, W) person crop images
            input_ids: (B, T) tokenized target text (teacher forcing)
            labels: (B, T) target labels for loss computation
            
        Returns:
            dict with 'logits' and optionally 'loss'
        """
        # Encode images to visual tokens
        visual_tokens = self.encode_image(images)
        
        # Decode to text logits
        logits = self.decoder(input_ids, visual_tokens)
        
        output = {"logits": logits}
        
        # Compute loss if labels provided
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Flatten for cross-entropy
            # Use -100 as ignore_index (standard PyTorch convention for masked labels)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                label_smoothing=self.config.label_smoothing,
            )
            output["loss"] = loss
        
        return output
    
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_length: int = 32,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        return_tokens: bool = False,
    ) -> List[str]:
        """
        Generate descriptions for person images.
        
        Args:
            images: (B, 3, H, W) person crop images
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            return_tokens: If True, return token IDs instead of strings
            
        Returns:
            List of generated descriptions
        """
        self.eval()
        
        # Encode images
        visual_tokens = self.encode_image(images)
        
        # Generate tokens
        generated_ids = self.decoder.generate(
            visual_tokens,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        
        if return_tokens or self.tokenizer is None:
            return generated_ids
        
        # Decode to strings
        descriptions = []
        for ids in generated_ids:
            # Convert tensor to list of integers for vocabulary lookup
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            # Remove special tokens and decode
            text = self.tokenizer.decode(
                ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            descriptions.append(text.strip())
        
        return descriptions
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters for optimizer."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def save_pretrained(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "config": self.config,
            "state_dict": self.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def from_pretrained(cls, path: str, tokenizer=None) -> "PersonVLM":
        """Load model from checkpoint."""
        # Allow loading custom config class (safe for our own checkpoints)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(config=checkpoint["config"], tokenizer=tokenizer)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Model loaded from {path}")
        return model


def create_person_vlm(
    vision_backbone: str = "mobilevit_xs",
    decoder_size: str = "small",
    vision_freeze_ratio: float = 0.9,
    vocab_size: int = 3179,
    max_seq_length: int = 128,
    pad_token_id: int = 0,
    bos_token_id: int = 1,
    eos_token_id: int = 2,
    tokenizer=None,
) -> PersonVLM:
    """
    Factory function to create PersonVLM with common configurations.
    
    Recommended configurations for ≤100M params:
    
    1. Lightweight (fast, ~15M params):
       - vision_backbone="mobilevit_xxs"
       - decoder_size="tiny"
       
    2. Balanced (recommended, ~30M params):
       - vision_backbone="mobilevit_xs"
       - decoder_size="small"
       
    3. Quality (better output, ~50M params):
       - vision_backbone="mobilevit_xs"
       - decoder_size="medium"
    """
    config = PersonVLMConfig(
        vision_backbone=vision_backbone,
        vision_freeze_ratio=vision_freeze_ratio,
        decoder_size=decoder_size,
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )
    
    return PersonVLM(config, tokenizer=tokenizer)


if __name__ == "__main__":
    # Test full model
    print("\n" + "=" * 60)
    print("Testing PersonVLM")
    print("=" * 60)
    
    # Create model
    model = create_person_vlm(
        vision_backbone="mobilevit_xs",
        decoder_size="small",
    )
    
    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (batch_size, 20))
    labels = input_ids.clone()
    
    output = model(images, input_ids, labels)
    print(f"\nForward pass:")
    print(f"  Images: {images.shape}")
    print(f"  Input IDs: {input_ids.shape}")
    print(f"  Logits: {output['logits'].shape}")
    print(f"  Loss: {output['loss'].item():.4f}")
    
    # Test generation
    generated = model.generate(images, max_length=16, return_tokens=True)
    print(f"\nGeneration:")
    print(f"  Generated shape: {generated.shape}")
    
    # Print memory usage
    if torch.cuda.is_available():
        model = model.cuda()
        images = images.cuda()
        torch.cuda.reset_peak_memory_stats()
        _ = model.generate(images, return_tokens=True)
        print(f"\nGPU Memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")

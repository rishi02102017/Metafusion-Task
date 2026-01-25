"""
Vision Encoder Module
Uses lightweight pretrained models: MobileViT, EfficientNet-Lite, or ViT-Tiny
Most weights are frozen to keep training efficient
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Tuple


class VisionEncoder(nn.Module):
    """
    Lightweight vision encoder for person blob feature extraction.
    
    Supported backbones:
    - mobilevit_xxs: ~1.3M params, very efficient
    - mobilevit_xs: ~2.3M params
    - efficientnet_lite0: ~4.7M params
    - vit_tiny_patch16_224: ~5.7M params
    - mobilenetv3_small_100: ~2.5M params
    
    By default, most layers are frozen except the last few.
    """
    
    SUPPORTED_BACKBONES = {
        "mobilevit_xxs": {"embed_dim": 320, "params": "1.3M"},
        "mobilevit_xs": {"embed_dim": 384, "params": "2.3M"},
        "efficientnet_lite0": {"embed_dim": 1280, "params": "4.7M"},
        "vit_tiny_patch16_224": {"embed_dim": 192, "params": "5.7M"},
        "mobilenetv3_small_100": {"embed_dim": 576, "params": "2.5M"},
    }
    
    def __init__(
        self,
        backbone: str = "mobilevit_xs",
        pretrained: bool = True,
        freeze_ratio: float = 0.8,
        output_dim: Optional[int] = None,
    ):
        """
        Args:
            backbone: Name of the pretrained backbone
            pretrained: Whether to load pretrained weights
            freeze_ratio: Fraction of layers to freeze (0.0 = none, 1.0 = all)
            output_dim: If specified, adds a projection to this dimension
        """
        super().__init__()
        
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Backbone {backbone} not supported. "
                f"Choose from: {list(self.SUPPORTED_BACKBONES.keys())}"
            )
        
        self.backbone_name = backbone
        self.embed_dim = self.SUPPORTED_BACKBONES[backbone]["embed_dim"]
        
        # Load pretrained model without classification head
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool="avg",  # Global average pooling
        )
        
        # Freeze layers based on ratio
        self._freeze_layers(freeze_ratio)
        
        # Optional output projection
        self.output_proj = None
        if output_dim is not None and output_dim != self.embed_dim:
            self.output_proj = nn.Linear(self.embed_dim, output_dim)
            self.embed_dim = output_dim
    
    def _freeze_layers(self, freeze_ratio: float):
        """Freeze a portion of the backbone layers."""
        if freeze_ratio <= 0:
            return
            
        # Get all parameters
        params = list(self.backbone.parameters())
        num_params = len(params)
        freeze_until = int(num_params * freeze_ratio)
        
        for i, param in enumerate(params):
            if i < freeze_until:
                param.requires_grad = False
        
        # Log frozen/trainable stats
        total = sum(p.numel() for p in self.backbone.parameters())
        frozen = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)
        trainable = total - frozen
        
        print(f"Vision Encoder [{self.backbone_name}]:")
        print(f"  Total params: {total:,}")
        print(f"  Frozen params: {frozen:,} ({frozen/total*100:.1f}%)")
        print(f"  Trainable params: {trainable:,} ({trainable/total*100:.1f}%)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from person blob images.
        
        Args:
            x: Input tensor of shape (B, C, H, W), typically (B, 3, 224, 224)
            
        Returns:
            features: Tensor of shape (B, embed_dim)
        """
        features = self.backbone(x)
        
        if self.output_proj is not None:
            features = self.output_proj(features)
        
        return features
    
    def get_num_params(self) -> Tuple[int, int]:
        """Return (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def get_vision_encoder(
    backbone: str = "mobilevit_xs",
    output_dim: int = 256,
    freeze_ratio: float = 0.9,
    pretrained: bool = True,
) -> VisionEncoder:
    """
    Factory function to create a vision encoder with recommended settings.
    
    Args:
        backbone: Backbone architecture name
        output_dim: Output feature dimension (for projection layer compatibility)
        freeze_ratio: How much to freeze (0.9 recommended for fine-tuning)
        pretrained: Use pretrained weights
        
    Returns:
        Configured VisionEncoder instance
    """
    return VisionEncoder(
        backbone=backbone,
        pretrained=pretrained,
        freeze_ratio=freeze_ratio,
        output_dim=output_dim,
    )


if __name__ == "__main__":
    # Quick test
    encoder = get_vision_encoder(backbone="mobilevit_xs", output_dim=256)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = encoder(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    total, trainable = encoder.get_num_params()
    print(f"\nTotal params: {total:,}")
    print(f"Trainable params: {trainable:,}")

"""
PersonVLM with Pretrained Decoder (DistilGPT-2)
===============================================
Uses a pretrained language model as the decoder instead of training from scratch.
This should give better language generation quality due to prior knowledge.

Architecture:
- Vision Encoder: MobileViT-XS (pretrained, mostly frozen)
- Projection: MLP to map vision features to GPT-2's embedding space
- Decoder: DistilGPT-2 (pretrained, fine-tuned)

Total: ~87M parameters (within 100M budget)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from transformers import GPT2LMHeadModel, GPT2Config

from .vision_encoder import get_vision_encoder


@dataclass
class PersonVLMPretrainedConfig:
    """Configuration for PersonVLM with pretrained decoder."""
    # Vision encoder
    vision_backbone: str = "mobilevit_xs"
    vision_pretrained: bool = True
    vision_freeze_ratio: float = 0.8
    
    # Projection
    num_visual_tokens: int = 8
    
    # Decoder (DistilGPT-2)
    decoder_model: str = "distilgpt2"
    freeze_decoder_ratio: float = 0.0  # Fine-tune all decoder layers
    
    # Training
    max_seq_length: int = 128
    dropout: float = 0.1
    label_smoothing: float = 0.1


class ProjectionLayer(nn.Module):
    """Project vision features to GPT-2 embedding space."""
    
    def __init__(
        self,
        vision_dim: int,
        text_dim: int,  # GPT-2 hidden size (768 for distilgpt2)
        num_tokens: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_tokens = num_tokens
        self.text_dim = text_dim
        
        # MLP to project and expand to multiple tokens
        self.projection = nn.Sequential(
            nn.Linear(vision_dim, text_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim * 2, text_dim * num_tokens),
            nn.Dropout(dropout),
        )
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(text_dim)
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (B, vision_dim) pooled vision features
            
        Returns:
            visual_tokens: (B, num_tokens, text_dim) tokens for GPT-2
        """
        # Project: (B, vision_dim) -> (B, text_dim * num_tokens)
        projected = self.projection(vision_features)
        
        # Reshape to tokens: (B, num_tokens, text_dim)
        B = vision_features.shape[0]
        visual_tokens = projected.view(B, self.num_tokens, self.text_dim)
        
        # Normalize
        visual_tokens = self.layer_norm(visual_tokens)
        
        return visual_tokens


class PersonVLMPretrained(nn.Module):
    """
    PersonVLM with Pretrained DistilGPT-2 Decoder.
    
    Uses prefix-based visual conditioning:
    - Visual features are projected to token embeddings
    - These are prepended to the text sequence
    - GPT-2 generates conditioned on this visual prefix
    """
    
    def __init__(self, config: Optional[PersonVLMPretrainedConfig] = None, tokenizer=None):
        super().__init__()
        
        self.config = config or PersonVLMPretrainedConfig()
        self.tokenizer = tokenizer
        
        print("=" * 60)
        print("Initializing PersonVLM with Pretrained Decoder")
        print("=" * 60)
        
        # 1. Vision Encoder (MobileViT-XS)
        # Use 256 as intermediate dim, will project to 768
        self.vision_encoder = get_vision_encoder(
            backbone=self.config.vision_backbone,
            output_dim=256,
            freeze_ratio=self.config.vision_freeze_ratio,
            pretrained=self.config.vision_pretrained,
        )
        
        # 2. Load pretrained GPT-2
        print(f"\nLoading pretrained decoder: {self.config.decoder_model}")
        self.decoder = GPT2LMHeadModel.from_pretrained(self.config.decoder_model)
        self.decoder_config = self.decoder.config
        
        # Get GPT-2's hidden size
        self.text_dim = self.decoder_config.n_embd  # 768 for distilgpt2
        print(f"Decoder hidden size: {self.text_dim}")
        
        # 3. Projection Layer (vision -> GPT-2 embedding space)
        self.projection = ProjectionLayer(
            vision_dim=256,
            text_dim=self.text_dim,
            num_tokens=self.config.num_visual_tokens,
            dropout=self.config.dropout,
        )
        
        # 4. Optionally freeze some decoder layers
        if self.config.freeze_decoder_ratio > 0:
            self._freeze_decoder_layers(self.config.freeze_decoder_ratio)
        
        # Print parameter summary
        self._print_param_summary()
    
    def _freeze_decoder_layers(self, ratio: float):
        """Freeze a portion of decoder layers."""
        num_layers = self.decoder_config.n_layer
        freeze_layers = int(num_layers * ratio)
        
        print(f"Freezing {freeze_layers}/{num_layers} decoder layers")
        
        # Freeze embeddings
        for param in self.decoder.transformer.wte.parameters():
            param.requires_grad = False
        for param in self.decoder.transformer.wpe.parameters():
            param.requires_grad = False
        
        # Freeze early layers
        for i in range(freeze_layers):
            for param in self.decoder.transformer.h[i].parameters():
                param.requires_grad = False
    
    def _print_param_summary(self):
        """Print parameter breakdown."""
        vision_total = sum(p.numel() for p in self.vision_encoder.parameters())
        vision_train = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
        
        proj_total = sum(p.numel() for p in self.projection.parameters())
        
        decoder_total = sum(p.numel() for p in self.decoder.parameters())
        decoder_train = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        
        total = vision_total + proj_total + decoder_total
        trainable = vision_train + proj_total + decoder_train
        
        print("\n" + "=" * 60)
        print("Parameter Summary")
        print("=" * 60)
        print(f"Vision Encoder:  {vision_total:>12,} total | {vision_train:>12,} trainable")
        print(f"Projection:      {proj_total:>12,} total | {proj_total:>12,} trainable")
        print(f"Decoder (GPT-2): {decoder_total:>12,} total | {decoder_train:>12,} trainable")
        print("-" * 60)
        print(f"TOTAL:           {total:>12,} total | {trainable:>12,} trainable")
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
            visual_embeds: (B, num_visual_tokens, text_dim)
        """
        # Vision encoding
        vision_features = self.vision_encoder(images)
        
        # Project to GPT-2 embedding space
        visual_embeds = self.projection(vision_features)
        
        return visual_embeds
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            images: (B, 3, H, W) person crop images
            input_ids: (B, T) tokenized target text
            labels: (B, T) target labels for loss computation
            attention_mask: (B, T) attention mask
            
        Returns:
            dict with 'logits' and optionally 'loss'
        """
        B, T = input_ids.shape
        device = input_ids.device
        
        # Encode images to visual tokens
        visual_embeds = self.encode_image(images)  # (B, num_visual_tokens, text_dim)
        num_visual = visual_embeds.shape[1]
        
        # Get text embeddings from GPT-2
        text_embeds = self.decoder.transformer.wte(input_ids)  # (B, T, text_dim)
        
        # Concatenate: [visual_tokens, text_tokens]
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)  # (B, num_visual + T, text_dim)
        
        # Create attention mask for the full sequence
        if attention_mask is None:
            attention_mask = torch.ones(B, T, device=device)
        
        # Prepend 1s for visual tokens (always attend to them)
        visual_mask = torch.ones(B, num_visual, device=device)
        full_attention_mask = torch.cat([visual_mask, attention_mask], dim=1)
        
        # Create position IDs
        seq_length = inputs_embeds.shape[1]
        position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(B, -1)
        
        # Forward through GPT-2
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            position_ids=position_ids,
            output_hidden_states=False,
        )
        
        logits = outputs.logits  # (B, num_visual + T, vocab_size)
        
        # Only take logits for text positions (exclude visual prefix)
        text_logits = logits[:, num_visual:, :]  # (B, T, vocab_size)
        
        output = {"logits": text_logits}
        
        # Compute loss if labels provided
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = text_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                label_smoothing=self.config.label_smoothing,
            )
            output["loss"] = loss
        
        return output
    
    def _clean_generated_text(self, text: str) -> str:
        """
        Post-process generated text to fix common issues.
        
        Fixes:
        - Leading fragments ("of", ",", "(presumably", etc.)
        - Lowercase first letter
        - Incomplete sentences
        - Repeated phrases
        """
        import re
        
        # Strip whitespace
        text = text.strip()
        
        if not text:
            return "The image shows a person."
        
        # Remove common leading fragments that indicate incomplete generation
        fragment_patterns = [
            r'^[\(\[\{].*?[\)\]\}]\s*',  # Remove leading parenthetical/bracket content
            r'^[,;:]\s*',                 # Remove leading punctuation
            r'^(of|and|or|but|with|in|on|at|to|for|from|by)\s+',  # Remove leading prepositions/conjunctions (case insensitive)
            r'^[-–—]\s*',                 # Remove leading dashes
            r'^\.\s*',                    # Remove leading periods
        ]
        
        for pattern in fragment_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        text = text.strip()
        
        if not text:
            return "The image shows a person."
        
        # Ensure first letter is capitalized
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        # If text doesn't start with a proper sentence structure, add "The image shows"
        proper_starts = [
            'the image', 'this image', 'here is', 'here\'s', 'a ', 'an ', 
            'the person', 'this person', 'a person', 'the individual',
            'this is', 'we see', 'shown here', 'the photo', 'this photo',
            'he ', 'she ', 'they '
        ]
        
        text_lower = text.lower()
        has_proper_start = any(text_lower.startswith(start) for start in proper_starts)
        
        if not has_proper_start:
            # Check if it starts with a descriptor that could be a valid start
            valid_descriptor_starts = ['adult', 'male', 'female', 'young', 'child', 'person', 'man', 'woman', 'boy', 'girl']
            starts_with_descriptor = any(text_lower.startswith(desc) for desc in valid_descriptor_starts)
            
            if starts_with_descriptor:
                text = "The image shows " + text[0].lower() + text[1:]
            elif not text_lower.startswith(('i ', 'you ')):  # Avoid "I" or "You" starts
                # For other cases, prepend proper start
                text = "The image shows " + text[0].lower() + text[1:]
        
        # Remove duplicate consecutive sentences/phrases
        sentences = text.split('. ')
        seen = set()
        unique_sentences = []
        for sent in sentences:
            sent_normalized = sent.strip().lower()
            if sent_normalized and sent_normalized not in seen:
                seen.add(sent_normalized)
                unique_sentences.append(sent.strip())
        text = '. '.join(unique_sentences)
        
        # Ensure proper ending
        text = text.rstrip()
        if text and text[-1] not in '.!?':
            # Check if the last sentence is incomplete (ends mid-word or with conjunction)
            last_words = text.split()[-3:] if len(text.split()) >= 3 else text.split()
            incomplete_endings = ['the', 'a', 'an', 'and', 'or', 'but', 'with', 'in', 'on', 'to', 'is', 'are', 'was', 'were', 'has', 'have', 'no']
            
            if last_words and last_words[-1].lower() in incomplete_endings:
                # Remove incomplete ending
                words = text.split()
                while words and words[-1].lower() in incomplete_endings:
                    words.pop()
                text = ' '.join(words)
            
            if text and text[-1] not in '.!?':
                text += '.'
        
        return text
    
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_length: int = 64,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        use_prompt: bool = True,
    ) -> List[str]:
        """
        Generate descriptions for person images.
        
        Args:
            images: (B, 3, H, W) person crop images
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            do_sample: Whether to sample or use greedy decoding
            use_prompt: Whether to use a prompt prefix for better sentence structure
            
        Returns:
            List of generated descriptions
        """
        self.eval()
        device = images.device
        B = images.shape[0]
        
        # Get tokenizer
        if self.tokenizer is not None:
            tokenizer = self.tokenizer
        else:
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained(self.config.decoder_model)
        
        # Encode images
        visual_embeds = self.encode_image(images)  # (B, num_visual, text_dim)
        num_visual = visual_embeds.shape[1]
        
        eos_token_id = self.decoder_config.eos_token_id or 50256
        
        # Use prompt prefix for better sentence structure
        if use_prompt:
            # Start with "The image shows" to guide generation
            prompt = "The image shows"
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
            prompt_ids = prompt_ids.to(device).expand(B, -1)  # (B, prompt_len)
            current_ids = prompt_ids
        else:
            # Generate first token from visual context only (original approach)
            position_ids = torch.arange(num_visual, device=device).unsqueeze(0).expand(B, -1)
            attention_mask = torch.ones(B, num_visual, device=device)
            
            outputs = self.decoder(
                inputs_embeds=visual_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            
            first_logits = outputs.logits[:, -1, :] / temperature
            
            if top_k > 0:
                indices_to_remove = first_logits < torch.topk(first_logits, top_k)[0][:, -1, None]
                first_logits[indices_to_remove] = float('-inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(first_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                first_logits[indices_to_remove] = float('-inf')
            
            if do_sample:
                probs = F.softmax(first_logits, dim=-1)
                first_token = torch.multinomial(probs, num_samples=1)
            else:
                first_token = first_logits.argmax(dim=-1, keepdim=True)
            
            current_ids = first_token
        
        # Generate remaining tokens
        for _ in range(max_length - current_ids.shape[1]):
            # Check for EOS
            if (current_ids[:, -1] == eos_token_id).all():
                break
                
            # Get text embeddings for current generated sequence
            text_embeds = self.decoder.transformer.wte(current_ids)
            
            # Concatenate with visual prefix
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            
            # Create position IDs for full sequence
            seq_len = inputs_embeds.shape[1]
            attention_mask = torch.ones(B, seq_len, device=device)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
            
            # Forward pass
            outputs = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            
            # Get logits for last position
            next_logits = outputs.logits[:, -1, :] / temperature
            
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
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')
            
            # Sample or greedy
            if do_sample:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            
            # Append to sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)
        
        # Decode to strings
        descriptions = []
        for ids in current_ids:
            ids_list = ids.tolist()
            # Remove EOS tokens from end
            while ids_list and ids_list[-1] == eos_token_id:
                ids_list.pop()
            
            text = tokenizer.decode(ids_list, skip_special_tokens=True)
            
            # Apply post-processing to fix any remaining issues
            text = self._clean_generated_text(text)
            
            descriptions.append(text)
        
        return descriptions
    
    def save_pretrained(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "config": self.config,
            "state_dict": self.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def from_pretrained(cls, path: str, tokenizer=None) -> "PersonVLMPretrained":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(config=checkpoint["config"], tokenizer=tokenizer)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Model loaded from {path}")
        return model


def create_person_vlm_pretrained(
    vision_backbone: str = "mobilevit_xs",
    vision_freeze_ratio: float = 0.8,
    decoder_model: str = "distilgpt2",
    freeze_decoder_ratio: float = 0.0,
    tokenizer=None,
) -> PersonVLMPretrained:
    """Factory function to create PersonVLM with pretrained decoder."""
    config = PersonVLMPretrainedConfig(
        vision_backbone=vision_backbone,
        vision_freeze_ratio=vision_freeze_ratio,
        decoder_model=decoder_model,
        freeze_decoder_ratio=freeze_decoder_ratio,
    )
    return PersonVLMPretrained(config, tokenizer=tokenizer)


if __name__ == "__main__":
    # Test the model
    print("\n" + "=" * 60)
    print("Testing PersonVLM with Pretrained Decoder")
    print("=" * 60)
    
    model = create_person_vlm_pretrained()
    
    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    
    # Use GPT-2 tokenizer for testing
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dummy input
    texts = ["person wearing blue shirt", "woman in red dress walking"]
    encoded = tokenizer(texts, padding=True, return_tensors="pt")
    input_ids = encoded.input_ids
    labels = input_ids.clone()
    labels[encoded.attention_mask == 0] = -100
    
    output = model(images, input_ids, labels)
    print(f"\nForward pass:")
    print(f"  Images: {images.shape}")
    print(f"  Input IDs: {input_ids.shape}")
    print(f"  Logits: {output['logits'].shape}")
    print(f"  Loss: {output['loss'].item():.4f}")

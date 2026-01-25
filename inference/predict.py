"""
Inference Module for Person VLM
Easy-to-use interface for generating person descriptions
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

try:
    from torchvision import transforms
except ImportError:
    transforms = None


class PersonDescriber:
    """
    High-level interface for person description generation.
    
    Usage:
        describer = PersonDescriber.from_pretrained("checkpoints/best_model.pt")
        description = describer.describe("person.jpg")
        # -> "person wearing blue shirt and black pants, holding a phone, walking"
    """
    
    def __init__(
        self,
        model,
        vocab,
        device: str = "auto",
        image_size: int = 224,
    ):
        """
        Args:
            model: PersonVLM model
            vocab: PersonVocabulary
            device: Device to use ("auto", "cuda", "cpu")
            image_size: Input image size
        """
        self.vocab = vocab
        self.image_size = image_size
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        
        print(f"PersonDescriber initialized on {self.device}")
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        vocab_path: Optional[str] = None,
        device: str = "auto",
    ) -> "PersonDescriber":
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            vocab_path: Optional path to vocabulary (uses default if not provided)
            device: Device to use
            
        Returns:
            PersonDescriber instance
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from models import PersonVLM
        from data import PersonVocabulary
        
        # Load vocabulary
        if vocab_path and os.path.exists(vocab_path):
            vocab = PersonVocabulary.load(vocab_path)
        else:
            vocab = PersonVocabulary()
            print("Using default vocabulary")
        
        # Load model
        model = PersonVLM.from_pretrained(checkpoint_path, tokenizer=vocab)
        
        return cls(model, vocab, device=device)
    
    def _load_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """Load and preprocess image."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        return self.transform(image)
    
    @torch.no_grad()
    def describe(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        max_length: int = 32,
    ) -> str:
        """
        Generate description for a single image.
        
        Args:
            image: Image path, PIL Image, or numpy array
            temperature: Sampling temperature (lower = more deterministic)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            max_length: Maximum output length
            
        Returns:
            Generated description string
        """
        # Preprocess
        img_tensor = self._load_image(image).unsqueeze(0).to(self.device)
        
        # Generate
        descriptions = self.model.generate(
            img_tensor,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_tokens=False,
        )
        
        return descriptions[0] if descriptions else ""
    
    @torch.no_grad()
    def describe_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        max_length: int = 32,
        batch_size: int = 16,
    ) -> List[str]:
        """
        Generate descriptions for multiple images.
        
        Args:
            images: List of images
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            max_length: Maximum output length
            batch_size: Processing batch size
            
        Returns:
            List of generated descriptions
        """
        descriptions = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Preprocess batch
            tensors = [self._load_image(img) for img in batch_images]
            batch_tensor = torch.stack(tensors).to(self.device)
            
            # Generate
            batch_descriptions = self.model.generate(
                batch_tensor,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                return_tokens=False,
            )
            
            descriptions.extend(batch_descriptions)
        
        return descriptions
    
    @torch.no_grad()
    def describe_with_confidence(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        num_samples: int = 5,
        temperature: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Generate description with confidence estimation.
        
        Generates multiple samples and returns the most common one
        along with consistency-based confidence.
        
        Args:
            image: Input image
            num_samples: Number of samples to generate
            temperature: Sampling temperature
            
        Returns:
            Dict with 'description', 'confidence', and 'alternatives'
        """
        # Preprocess
        img_tensor = self._load_image(image).unsqueeze(0).to(self.device)
        
        # Generate multiple samples
        samples = []
        for _ in range(num_samples):
            desc = self.model.generate(
                img_tensor,
                temperature=temperature,
                return_tokens=False,
            )[0]
            samples.append(desc)
        
        # Find most common description (exact match)
        from collections import Counter
        counter = Counter(samples)
        most_common, count = counter.most_common(1)[0]
        
        # Confidence based on consistency
        confidence = count / num_samples
        
        # Get alternatives
        alternatives = [desc for desc, _ in counter.most_common() if desc != most_common]
        
        return {
            "description": most_common,
            "confidence": confidence,
            "alternatives": alternatives[:3],  # Top 3 alternatives
        }
    
    def benchmark(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        num_runs: int = 100,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """
        Benchmark inference speed.
        
        Args:
            image: Test image
            num_runs: Number of inference runs
            warmup: Warmup runs
            
        Returns:
            Dict with timing statistics
        """
        # Preprocess once
        img_tensor = self._load_image(image).unsqueeze(0).to(self.device)
        
        # Warmup
        for _ in range(warmup):
            self.model.generate(img_tensor, return_tokens=True)
        
        # Synchronize if CUDA
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.model.generate(img_tensor, return_tokens=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        times = np.array(times)
        
        return {
            "mean_ms": times.mean() * 1000,
            "std_ms": times.std() * 1000,
            "min_ms": times.min() * 1000,
            "max_ms": times.max() * 1000,
            "fps": 1.0 / times.mean(),
        }


def load_model(
    checkpoint_path: str,
    device: str = "auto",
) -> PersonDescriber:
    """
    Convenience function to load model.
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device to use
        
    Returns:
        PersonDescriber instance
    """
    return PersonDescriber.from_pretrained(checkpoint_path, device=device)


# Example usage with ONNX export for deployment
class ONNXExporter:
    """Export model to ONNX format for deployment."""
    
    @staticmethod
    def export(
        model,
        output_path: str,
        image_size: int = 224,
        opset_version: int = 14,
    ):
        """
        Export model to ONNX.
        
        Args:
            model: PersonVLM model
            output_path: Output ONNX file path
            image_size: Input image size
            opset_version: ONNX opset version
        """
        try:
            import onnx
        except ImportError:
            raise ImportError("onnx required for export. Run: pip install onnx")
        
        model.eval()
        device = next(model.parameters()).device
        
        # Dummy inputs
        dummy_image = torch.randn(1, 3, image_size, image_size).to(device)
        dummy_input_ids = torch.ones(1, 1, dtype=torch.long).to(device)  # BOS token
        
        # Export
        torch.onnx.export(
            model,
            (dummy_image, dummy_input_ids),
            output_path,
            input_names=["image", "input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "image": {0: "batch"},
                "input_ids": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
            opset_version=opset_version,
        )
        
        # Verify
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        print(f"Model exported to {output_path}")


if __name__ == "__main__":
    print("Person VLM Inference Module")
    print("=" * 50)
    print("\nExample usage:")
    print("""
    from inference import PersonDescriber
    
    # Load model
    describer = PersonDescriber.from_pretrained("checkpoints/best_model.pt")
    
    # Single image
    description = describer.describe("person.jpg")
    print(description)
    # -> "person wearing blue shirt and black pants, holding a phone, walking"
    
    # Batch processing
    descriptions = describer.describe_batch(["person1.jpg", "person2.jpg"])
    
    # With confidence
    result = describer.describe_with_confidence("person.jpg")
    print(f"Description: {result['description']}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    # Benchmark
    stats = describer.benchmark("person.jpg")
    print(f"Inference time: {stats['mean_ms']:.2f} ms ({stats['fps']:.1f} FPS)")
    """)

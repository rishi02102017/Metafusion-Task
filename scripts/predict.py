#!/usr/bin/env python3
"""
Prediction Script for Person VLM
================================
Generate descriptions for person blob images

Usage:
    python predict.py --checkpoint best_model.pt --image person.jpg
    python predict.py --checkpoint best_model.pt --image_dir ./persons/ --output results.json
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

from inference import PersonDescriber


def main():
    parser = argparse.ArgumentParser(
        description="Generate person descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python predict.py --checkpoint best_model.pt --image person.jpg
  
  # Directory of images
  python predict.py --checkpoint best_model.pt --image_dir ./persons/ --output results.json
  
  # With confidence estimation
  python predict.py --checkpoint best_model.pt --image person.jpg --with_confidence
  
  # Benchmark speed
  python predict.py --checkpoint best_model.pt --image person.jpg --benchmark
        """,
    )
    
    # Model
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--vocab", type=str,
        help="Path to vocabulary file (optional)"
    )
    
    # Input
    parser.add_argument(
        "--image", type=str,
        help="Single image path"
    )
    parser.add_argument(
        "--image_dir", type=str,
        help="Directory of images"
    )
    parser.add_argument(
        "--image_list", type=str,
        help="Text file with image paths (one per line)"
    )
    
    # Output
    parser.add_argument(
        "--output", type=str,
        help="Output JSON file for batch predictions"
    )
    
    # Generation parameters
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_k", type=int, default=50,
        help="Top-k sampling"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9,
        help="Nucleus sampling"
    )
    parser.add_argument(
        "--max_length", type=int, default=32,
        help="Maximum output length"
    )
    
    # Options
    parser.add_argument(
        "--with_confidence", action="store_true",
        help="Include confidence estimation"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run speed benchmark"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for batch processing"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use (auto, cuda, cpu)"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not any([args.image, args.image_dir, args.image_list]):
        parser.error("One of --image, --image_dir, or --image_list is required")
    
    # Load model
    print("Loading model...")
    describer = PersonDescriber.from_pretrained(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        device=args.device,
    )
    
    # Collect images
    images = []
    if args.image:
        images = [args.image]
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            images.extend(image_dir.glob(f"*{ext}"))
            images.extend(image_dir.glob(f"*{ext.upper()}"))
        images = sorted([str(p) for p in images])
    elif args.image_list:
        with open(args.image_list, 'r') as f:
            images = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(images)} image(s)...")
    
    # Benchmark mode
    if args.benchmark:
        if not images:
            parser.error("--benchmark requires at least one image")
        
        print("\nRunning benchmark...")
        stats = describer.benchmark(images[0], num_runs=100)
        
        print(f"\nBenchmark Results:")
        print(f"  Mean: {stats['mean_ms']:.2f} ms")
        print(f"  Std:  {stats['std_ms']:.2f} ms")
        print(f"  Min:  {stats['min_ms']:.2f} ms")
        print(f"  Max:  {stats['max_ms']:.2f} ms")
        print(f"  FPS:  {stats['fps']:.1f}")
        return
    
    # Process images
    results = []
    
    if len(images) == 1:
        # Single image
        image_path = images[0]
        
        if args.with_confidence:
            result = describer.describe_with_confidence(
                image_path,
                temperature=args.temperature,
            )
            print(f"\nImage: {image_path}")
            print(f"Description: {result['description']}")
            print(f"Confidence: {result['confidence']:.2f}")
            if result['alternatives']:
                print(f"Alternatives: {result['alternatives']}")
            
            results.append({
                "image": image_path,
                **result,
            })
        else:
            description = describer.describe(
                image_path,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                max_length=args.max_length,
            )
            print(f"\nImage: {image_path}")
            print(f"Description: {description}")
            
            results.append({
                "image": image_path,
                "description": description,
            })
    else:
        # Batch processing
        descriptions = describer.describe_batch(
            images,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        
        print("\nResults:")
        print("-" * 60)
        
        for image_path, description in zip(images, descriptions):
            print(f"{Path(image_path).name}: {description}")
            results.append({
                "image": str(image_path),
                "description": description,
            })
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

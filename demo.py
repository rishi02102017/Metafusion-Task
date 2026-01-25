#!/usr/bin/env python3
"""
PersonVLM Demo Script
Demonstrates model inference on test images with comparison to ground truth.

Usage:
    python3 demo.py                    # Run demo on random test samples
    python3 demo.py --num_samples 10   # Specify number of samples
    python3 demo.py --save_html        # Save results as HTML report
"""

import sys
import os
import json
import random
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '.')

import torch
from PIL import Image
from torchvision import transforms

from models import PersonVLM
from data.vocabulary import PersonVocabulary


def load_model_and_vocab():
    """Load the trained model and vocabulary."""
    print("Loading vocabulary...")
    vocab = PersonVocabulary.load('data/vocabulary.json')
    
    print("Loading model...")
    model = PersonVLM.from_pretrained('checkpoints/best_model.pt', tokenizer=vocab)
    
    device = torch.device('mps' if torch.backends.mps.is_available() 
                          else 'cuda' if torch.cuda.is_available() 
                          else 'cpu')
    model = model.to(device)
    model.eval()
    
    return model, vocab, device


def load_test_data(test_file='PERSON_DATA/caption_with_attribute_labels/val.jsonl'):
    """Load test data with ground truth captions."""
    samples = []
    with open(test_file, 'r') as f:
        for line in f:
            sample = json.loads(line)
            samples.append({
                'image': sample.get('image', ''),
                'ground_truth': sample.get('answer', ''),
            })
    return samples


def get_image_transform():
    """Get the image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def run_inference(model, image_path, transform, device):
    """Run inference on a single image."""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        description = model.generate(
            image_tensor,
            max_length=100,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
        )[0]
    
    return description


def truncate_text(text, max_words=50):
    """Truncate text to max words for display."""
    words = text.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words]) + '...'
    return text


def print_results(results):
    """Print results to console in a nice format."""
    print("\n" + "=" * 80)
    print("PERSONVLM INFERENCE RESULTS")
    print("=" * 80)
    
    for i, r in enumerate(results, 1):
        print(f"\n[Sample {i}] {r['image_name']}")
        print("-" * 60)
        print(f"GENERATED: {truncate_text(r['generated'], 40)}")
        print(f"GROUND TRUTH: {truncate_text(r['ground_truth'], 40)}")
    
    print("\n" + "=" * 80)


def generate_html_report(results, output_path='demo_results.html'):
    """Generate an HTML report with images and descriptions."""
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PersonVLM Demo Results</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            padding: 20px;
            line-height: 1.6;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { 
            text-align: center; 
            margin-bottom: 10px;
            color: #58a6ff;
        }
        .subtitle {
            text-align: center;
            color: #8b949e;
            margin-bottom: 30px;
        }
        .stats {
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        .stat-box {
            background: #161b22;
            padding: 15px 25px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #30363d;
        }
        .stat-value { font-size: 24px; font-weight: bold; color: #58a6ff; }
        .stat-label { font-size: 12px; color: #8b949e; }
        .sample {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            margin-bottom: 20px;
            overflow: hidden;
        }
        .sample-header {
            background: #21262d;
            padding: 12px 20px;
            border-bottom: 1px solid #30363d;
            font-weight: 600;
        }
        .sample-content {
            display: flex;
            gap: 20px;
            padding: 20px;
        }
        .image-container {
            flex-shrink: 0;
        }
        .image-container img {
            width: 200px;
            height: 200px;
            object-fit: cover;
            border-radius: 8px;
            border: 2px solid #30363d;
        }
        .descriptions {
            flex-grow: 1;
        }
        .desc-section {
            margin-bottom: 15px;
        }
        .desc-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #8b949e;
            margin-bottom: 5px;
        }
        .desc-text {
            background: #0d1117;
            padding: 12px;
            border-radius: 6px;
            font-size: 14px;
            border: 1px solid #30363d;
        }
        .generated .desc-text {
            border-left: 3px solid #238636;
        }
        .ground-truth .desc-text {
            border-left: 3px solid #58a6ff;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            color: #8b949e;
            font-size: 12px;
        }
        @media (max-width: 600px) {
            .sample-content { flex-direction: column; }
            .image-container img { width: 100%; height: auto; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>PersonVLM Demo Results</h1>
        <p class="subtitle">Vision-Language Model for Person Description Generation</p>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-value">7.26M</div>
                <div class="stat-label">Parameters</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">""" + str(len(results)) + """</div>
                <div class="stat-label">Test Samples</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">~100ms</div>
                <div class="stat-label">Inference Time</div>
            </div>
        </div>
"""
    
    for i, r in enumerate(results, 1):
        # Get relative image path for HTML
        img_path = f"PERSON_DATA/images/{r['image_name']}"
        
        html_content += f"""
        <div class="sample">
            <div class="sample-header">Sample {i}: {r['image_name']}</div>
            <div class="sample-content">
                <div class="image-container">
                    <img src="{img_path}" alt="Person {i}" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22200%22 height=%22200%22><rect fill=%22%23161b22%22 width=%22200%22 height=%22200%22/><text fill=%22%238b949e%22 x=%2250%%22 y=%2250%%22 text-anchor=%22middle%22>Image</text></svg>'">
                </div>
                <div class="descriptions">
                    <div class="desc-section generated">
                        <div class="desc-label">Model Generated</div>
                        <div class="desc-text">{r['generated']}</div>
                    </div>
                    <div class="desc-section ground-truth">
                        <div class="desc-label">Ground Truth</div>
                        <div class="desc-text">{truncate_text(r['ground_truth'], 80)}</div>
                    </div>
                </div>
            </div>
        </div>
"""
    
    html_content += f"""
        <div class="footer">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | PersonVLM Demo
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"\nHTML report saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='PersonVLM Demo')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to test')
    parser.add_argument('--save_html', action='store_true', help='Save results as HTML report')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("=" * 60)
    print("PersonVLM Demo")
    print("=" * 60)
    
    # Load model
    model, vocab, device = load_model_and_vocab()
    print(f"Device: {device}")
    print(f"Vocabulary size: {len(vocab)}")
    
    # Load test data
    test_samples = load_test_data()
    print(f"Total test samples: {len(test_samples)}")
    
    # Select random samples
    selected = random.sample(test_samples, min(args.num_samples, len(test_samples)))
    
    # Get transform
    transform = get_image_transform()
    
    # Run inference
    results = []
    image_dir = 'PERSON_DATA/images'
    
    print(f"\nRunning inference on {len(selected)} samples...")
    
    for i, sample in enumerate(selected, 1):
        image_name = os.path.basename(sample['image'])
        image_path = os.path.join(image_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"  [{i}] Image not found: {image_name}")
            continue
        
        generated = run_inference(model, image_path, transform, device)
        
        results.append({
            'image_name': image_name,
            'image_path': image_path,
            'generated': generated,
            'ground_truth': sample['ground_truth'],
        })
        
        print(f"  [{i}] Processed: {image_name}")
    
    # Print results
    print_results(results)
    
    # Save HTML if requested
    if args.save_html:
        generate_html_report(results)
    
    print("\nDemo complete!")
    
    return results


if __name__ == '__main__':
    main()

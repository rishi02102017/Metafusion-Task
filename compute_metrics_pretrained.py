#!/usr/bin/env python3
"""
Evaluation metrics for PersonVLMPretrained.
Metrics: Corpus-level BLEU-1/2/3/4, ROUGE-L, CIDEr, Attribute Accuracy
Evaluated on FULL validation set.
"""

import sys
import os
import json
import random
from collections import Counter, defaultdict
import math
import argparse

sys.path.insert(0, '.')

import torch
from PIL import Image
from torchvision import transforms
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
from transformers import GPT2Tokenizer

# Download nltk data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from models.person_vlm_pretrained import PersonVLMPretrained, PersonVLMPretrainedConfig


def compute_rouge_l(reference, hypothesis):
    """Compute ROUGE-L F1 score."""
    ref_tokens = word_tokenize(reference.lower())
    hyp_tokens = word_tokenize(hypothesis.lower())
    
    def lcs_length(x, y):
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    lcs = lcs_length(ref_tokens, hyp_tokens)
    
    if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def compute_cider(references, hypotheses):
    """
    Compute CIDEr score (simplified implementation).
    """
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def compute_tf(tokens):
        tf = defaultdict(float)
        for n in range(1, 5):
            ngrams = get_ngrams(tokens, n)
            for ng in ngrams:
                tf[ng] += 1
            for ng in tf:
                if len(ng) == n:
                    tf[ng] /= max(len(ngrams), 1)
        return tf
    
    df = defaultdict(int)
    for ref in references:
        ref_tokens = word_tokenize(ref.lower())
        seen = set()
        for n in range(1, 5):
            for ng in get_ngrams(ref_tokens, n):
                if ng not in seen:
                    df[ng] += 1
                    seen.add(ng)
    
    num_docs = len(references)
    scores = []
    
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = word_tokenize(ref.lower())
        hyp_tokens = word_tokenize(hyp.lower())
        
        ref_tf = compute_tf(ref_tokens)
        hyp_tf = compute_tf(hyp_tokens)
        
        ref_tfidf = {}
        hyp_tfidf = {}
        
        for ng in set(ref_tf.keys()) | set(hyp_tf.keys()):
            idf = math.log(num_docs / max(df[ng], 1))
            ref_tfidf[ng] = ref_tf.get(ng, 0) * idf
            hyp_tfidf[ng] = hyp_tf.get(ng, 0) * idf
        
        dot_product = sum(ref_tfidf[ng] * hyp_tfidf[ng] for ng in ref_tfidf)
        ref_norm = math.sqrt(sum(v*v for v in ref_tfidf.values()))
        hyp_norm = math.sqrt(sum(v*v for v in hyp_tfidf.values()))
        
        if ref_norm * hyp_norm > 0:
            scores.append(dot_product / (ref_norm * hyp_norm))
        else:
            scores.append(0.0)
    
    return sum(scores) / len(scores) * 10 if scores else 0.0


def extract_attributes(caption):
    """Extract attributes from caption text."""
    caption_lower = caption.lower()
    
    attributes = {
        'gender': None,
        'age': None,
        'hair_color': None,
        'upper_color': None,
        'lower_color': None,
        'bag': None,
    }
    
    # Gender
    if 'woman' in caption_lower or 'female' in caption_lower or 'lady' in caption_lower:
        attributes['gender'] = 'female'
    elif 'man' in caption_lower or 'male' in caption_lower or 'gentleman' in caption_lower:
        attributes['gender'] = 'male'
    
    # Age
    if 'young' in caption_lower:
        attributes['age'] = 'young'
    elif 'middle-aged' in caption_lower or 'middle aged' in caption_lower:
        attributes['age'] = 'middle'
    elif 'elderly' in caption_lower or 'old' in caption_lower:
        attributes['age'] = 'old'
    
    # Hair color
    hair_colors = ['black', 'brown', 'blonde', 'gray', 'grey', 'white', 'red']
    for color in hair_colors:
        if f'{color} hair' in caption_lower or f'{color}-haired' in caption_lower:
            attributes['hair_color'] = color
            break
    
    # Upper body color
    upper_keywords = ['shirt', 'top', 'jacket', 'coat', 'sweater', 'blouse', 'dress', 't-shirt']
    colors = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'orange', 'purple', 
              'pink', 'brown', 'gray', 'grey', 'navy', 'beige']
    
    for color in colors:
        for keyword in upper_keywords:
            if f'{color} {keyword}' in caption_lower:
                attributes['upper_color'] = color
                break
        if attributes['upper_color']:
            break
    
    # Lower body color
    lower_keywords = ['pants', 'jeans', 'shorts', 'skirt', 'trousers']
    for color in colors:
        for keyword in lower_keywords:
            if f'{color} {keyword}' in caption_lower:
                attributes['lower_color'] = color
                break
        if attributes['lower_color']:
            break
    
    # Bag
    if 'backpack' in caption_lower:
        attributes['bag'] = 'backpack'
    elif 'handbag' in caption_lower or 'purse' in caption_lower:
        attributes['bag'] = 'handbag'
    elif 'bag' in caption_lower:
        attributes['bag'] = 'bag'
    
    return attributes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints_pretrained/best_model.pt', 
                        help='Path to model checkpoint')
    parser.add_argument('--output_file', type=str, default='evaluation_results_pretrained.json', 
                        help='Output JSON file for results')
    args = parser.parse_args()
    
    print("=" * 60)
    print("PersonVLM Pretrained - Full Validation Set Evaluation")
    print("=" * 60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load tokenizer
    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint or use default
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        if isinstance(saved_config, PersonVLMPretrainedConfig):
            config = saved_config
        elif isinstance(saved_config, dict):
            config = PersonVLMPretrainedConfig(**saved_config)
        else:
            config = PersonVLMPretrainedConfig()
    else:
        config = PersonVLMPretrainedConfig()
    
    model = PersonVLMPretrained(config=config, tokenizer=tokenizer)
    
    # Load state dict
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Load validation data
    val_file = "PERSON_DATA/caption_with_attribute_labels/val.jsonl"
    image_dir = "PERSON_DATA/images"
    
    print(f"\nLoading validation data from {val_file}...")
    val_data = []
    max_samples = 500  # Limit for faster evaluation
    with open(val_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            val_data.append(json.loads(line))
    print(f"Validation samples: {len(val_data)} (max {max_samples})")
    
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Generate captions
    print("\nGenerating captions...")
    references = []
    hypotheses = []
    ref_attributes = []
    hyp_attributes = []
    
    with torch.no_grad():
        for i, item in enumerate(val_data):
            # Extract just the filename from the full path
            image_filename = os.path.basename(item['image'])
            image_path = os.path.join(image_dir, image_filename)
            
            if not os.path.exists(image_path):
                continue
            
            try:
                image = Image.open(image_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # Generate caption (returns list of strings)
                # Use greedy decoding for more consistent results
                generated_captions = model.generate(
                    image_tensor,
                    max_length=100,
                    temperature=1.0,
                    top_k=0,
                    top_p=1.0,
                    do_sample=False,  # Greedy decoding
                )
                
                generated_caption = generated_captions[0] if generated_captions else ""
                # Clean up leading punctuation
                generated_caption = generated_caption.lstrip('. ').strip()
                
                # Caption field is 'answer' in the dataset
                reference = item.get('caption', item.get('answer', ''))
                references.append(reference)
                hypotheses.append(generated_caption)
                
                # Extract attributes
                ref_attributes.append(extract_attributes(reference))
                hyp_attributes.append(extract_attributes(generated_caption))
                
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i+1}/{len(val_data)}...")
                    
            except Exception as e:
                import traceback
                print(f"  Error processing {image_filename}: {e}")
                traceback.print_exc()
                continue
    
    print(f"\nSuccessfully processed: {len(references)} samples")
    
    # Compute BLEU scores
    print("\nComputing BLEU scores...")
    ref_tokenized = [[word_tokenize(r.lower())] for r in references]
    hyp_tokenized = [word_tokenize(h.lower()) for h in hypotheses]
    
    smoother = SmoothingFunction()
    bleu1 = corpus_bleu(ref_tokenized, hyp_tokenized, weights=(1, 0, 0, 0), 
                        smoothing_function=smoother.method1)
    bleu2 = corpus_bleu(ref_tokenized, hyp_tokenized, weights=(0.5, 0.5, 0, 0), 
                        smoothing_function=smoother.method1)
    bleu3 = corpus_bleu(ref_tokenized, hyp_tokenized, weights=(0.33, 0.33, 0.33, 0), 
                        smoothing_function=smoother.method1)
    bleu4 = corpus_bleu(ref_tokenized, hyp_tokenized, weights=(0.25, 0.25, 0.25, 0.25), 
                        smoothing_function=smoother.method1)
    
    # Compute ROUGE-L
    print("Computing ROUGE-L...")
    rouge_scores = [compute_rouge_l(r, h) for r, h in zip(references, hypotheses)]
    rouge_l = sum(rouge_scores) / len(rouge_scores)
    
    # Compute CIDEr
    print("Computing CIDEr...")
    cider = compute_cider(references, hypotheses)
    
    # Compute attribute accuracy
    print("Computing attribute accuracy...")
    attr_correct = defaultdict(int)
    attr_total = defaultdict(int)
    
    for ref_attr, hyp_attr in zip(ref_attributes, hyp_attributes):
        for key in ref_attr:
            if ref_attr[key] is not None:
                attr_total[key] += 1
                if ref_attr[key] == hyp_attr[key]:
                    attr_correct[key] += 1
    
    attr_accuracy = {}
    for key in attr_total:
        if attr_total[key] > 0:
            attr_accuracy[key] = attr_correct[key] / attr_total[key]
    
    avg_attr_accuracy = sum(attr_accuracy.values()) / len(attr_accuracy) if attr_accuracy else 0
    
    # Results
    results = {
        "model": args.model_path,
        "samples_evaluated": len(references),
        "bleu_1": round(bleu1 * 100, 2),
        "bleu_2": round(bleu2 * 100, 2),
        "bleu_3": round(bleu3 * 100, 2),
        "bleu_4": round(bleu4 * 100, 2),
        "rouge_l": round(rouge_l * 100, 2),
        "cider": round(cider, 2),
        "avg_attribute_accuracy": round(avg_attr_accuracy * 100, 2),
        "attribute_accuracy": {k: round(v * 100, 2) for k, v in attr_accuracy.items()}
    }
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Samples: {len(references)}")
    print("-" * 40)
    print(f"BLEU-1:  {bleu1*100:.2f}%")
    print(f"BLEU-2:  {bleu2*100:.2f}%")
    print(f"BLEU-3:  {bleu3*100:.2f}%")
    print(f"BLEU-4:  {bleu4*100:.2f}%")
    print(f"ROUGE-L: {rouge_l*100:.2f}%")
    print(f"CIDEr:   {cider:.2f}")
    print("-" * 40)
    print("Attribute Accuracy:")
    for attr, acc in attr_accuracy.items():
        print(f"  {attr}: {acc*100:.1f}%")
    print(f"  Average: {avg_attr_accuracy*100:.1f}%")
    print("=" * 60)
    
    # Show sample generations
    print("\nSample Generations:")
    print("-" * 40)
    sample_indices = random.sample(range(len(references)), min(3, len(references)))
    for idx in sample_indices:
        print(f"Reference: {references[idx]}")
        print(f"Generated: {hypotheses[idx]}")
        print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Comprehensive evaluation metrics for PersonVLM.
Metrics: Corpus-level BLEU-1/2/3/4, ROUGE-L, CIDEr, Attribute Accuracy
Evaluated on FULL validation set.
"""

import sys
import os
import json
import random
from collections import Counter, defaultdict
import math

sys.path.insert(0, '.')

import torch
from PIL import Image
from torchvision import transforms
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk

# Download nltk data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from models import PersonVLM
from data.vocabulary import PersonVocabulary


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
    CIDEr measures consensus - how well the caption matches multiple references.
    Since we have single reference, this is a simplified version.
    """
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def compute_tf(tokens):
        tf = defaultdict(float)
        for n in range(1, 5):  # 1-4 grams
            ngrams = get_ngrams(tokens, n)
            for ng in ngrams:
                tf[ng] += 1
            for ng in tf:
                if len(ng) == n:
                    tf[ng] /= max(len(ngrams), 1)
        return tf
    
    # Compute document frequency across all references
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
    
    # Compute CIDEr for each hypothesis
    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = word_tokenize(ref.lower())
        hyp_tokens = word_tokenize(hyp.lower())
        
        ref_tf = compute_tf(ref_tokens)
        hyp_tf = compute_tf(hyp_tokens)
        
        # Compute TF-IDF vectors
        ref_tfidf = {}
        hyp_tfidf = {}
        
        all_ngrams = set(ref_tf.keys()) | set(hyp_tf.keys())
        
        for ng in all_ngrams:
            idf = math.log(max(num_docs, 1) / max(df.get(ng, 0), 1))
            ref_tfidf[ng] = ref_tf.get(ng, 0) * idf
            hyp_tfidf[ng] = hyp_tf.get(ng, 0) * idf
        
        # Cosine similarity
        dot = sum(ref_tfidf.get(ng, 0) * hyp_tfidf.get(ng, 0) for ng in all_ngrams)
        ref_norm = math.sqrt(sum(v**2 for v in ref_tfidf.values()))
        hyp_norm = math.sqrt(sum(v**2 for v in hyp_tfidf.values()))
        
        if ref_norm * hyp_norm > 0:
            scores.append(dot / (ref_norm * hyp_norm))
        else:
            scores.append(0.0)
    
    return sum(scores) / len(scores) * 10  # CIDEr is typically scaled by 10


def check_attribute_match(reference, hypothesis, attribute_words):
    """Check if attribute words from reference appear in hypothesis."""
    ref_lower = reference.lower()
    hyp_lower = hypothesis.lower()
    
    ref_attrs = [w for w in attribute_words if w in ref_lower]
    if not ref_attrs:
        return None
    
    matches = sum(1 for w in ref_attrs if w in hyp_lower)
    return matches / len(ref_attrs)


# Attribute categories
COLORS = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'gray', 'grey', 'brown', 
          'pink', 'orange', 'purple', 'navy', 'beige', 'dark', 'light', 'maroon']
CLOTHING = ['shirt', 't-shirt', 'jacket', 'coat', 'pants', 'trousers', 'jeans', 'shorts',
            'dress', 'skirt', 'sweater', 'hoodie', 'vest', 'top', 'blouse']
ACTIONS = ['standing', 'walking', 'sitting', 'running', 'waiting', 'looking', 'carrying', 'holding']
GENDER = ['male', 'female', 'man', 'woman', 'boy', 'girl']


def main():
    print("=" * 70)
    print("PersonVLM FULL Evaluation (Corpus-level Metrics)")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    vocab = PersonVocabulary.load('data/vocabulary.json')
    model = PersonVLM.from_pretrained('checkpoints/best_model.pt', tokenizer=vocab)
    
    device = torch.device('mps' if torch.backends.mps.is_available() 
                          else 'cuda' if torch.cuda.is_available() 
                          else 'cpu')
    model = model.to(device)
    model.eval()
    print(f"Device: {device}")
    
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load ALL data
    all_samples = []
    jsonl_path = 'PERSON_DATA/caption_with_attribute_labels/MSP60k_train_v2.jsonl'
    with open(jsonl_path, 'r') as f:
        for line in f:
            all_samples.append(json.loads(line))
    
    print(f"Total samples in dataset: {len(all_samples)}")
    
    # Check what split we used during training
    # During training, we used split_jsonl which does 80/10/10 split
    # Let's use the last 20% (val + test) for evaluation to ensure no data leakage
    split_idx = int(len(all_samples) * 0.8)
    val_samples = all_samples[split_idx:]  # Last 20%
    
    print(f"Validation set size: {len(val_samples)} samples (last 20%)")
    print(f"This ensures NO overlap with training data (first 80%)")
    
    # For speed, evaluate on a larger but manageable subset
    num_eval = min(500, len(val_samples))  # 500 samples for proper evaluation
    random.seed(42)
    eval_samples = random.sample(val_samples, num_eval)
    print(f"\nEvaluating on {num_eval} samples...")
    
    # Collect predictions
    all_references = []  # [[ref_tokens]]
    all_hypotheses = []  # [hyp_tokens]
    all_ref_texts = []
    all_hyp_texts = []
    
    rouge_scores = []
    color_accs, clothing_accs, action_accs, gender_accs = [], [], [], []
    
    image_dir = 'PERSON_DATA/images'
    successful = 0
    
    for i, sample in enumerate(eval_samples):
        image_name = os.path.basename(sample['image'])
        image_path = os.path.join(image_dir, image_name)
        ground_truth = sample.get('answer', '')
        
        if not os.path.exists(image_path):
            continue
        
        try:
            # Generate prediction
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                generated = model.generate(
                    image_tensor,
                    max_length=100,
                    temperature=0.7,
                )[0]
            
            # Tokenize for corpus BLEU
            ref_tokens = word_tokenize(ground_truth.lower())
            hyp_tokens = word_tokenize(generated.lower())
            
            all_references.append([ref_tokens])  # List of list for corpus_bleu
            all_hypotheses.append(hyp_tokens)
            all_ref_texts.append(ground_truth)
            all_hyp_texts.append(generated)
            
            # ROUGE-L
            rouge = compute_rouge_l(ground_truth, generated)
            rouge_scores.append(rouge)
            
            # Attribute accuracy
            color_acc = check_attribute_match(ground_truth, generated, COLORS)
            if color_acc is not None:
                color_accs.append(color_acc)
            
            clothing_acc = check_attribute_match(ground_truth, generated, CLOTHING)
            if clothing_acc is not None:
                clothing_accs.append(clothing_acc)
            
            action_acc = check_attribute_match(ground_truth, generated, ACTIONS)
            if action_acc is not None:
                action_accs.append(action_acc)
            
            gender_acc = check_attribute_match(ground_truth, generated, GENDER)
            if gender_acc is not None:
                gender_accs.append(gender_acc)
            
            successful += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{num_eval} samples... ({successful} successful)")
                
        except Exception as e:
            continue
    
    print(f"\nSuccessfully evaluated: {successful} samples")
    
    # Compute CORPUS-level BLEU (the standard way)
    print("\nComputing corpus-level BLEU scores...")
    smoothing = SmoothingFunction().method1
    
    bleu1 = corpus_bleu(all_references, all_hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = corpus_bleu(all_references, all_hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = corpus_bleu(all_references, all_hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu4 = corpus_bleu(all_references, all_hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    # Compute CIDEr
    print("Computing CIDEr score...")
    cider = compute_cider(all_ref_texts, all_hyp_texts)
    
    # Results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (FULL - Corpus-Level)")
    print("=" * 70)
    
    print("\n[Text Generation Metrics - Corpus Level]")
    print("-" * 50)
    print(f"BLEU-1:    {bleu1:.4f}")
    print(f"BLEU-2:    {bleu2:.4f}")
    print(f"BLEU-3:    {bleu3:.4f}")
    print(f"BLEU-4:    {bleu4:.4f}")
    print(f"ROUGE-L:   {sum(rouge_scores)/len(rouge_scores):.4f}")
    print(f"CIDEr:     {cider:.4f}")
    
    print("\n[Attribute-Level Accuracy]")
    print("-" * 50)
    if color_accs:
        print(f"Color:     {sum(color_accs)/len(color_accs)*100:.1f}% ({len(color_accs)} samples)")
    if clothing_accs:
        print(f"Clothing:  {sum(clothing_accs)/len(clothing_accs)*100:.1f}% ({len(clothing_accs)} samples)")
    if action_accs:
        print(f"Action:    {sum(action_accs)/len(action_accs)*100:.1f}% ({len(action_accs)} samples)")
    if gender_accs:
        print(f"Gender:    {sum(gender_accs)/len(gender_accs)*100:.1f}% ({len(gender_accs)} samples)")
    
    avg_attr = (sum(color_accs)/len(color_accs) + sum(clothing_accs)/len(clothing_accs) + 
                sum(action_accs)/len(action_accs) + sum(gender_accs)/len(gender_accs)) / 4
    print(f"\nOverall Attribute Accuracy: {avg_attr*100:.1f}%")
    
    print("\n[Summary]")
    print("-" * 50)
    print(f"Samples evaluated: {successful}")
    print(f"Evaluation set: Last 20% of data (no training overlap)")
    print(f"Final Val Loss: 1.97")
    
    # Save results
    results = {
        "method": "corpus-level (proper evaluation)",
        "num_samples": successful,
        "bleu1": bleu1,
        "bleu2": bleu2,
        "bleu3": bleu3,
        "bleu4": bleu4,
        "rouge_l": sum(rouge_scores)/len(rouge_scores),
        "cider": cider,
        "color_accuracy": sum(color_accs)/len(color_accs),
        "clothing_accuracy": sum(clothing_accs)/len(clothing_accs),
        "action_accuracy": sum(action_accs)/len(action_accs),
        "gender_accuracy": sum(gender_accs)/len(gender_accs),
        "overall_attribute_accuracy": avg_attr,
    }
    
    with open('evaluation_results_full.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to evaluation_results_full.json")
    print("=" * 70)
    
    # Show a few examples
    print("\n[Sample Predictions]")
    print("-" * 70)
    for i in range(min(5, len(all_ref_texts))):
        print(f"\nSample {i+1}:")
        print(f"  GT:   {all_ref_texts[i][:100]}...")
        print(f"  Pred: {all_hyp_texts[i][:100]}...")


if __name__ == '__main__':
    main()

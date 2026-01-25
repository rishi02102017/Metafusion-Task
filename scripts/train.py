#!/usr/bin/env python3
"""
Training Script for Person VLM
==============================
Train the lightweight VLM on person blob images with captions

Usage:
    python train.py --train_file data/train.json --val_file data/val.json
    python train.py --config configs/config.yaml
"""

import os
import sys
import argparse
import yaml
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

from models import PersonVLM, PersonVLMConfig
from data import PersonVocabulary, create_dataloaders
from training import Trainer, TrainingConfig


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Train Person VLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python train.py --train_file data/train.json --val_file data/val.json
  
  # Train with YAML config
  python train.py --config configs/config.yaml
  
  # Override config parameters
  python train.py --config configs/config.yaml --epochs 30 --batch_size 64
  
  # Resume training
  python train.py --config configs/config.yaml --resume checkpoints/checkpoint_epoch_10.pt
        """,
    )
    
    # Config
    parser.add_argument(
        "--config", type=str,
        help="Path to YAML config file"
    )
    
    # Data (can override config)
    parser.add_argument("--train_file", type=str, help="Training data JSON")
    parser.add_argument("--val_file", type=str, help="Validation data JSON")
    parser.add_argument("--vocab_file", type=str, help="Vocabulary JSON file")
    
    # Model (can override config)
    parser.add_argument(
        "--vision_backbone", type=str,
        choices=["mobilevit_xxs", "mobilevit_xs", "efficientnet_lite0", "vit_tiny_patch16_224"],
        help="Vision encoder backbone"
    )
    parser.add_argument(
        "--decoder_size", type=str,
        choices=["tiny", "small", "medium"],
        help="Text decoder size"
    )
    parser.add_argument(
        "--freeze_ratio", type=float,
        help="Fraction of vision encoder to freeze"
    )
    
    # Training (can override config)
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    
    # Hardware
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (auto detects best available)"
    )
    
    # Image directory
    parser.add_argument(
        "--image_dir", type=str,
        help="Directory containing person images"
    )
    parser.add_argument(
        "--no_amp", action="store_true",
        help="Disable mixed precision training"
    )
    
    # Resume
    parser.add_argument(
        "--resume", type=str,
        help="Resume from checkpoint"
    )
    
    args = parser.parse_args()
    
    # Load base config
    if args.config:
        config_dict = load_config(args.config)
    else:
        config_dict = {}
    
    # Build training config with overrides
    training_config = TrainingConfig(
        # Data
        train_file=args.train_file or config_dict.get("data", {}).get("train_file", ""),
        val_file=args.val_file or config_dict.get("data", {}).get("val_file"),
        
        # Model
        vision_backbone=args.vision_backbone or config_dict.get("model", {}).get("vision", {}).get("backbone", "mobilevit_xs"),
        decoder_size=args.decoder_size or config_dict.get("model", {}).get("decoder", {}).get("size", "small"),
        vision_freeze_ratio=args.freeze_ratio or config_dict.get("model", {}).get("vision", {}).get("freeze_ratio", 0.9),
        
        # Training
        epochs=args.epochs or config_dict.get("training", {}).get("epochs", 20),
        batch_size=args.batch_size or config_dict.get("training", {}).get("batch_size", 32),
        learning_rate=args.learning_rate or config_dict.get("training", {}).get("learning_rate", 1e-4),
        weight_decay=config_dict.get("training", {}).get("weight_decay", 0.01),
        warmup_ratio=config_dict.get("training", {}).get("warmup_ratio", 0.1),
        
        # Optimizer & Scheduler
        optimizer=config_dict.get("training", {}).get("optimizer", "adamw"),
        scheduler=config_dict.get("training", {}).get("scheduler", "cosine"),
        
        # Mixed precision
        use_amp=not args.no_amp and config_dict.get("training", {}).get("use_amp", True),
        
        # Regularization
        dropout=config_dict.get("model", {}).get("dropout", 0.1),
        label_smoothing=config_dict.get("model", {}).get("label_smoothing", 0.1),
        gradient_clip=config_dict.get("training", {}).get("gradient_clip", 1.0),
        
        # Data loading
        num_workers=config_dict.get("training", {}).get("num_workers", 4),
        image_size=config_dict.get("training", {}).get("image_size", 224),
        max_seq_length=config_dict.get("training", {}).get("max_seq_length", 64),
        
        # Checkpointing
        output_dir=args.output_dir or config_dict.get("training", {}).get("output_dir", "./checkpoints"),
        save_every=config_dict.get("training", {}).get("save_every", 1),
        
        # Early stopping
        early_stopping=config_dict.get("training", {}).get("early_stopping", True),
        patience=config_dict.get("training", {}).get("patience", 5),
        
        # Logging
        log_every=config_dict.get("training", {}).get("log_every", 50),
        eval_every=config_dict.get("training", {}).get("eval_every", 500),
        
        # Resume
        resume_from=args.resume,
    )
    
    # Validate required arguments
    if not training_config.train_file:
        parser.error("--train_file is required (or specify in config)")
    
    # Create vocabulary
    print("\n" + "=" * 60)
    print("Loading vocabulary...")
    print("=" * 60)
    
    vocab_file = args.vocab_file or config_dict.get("data", {}).get("vocab_file")
    if vocab_file and os.path.exists(vocab_file):
        vocab = PersonVocabulary.load(vocab_file)
    else:
        vocab = PersonVocabulary()
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create dataloaders
    print("\n" + "=" * 60)
    print("Creating dataloaders...")
    print("=" * 60)
    
    # Get image directory
    image_dir = args.image_dir or config_dict.get("data", {}).get("image_dir")
    
    train_loader, val_loader = create_dataloaders(
        train_file=training_config.train_file,
        val_file=training_config.val_file,
        vocab=vocab,
        image_dir=image_dir,
        batch_size=training_config.batch_size,
        image_size=training_config.image_size,
        max_seq_length=training_config.max_seq_length,
        num_workers=training_config.num_workers,
        augment_train=config_dict.get("data", {}).get("augment_train", True),
    )
    
    # Create model
    print("\n" + "=" * 60)
    print("Creating model...")
    print("=" * 60)
    
    model_config = PersonVLMConfig(
        vision_backbone=training_config.vision_backbone,
        vision_freeze_ratio=training_config.vision_freeze_ratio,
        decoder_size=training_config.decoder_size,
        dropout=training_config.dropout,
        label_smoothing=training_config.label_smoothing,
        # Vocabulary settings from loaded vocab
        vocab_size=len(vocab),
        max_seq_length=training_config.max_seq_length,
        pad_token_id=vocab.pad_id,
        bos_token_id=vocab.bos_id,
        eos_token_id=vocab.eos_id,
    )
    model = PersonVLM(model_config, tokenizer=vocab)
    
    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        config=training_config,
    )
    
    history = trainer.train()
    
    print("\nTraining complete!")
    print(f"  Best model: {training_config.output_dir}/best_model.pt")
    print(f"  Final model: {training_config.output_dir}/final_model.pt")
    print(f"  History: {training_config.output_dir}/history.json")


if __name__ == "__main__":
    main()

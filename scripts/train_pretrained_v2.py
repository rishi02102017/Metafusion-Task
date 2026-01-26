#!/usr/bin/env python3
"""
Training script for PersonVLM with Pretrained Decoder - V2 (Fine-tuned)
========================================================================
Key improvements over V1:
- Unfreeze all decoder layers (freeze_decoder_ratio: 0.0)
- Increase visual tokens (8 → 16)
- Lower learning rate for stable fine-tuning
- Longer training (40 epochs)

This represents Stage 4 of our progression:
1. Baseline (7M) → 2. Scaled (33M) → 3. Pretrained (93M frozen) → 4. Fine-tuned (93M unfrozen)
"""

import os
import sys
import json
import math
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from transformers import GPT2Tokenizer, get_cosine_schedule_with_warmup

from models.person_vlm_pretrained import PersonVLMPretrained, PersonVLMPretrainedConfig


@dataclass
class TrainingConfigV2:
    """Optimized training configuration for fine-tuning."""
    # Data
    train_file: str = "PERSON_DATA/caption_with_attribute_labels/train.jsonl"
    val_file: str = "PERSON_DATA/caption_with_attribute_labels/val.jsonl"
    image_dir: str = "PERSON_DATA/images"
    
    # Model - KEY CHANGES FOR V2
    vision_backbone: str = "mobilevit_xs"
    vision_freeze_ratio: float = 0.7  # Unfreeze more vision layers too
    decoder_model: str = "distilgpt2"
    freeze_decoder_ratio: float = 0.0  # UNFREEZE ALL decoder layers
    num_visual_tokens: int = 12  # Increase from 8 to 12 (16 exceeds 100M budget)
    
    # Training - Optimized for fine-tuning
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 2e-5  # Lower LR for unfrozen model
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1  # 10% warmup
    max_seq_length: int = 128
    
    # Mixed precision
    use_amp: bool = True
    
    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0
    
    # Checkpointing - Save to new directory
    output_dir: str = "./checkpoints_pretrained_v2"
    save_every: int = 5
    
    # Early stopping
    patience: int = 10
    
    # Logging
    log_every: int = 50


class PersonDatasetGPT2(Dataset):
    """Dataset that uses GPT-2 tokenizer."""
    
    def __init__(
        self,
        data_file: str,
        image_dir: str,
        tokenizer: GPT2Tokenizer,
        max_length: int = 128,
        image_size: int = 224,
        augment: bool = False,
    ):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        self.data = []
        with open(data_file, 'r') as f:
            for line in f:
                if line:
                    try:
                        item = json.loads(line)
                        if "answer" in item and item["answer"]:
                            self.data.append({
                                "image": os.path.basename(item["image"]),
                                "caption": item["answer"],
                            })
                    except:
                        continue
        
        print(f"Loaded {len(self.data)} samples from {data_file}")
        
        # Image transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, item["image"])
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224), (128, 128, 128))
        image = self.transform(image)
        
        # Tokenize caption
        caption = item["caption"]
        
        encoded = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = encoded.input_ids.squeeze(0)
        attention_mask = encoded.attention_mask.squeeze(0)
        
        # Labels: same as input_ids, but with padding tokens set to -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            "image": image,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def train_epoch(model, train_loader, optimizer, scheduler, scaler, config, device, epoch, is_main):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not is_main)
    
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast(enabled=config.use_amp):
            outputs = model(
                images=images,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
            loss = outputs["loss"]
        
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if is_main and batch_idx % config.log_every == 0:
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")
    
    return total_loss / num_batches


def validate(model, val_loader, device, is_main):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            
            outputs = model(
                images=images,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
            
            total_loss += outputs["loss"].item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_pretrained_v2")
    parser.add_argument("--num_visual_tokens", type=int, default=16)
    parser.add_argument("--freeze_decoder_ratio", type=float, default=0.0)
    args = parser.parse_args()
    
    # Setup
    is_distributed, rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Config
    config = TrainingConfigV2()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.output_dir = args.output_dir
    
    if is_main:
        os.makedirs(config.output_dir, exist_ok=True)
        print("\n" + "=" * 60)
        print("PersonVLM Pretrained V2 - Fine-tuned Training")
        print("=" * 60)
        print(f"Key improvements:")
        print(f"  - Decoder freeze ratio: {args.freeze_decoder_ratio} (unfrozen)")
        print(f"  - Visual tokens: {args.num_visual_tokens}")
        print(f"  - Epochs: {config.epochs}")
        print(f"  - Learning rate: {config.learning_rate}")
        print(f"  - Output: {config.output_dir}")
        print("=" * 60 + "\n")
    
    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Model config with V2 improvements
    model_config = PersonVLMPretrainedConfig(
        vision_backbone=config.vision_backbone,
        vision_pretrained=True,  # Explicitly set
        vision_freeze_ratio=config.vision_freeze_ratio,
        decoder_model=config.decoder_model,
        freeze_decoder_ratio=args.freeze_decoder_ratio,
        num_visual_tokens=args.num_visual_tokens,
        max_seq_length=config.max_seq_length,
        dropout=config.dropout,
        label_smoothing=config.label_smoothing,
    )
    
    # Create model
    model = PersonVLMPretrained(config=model_config, tokenizer=tokenizer)
    model = model.to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    # Datasets
    train_dataset = PersonDatasetGPT2(
        config.train_file, config.image_dir, tokenizer,
        max_length=config.max_seq_length, augment=True
    )
    val_dataset = PersonDatasetGPT2(
        config.val_file, config.image_dir, tokenizer,
        max_length=config.max_seq_length, augment=False
    )
    
    # DataLoaders
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, sampler=val_sampler,
        num_workers=4, pin_memory=True
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    scaler = GradScaler(enabled=config.use_amp)
    
    if is_main:
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Total steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(config.epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            config, device, epoch, is_main
        )
        val_loss = validate(model, val_loader, device, is_main)
        
        if is_distributed:
            # Average losses across processes
            train_loss_tensor = torch.tensor(train_loss, device=device)
            val_loss_tensor = torch.tensor(val_loss, device=device)
            dist.all_reduce(train_loss_tensor)
            dist.all_reduce(val_loss_tensor)
            train_loss = train_loss_tensor.item() / world_size
            val_loss = val_loss_tensor.item() / world_size
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        if is_main:
            print(f"\nEpoch {epoch+1}/{config.epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            
            # Get the model (unwrap DDP if needed)
            save_model = model.module if is_distributed else model
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_model.save_pretrained(os.path.join(config.output_dir, "best_model.pt"))
                print("  New best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"\nEarly stopping after {epoch+1} epochs (no improvement for {config.patience} epochs)")
                    break
            
            # Save periodic checkpoints
            if (epoch + 1) % config.save_every == 0:
                save_model.save_pretrained(
                    os.path.join(config.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
                )
            
            # Save history
            with open(os.path.join(config.output_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)
    
    # Save final model
    if is_main:
        save_model = model.module if is_distributed else model
        save_model.save_pretrained(os.path.join(config.output_dir, "final_model.pt"))
        
        # Save config
        config_dict = {
            "vision_backbone": model_config.vision_backbone,
            "vision_freeze_ratio": model_config.vision_freeze_ratio,
            "decoder_model": model_config.decoder_model,
            "freeze_decoder_ratio": model_config.freeze_decoder_ratio,
            "num_visual_tokens": model_config.num_visual_tokens,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
        }
        with open(os.path.join(config.output_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"Checkpoints: {config.output_dir}")
        print("=" * 60)
    
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

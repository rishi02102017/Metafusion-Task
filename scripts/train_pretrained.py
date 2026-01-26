#!/usr/bin/env python3
"""
Training Script for PersonVLM with Pretrained Decoder (DistilGPT-2)
===================================================================
Fine-tunes the model on person description task.

Usage:
    python scripts/train_pretrained.py --epochs 20 --batch_size 32
    
Multi-GPU:
    torchrun --nproc_per_node=4 scripts/train_pretrained.py --epochs 20
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import GPT2Tokenizer
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from models.person_vlm_pretrained import PersonVLMPretrained, PersonVLMPretrainedConfig


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data
    train_file: str = "PERSON_DATA/caption_with_attribute_labels/train.jsonl"
    val_file: str = "PERSON_DATA/caption_with_attribute_labels/val.jsonl"
    image_dir: str = "PERSON_DATA/images"
    
    # Model
    vision_backbone: str = "mobilevit_xs"
    vision_freeze_ratio: float = 0.8
    decoder_model: str = "distilgpt2"
    freeze_decoder_ratio: float = 0.5  # Freeze bottom 50% of GPT-2 layers
    
    # Training
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_seq_length: int = 128
    
    # Mixed precision
    use_amp: bool = True
    
    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0
    
    # Checkpointing
    output_dir: str = "./checkpoints_pretrained"
    save_every: int = 5
    
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
                line = line.strip()
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
        
        # Tokenize caption with GPT-2 tokenizer
        # Add BOS token at start
        caption = item["caption"]
        
        # Encode
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
            outputs = model(images, input_ids, labels, attention_mask)
            loss = outputs["loss"]
        
        if config.use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if is_main and batch_idx % config.log_every == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
    
    return total_loss / num_batches


@torch.no_grad()
def validate(model, val_loader, config, device):
    """Run validation."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for batch in val_loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        with autocast(enabled=config.use_amp):
            outputs = model(images, input_ids, labels, attention_mask)
            loss = outputs["loss"]
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_pretrained")
    parser.add_argument("--freeze_decoder_ratio", type=float, default=0.5)
    args = parser.parse_args()
    
    # Setup distributed
    distributed, rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    
    # Device
    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    if is_main:
        print("=" * 60)
        print("PersonVLM with Pretrained Decoder Training")
        print("=" * 60)
        if distributed:
            print(f"Distributed training with {world_size} GPUs")
        print(f"Device: {device}")
    
    # Config
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        freeze_decoder_ratio=args.freeze_decoder_ratio,
    )
    
    # Create output dir
    if is_main:
        os.makedirs(config.output_dir, exist_ok=True)
        with open(os.path.join(config.output_dir, "config.json"), "w") as f:
            json.dump(asdict(config), f, indent=2)
    
    # Load GPT-2 tokenizer
    if is_main:
        print("\nLoading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(config.decoder_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    if is_main:
        print("\nCreating datasets...")
    
    train_dataset = PersonDatasetGPT2(
        data_file=config.train_file,
        image_dir=config.image_dir,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        augment=True,
    )
    
    val_dataset = PersonDatasetGPT2(
        data_file=config.val_file,
        image_dir=config.image_dir,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        augment=False,
    )
    
    # Create dataloaders
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create model
    if is_main:
        print("\nCreating model...")
    
    model_config = PersonVLMPretrainedConfig(
        vision_backbone=config.vision_backbone,
        vision_freeze_ratio=config.vision_freeze_ratio,
        decoder_model=config.decoder_model,
        freeze_decoder_ratio=config.freeze_decoder_ratio,
        dropout=config.dropout,
        label_smoothing=config.label_smoothing,
    )
    
    model = PersonVLMPretrained(model_config, tokenizer=tokenizer)
    model = model.to(device)
    
    # Wrap with DDP
    if distributed:
        model = DDP(model, device_ids=[local_rank])
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Scheduler
    total_steps = len(train_loader) * config.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    # Warmup (manual)
    # Skip for simplicity - cosine annealing works well
    
    # Mixed precision
    scaler = GradScaler() if config.use_amp and device.type == "cuda" else None
    if scaler is None:
        config.use_amp = False
    
    # Training loop
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}
    
    if is_main:
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
    
    for epoch in range(config.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            config, device, epoch, is_main
        )
        
        # Gather loss across GPUs
        if distributed:
            train_loss_tensor = torch.tensor([train_loss], device=device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
            train_loss = train_loss_tensor.item()
        
        # Validate
        val_loss = validate(model, val_loader, config, device)
        
        if distributed:
            val_loss_tensor = torch.tensor([val_loss], device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            val_loss = val_loss_tensor.item()
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        if is_main:
            print(f"\nEpoch {epoch+1}/{config.epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_to_save = model.module if distributed else model
                model_to_save.save_pretrained(os.path.join(config.output_dir, "best_model.pt"))
                print(f"  New best model saved!")
            
            # Periodic checkpoint
            if (epoch + 1) % config.save_every == 0:
                model_to_save = model.module if distributed else model
                model_to_save.save_pretrained(
                    os.path.join(config.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
                )
    
    # Save final model and history
    if is_main:
        model_to_save = model.module if distributed else model
        model_to_save.save_pretrained(os.path.join(config.output_dir, "final_model.pt"))
        
        with open(os.path.join(config.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"Checkpoints: {config.output_dir}")
        print("=" * 60)
    
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""
Training Pipeline for Person VLM
"""

import os
import time
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .utils import (
    set_seed,
    get_optimizer,
    get_scheduler,
    AverageMeter,
    EarlyStopping,
    count_parameters,
    save_checkpoint,
    load_checkpoint,
)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data
    train_file: str = ""
    val_file: Optional[str] = None
    
    # Model
    vision_backbone: str = "mobilevit_xs"
    decoder_size: str = "small"
    vision_freeze_ratio: float = 0.9
    
    # Training
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Optimizer & Scheduler
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    
    # Mixed precision
    use_amp: bool = True
    
    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0
    
    # Data loading
    num_workers: int = 4
    image_size: int = 224
    max_seq_length: int = 256  # Accommodate MSP60k detailed captions
    
    # Checkpointing
    output_dir: str = "./checkpoints"
    save_every: int = 1  # Save every N epochs
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 5
    
    # Logging
    log_every: int = 50  # Log every N steps
    eval_every: int = 500  # Eval every N steps
    
    # Reproducibility
    seed: int = 42
    
    # Resume
    resume_from: Optional[str] = None


class Trainer:
    """
    Trainer for Person VLM.
    
    Handles:
    - Training loop with mixed precision
    - Validation
    - Checkpointing
    - Early stopping
    - Logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        vocab,
        config: TrainingConfig,
    ):
        self.config = config
        self.vocab = vocab
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)
        
        # Set seed
        set_seed(config.seed)
        
        # Device selection (CUDA > MPS > CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # Model
        self.model = model.to(self.device)
        
        # Print parameter counts
        param_counts = count_parameters(self.model)
        print(f"Model parameters:")
        print(f"  Total: {param_counts['total']:,}")
        print(f"  Trainable: {param_counts['trainable']:,}")
        print(f"  Frozen: {param_counts['frozen']:,}")
        
        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Calculate training steps
        self.steps_per_epoch = len(train_loader)
        self.total_steps = self.steps_per_epoch * config.epochs
        print(f"Training for {config.epochs} epochs ({self.total_steps} steps)")
        
        # Optimizer
        self.optimizer = get_optimizer(
            model,
            optimizer_type=config.optimizer,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = get_scheduler(
            self.optimizer,
            scheduler_type=config.scheduler,
            num_training_steps=self.total_steps,
            warmup_ratio=config.warmup_ratio,
        )
        
        # Mixed precision (only supported on CUDA)
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        if config.use_amp and not self.use_amp:
            print("Note: Mixed precision (AMP) is only supported on CUDA. Training in fp32.")
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            mode="min",
        ) if config.early_stopping else None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        
        # Resume if specified
        if config.resume_from:
            self._resume_training(config.resume_from)
        
        # Metrics
        self.train_metrics = {}
        self.val_metrics = {}
    
    def _resume_training(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        state = load_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            checkpoint_path,
            device=str(self.device),
        )
        self.epoch = state["epoch"]
        self.global_step = state["step"]
        self.best_val_loss = state.get("loss", float("inf"))
        print(f"Resumed from epoch {self.epoch}, step {self.global_step}")
    
    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.
        
        Returns:
            Training history
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
        }
        
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        
        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch
            
            # Training epoch
            train_loss = self._train_epoch()
            history["train_loss"].append(train_loss)
            
            # Validation
            val_loss = None
            if self.val_loader:
                val_loss = self._validate()
                history["val_loss"].append(val_loss)
                
                # Check for improvement
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_best_model()
            
            # Record learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            history["learning_rates"].append(current_lr)
            
            # Logging
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            if val_loss:
                print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Learning Rate: {current_lr:.2e}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(epoch)
            
            # Early stopping
            if self.early_stopping and val_loss:
                if self.early_stopping(val_loss):
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
        
        # Save final model
        self._save_final_model()
        
        # Save training history
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        print("\n" + "=" * 60)
        print("Training Complete")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.output_dir}")
        print("=" * 60)
        
        return history
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        
        loss_meter = AverageMeter("Loss")
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for batch in pbar:
            # Move to device
            images = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(images, input_ids, labels)
                loss = outputs["loss"]
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )
                
                self.optimizer.step()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Update metrics
            loss_meter.update(loss.item(), images.size(0))
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })
            
            # Mid-epoch validation
            if self.val_loader and self.global_step % self.config.eval_every == 0:
                val_loss = self._validate()
                self.model.train()  # Back to training mode
        
        return loss_meter.avg
    
    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation."""
        self.model.eval()
        
        loss_meter = AverageMeter("Val Loss")
        
        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            images = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(images, input_ids, labels)
                loss = outputs["loss"]
            
            loss_meter.update(loss.item(), images.size(0))
        
        return loss_meter.avg
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        path = self.output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        save_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            epoch,
            self.global_step,
            self.best_val_loss,
            str(path),
        )
    
    def _save_best_model(self):
        """Save best model based on validation loss."""
        path = self.output_dir / "best_model.pt"
        self.model.save_pretrained(str(path))
        print(f"  New best model saved!")
    
    def _save_final_model(self):
        """Save final model."""
        path = self.output_dir / "final_model.pt"
        self.model.save_pretrained(str(path))


def train_person_vlm(config: TrainingConfig):
    """
    Convenience function to train Person VLM from config.
    
    Args:
        config: Training configuration
    """
    # Import here to avoid circular imports
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from models import PersonVLM, PersonVLMConfig
    from data import PersonVocabulary, create_dataloaders
    
    # Create vocabulary
    print("Loading vocabulary...")
    vocab = PersonVocabulary()
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_file=config.train_file,
        val_file=config.val_file,
        vocab=vocab,
        batch_size=config.batch_size,
        image_size=config.image_size,
        max_seq_length=config.max_seq_length,
        num_workers=config.num_workers,
        augment_train=True,
    )
    
    # Create model
    print("\nCreating model...")
    model_config = PersonVLMConfig(
        vision_backbone=config.vision_backbone,
        vision_freeze_ratio=config.vision_freeze_ratio,
        decoder_size=config.decoder_size,
        dropout=config.dropout,
        label_smoothing=config.label_smoothing,
    )
    model = PersonVLM(model_config, tokenizer=vocab)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        config=config,
    )
    
    # Train
    history = trainer.train()
    
    return history


if __name__ == "__main__":
    # Example training script
    print("Person VLM Training Pipeline")
    print("=" * 50)
    print("\nExample usage:")
    print("""
    from training import TrainingConfig, train_person_vlm
    
    config = TrainingConfig(
        train_file="data/train.json",
        val_file="data/val.json",
        epochs=20,
        batch_size=32,
        learning_rate=1e-4,
        output_dir="./checkpoints",
    )
    
    train_person_vlm(config)
    """)

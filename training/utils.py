"""
Training Utilities
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    OneCycleLR,
)
from typing import List, Optional, Dict, Any


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # MPS doesn't need special seeding - torch.manual_seed covers it


def get_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
) -> torch.optim.Optimizer:
    """
    Create optimizer with proper weight decay handling.
    
    Args:
        model: Model to optimize
        optimizer_type: "adamw", "adam", or "sgd"
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        betas: Adam betas
        
    Returns:
        Configured optimizer
    """
    # Separate parameters that should/shouldn't have weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # No weight decay for biases and layer norm weights
        if "bias" in name or "norm" in name or "embedding" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    if optimizer_type.lower() == "adamw":
        return AdamW(param_groups, lr=learning_rate, betas=betas)
    elif optimizer_type.lower() == "adam":
        return Adam(param_groups, lr=learning_rate, betas=betas)
    elif optimizer_type.lower() == "sgd":
        return SGD(param_groups, lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    num_training_steps: int = 10000,
    warmup_steps: int = 500,
    warmup_ratio: float = 0.1,
    min_lr_ratio: float = 0.1,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: "cosine", "linear", "one_cycle", or "none"
        num_training_steps: Total training steps
        warmup_steps: Number of warmup steps (overrides warmup_ratio)
        warmup_ratio: Warmup as fraction of total steps
        min_lr_ratio: Minimum LR as fraction of initial LR
        
    Returns:
        Configured scheduler or None
    """
    if scheduler_type.lower() == "none":
        return None
    
    # Calculate warmup steps
    if warmup_steps == 0:
        warmup_steps = int(num_training_steps * warmup_ratio)
    
    if scheduler_type.lower() == "cosine":
        # Warmup + Cosine decay
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=optimizer.defaults["lr"] * min_lr_ratio,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
    
    elif scheduler_type.lower() == "linear":
        # Warmup + Linear decay
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        linear = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr_ratio,
            total_iters=num_training_steps - warmup_steps,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, linear],
            milestones=[warmup_steps],
        )
    
    elif scheduler_type.lower() == "one_cycle":
        return OneCycleLR(
            optimizer,
            max_lr=optimizer.defaults["lr"],
            total_steps=num_training_steps,
            pct_start=warmup_ratio,
            anneal_strategy="cos",
        )
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: "min" or "max"
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            
        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    step: int,
    loss: float,
    path: str,
):
    """Save training checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "step": step,
        "loss": loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    path: str,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    print(f"Checkpoint loaded: {path}")
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss", float("inf")),
    }

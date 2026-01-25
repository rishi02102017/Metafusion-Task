"""
Dataset Module for Person VLM Training
Loads person blobs with their captions from JSONL format (MSP60k)
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

try:
    from torchvision import transforms
except ImportError:
    transforms = None

from .vocabulary import PersonVocabulary


class PersonBlobDataset(Dataset):
    """
    Dataset for person blob images with structured captions.
    
    Supports JSONL format (MSP60k style):
    {"image": "path/to/image.jpg", "answer": "description...", "one_hot": [...]}
    
    Also supports JSON array format:
    [{"image_path": "path/to/image.jpg", "caption": "description..."}]
    """
    
    def __init__(
        self,
        data_file: str,
        vocab: PersonVocabulary,
        image_dir: Optional[str] = None,
        image_size: int = 224,
        max_seq_length: int = 128,
        transform: Optional[Callable] = None,
        augment: bool = False,
    ):
        """
        Args:
            data_file: Path to JSONL or JSON file with image paths and captions
            vocab: PersonVocabulary instance
            image_dir: Directory containing images (for resolving relative paths)
            image_size: Size to resize images to
            max_seq_length: Maximum caption length
            transform: Optional custom transform
            augment: Whether to apply data augmentation
        """
        self.vocab = vocab
        self.image_size = image_size
        self.max_seq_length = max_seq_length
        self.image_dir = image_dir
        
        # Determine file format and load data
        self.data = self._load_data(data_file)
        
        print(f"Loaded {len(self.data)} samples from {data_file}")
        
        # Setup transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._default_transform(augment)
    
    def _load_data(self, data_file: str) -> List[Dict]:
        """Load data from JSONL or JSON file."""
        data = []
        
        if data_file.endswith('.jsonl'):
            # JSONL format (MSP60k style)
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            # Normalize to common format
                            normalized = self._normalize_item(item)
                            if normalized:
                                data.append(normalized)
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line: {e}")
        else:
            # JSON array format
            with open(data_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            for item in raw_data:
                normalized = self._normalize_item(item)
                if normalized:
                    data.append(normalized)
        
        return data
    
    def _normalize_item(self, item: Dict) -> Optional[Dict]:
        """Normalize different data formats to common structure."""
        # MSP60k format: {"image": "...", "answer": "...", "one_hot": [...]}
        if "answer" in item:
            image_path = item.get("image", "")
            caption = item.get("answer", "")
            
            # Extract filename from full path
            if "/" in image_path:
                filename = os.path.basename(image_path)
            else:
                filename = image_path
            
            # Clean caption (remove extra whitespace, newlines)
            caption = self._clean_caption(caption)
            
            if caption:
                return {
                    "image_path": filename,
                    "caption": caption,
                    "attributes": item.get("one_hot", []),
                }
        
        # Standard format: {"image_path": "...", "caption": "..."}
        elif "caption" in item:
            if item.get("caption") and item.get("success", True):
                return {
                    "image_path": item.get("image_path", ""),
                    "caption": item["caption"],
                    "attributes": item.get("attributes", []),
                }
        
        return None
    
    def _clean_caption(self, caption: str) -> str:
        """Clean and truncate caption."""
        if not caption:
            return ""
        
        # Remove extra whitespace and newlines
        caption = " ".join(caption.split())
        
        # Truncate very long captions (keep first few sentences)
        if len(caption) > 500:
            # Find a good breaking point
            sentences = caption.split('. ')
            truncated = []
            length = 0
            for sent in sentences:
                if length + len(sent) > 400:
                    break
                truncated.append(sent)
                length += len(sent)
            caption = '. '.join(truncated)
            if not caption.endswith('.'):
                caption += '.'
        
        return caption
    
    def _resolve_image_path(self, filename: str) -> str:
        """Resolve image path from filename."""
        if self.image_dir:
            return os.path.join(self.image_dir, filename)
        return filename
    
    def _default_transform(self, augment: bool) -> Callable:
        """Create default image transform."""
        if transforms is None:
            raise ImportError("torchvision required for default transforms")
        
        if augment:
            return transforms.Compose([
                transforms.Resize((self.image_size + 32, self.image_size + 32)),
                transforms.RandomCrop(self.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            dict with keys:
                - image: (3, H, W) tensor
                - input_ids: (seq_len,) tensor
                - labels: (seq_len,) tensor
                - attention_mask: (seq_len,) tensor
        """
        item = self.data[idx]
        
        # Load and transform image
        image_path = self._resolve_image_path(item["image_path"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image
            image = Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))
        
        image = self.transform(image)
        
        # Encode caption
        caption = item["caption"]
        token_ids = self.vocab.encode(
            caption,
            add_bos=True,
            add_eos=True,
            max_length=self.max_seq_length,
        )
        
        # Pad to max length
        attention_mask = [1] * len(token_ids)
        padding_length = self.max_seq_length - len(token_ids)
        
        if padding_length > 0:
            token_ids = token_ids + [self.vocab.pad_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        # Convert to tensors
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = input_ids.clone()
        
        # Mask padding in labels (set to -100 for ignore)
        labels[attention_mask == 0] = -100
        
        return {
            "image": image,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class PersonBlobDatasetInMemory(Dataset):
    """
    In-memory dataset for faster training on smaller datasets.
    Loads all images into RAM.
    """
    
    def __init__(
        self,
        data_file: str,
        vocab: PersonVocabulary,
        image_dir: Optional[str] = None,
        image_size: int = 224,
        max_seq_length: int = 128,
        augment: bool = False,
    ):
        self.vocab = vocab
        self.image_size = image_size
        self.max_seq_length = max_seq_length
        self.image_dir = image_dir
        
        # Load and normalize data using the same logic as PersonBlobDataset
        temp_dataset = PersonBlobDataset.__new__(PersonBlobDataset)
        temp_dataset.image_dir = image_dir
        data = temp_dataset._load_data(data_file)
        
        print(f"Loading {len(data)} images into memory...")
        
        # Base transform (no augmentation - apply augment at runtime)
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        
        # Augmentation transforms
        self.augment = augment
        if augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])
        
        # Load all data
        self.images = []
        self.captions = []
        
        for item in data:
            try:
                image_path = item["image_path"]
                if image_dir:
                    image_path = os.path.join(image_dir, image_path)
                img = Image.open(image_path).convert("RGB")
                self.images.append(img)
                self.captions.append(item["caption"])
            except Exception as e:
                print(f"Error loading {item['image_path']}: {e}")
        
        print(f"Loaded {len(self.images)} images into memory")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = self.images[idx].copy()  # Copy to avoid mutation
        caption = self.captions[idx]
        
        # Apply augmentation if enabled
        if self.augment:
            image = self.aug_transform(image)
        
        # Apply base transform
        image = self.base_transform(image)
        
        # Encode caption
        token_ids = self.vocab.encode(
            caption,
            add_bos=True,
            add_eos=True,
            max_length=self.max_seq_length,
        )
        
        # Pad
        attention_mask = [1] * len(token_ids)
        padding_length = self.max_seq_length - len(token_ids)
        
        if padding_length > 0:
            token_ids = token_ids + [self.vocab.pad_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            "image": image,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def create_dataloaders(
    train_file: str,
    val_file: Optional[str],
    vocab: PersonVocabulary,
    image_dir: Optional[str] = None,
    batch_size: int = 32,
    image_size: int = 224,
    max_seq_length: int = 128,
    num_workers: int = 4,
    augment_train: bool = True,
    in_memory: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_file: Path to training JSONL/JSON file
        val_file: Optional path to validation JSONL/JSON file
        vocab: PersonVocabulary instance
        image_dir: Directory containing images
        batch_size: Batch size
        image_size: Image size
        max_seq_length: Maximum sequence length
        num_workers: DataLoader workers
        augment_train: Whether to augment training data
        in_memory: Whether to load all data into memory
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    DatasetClass = PersonBlobDatasetInMemory if in_memory else PersonBlobDataset
    
    # Training dataset
    train_dataset = DatasetClass(
        data_file=train_file,
        vocab=vocab,
        image_dir=image_dir,
        image_size=image_size,
        max_seq_length=max_seq_length,
        augment=augment_train,
    )
    
    # pin_memory only helps with CUDA
    use_pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,
    )
    
    # Validation dataset
    val_loader = None
    if val_file:
        val_dataset = DatasetClass(
            data_file=val_file,
            vocab=vocab,
            image_dir=image_dir,
            image_size=image_size,
            max_seq_length=max_seq_length,
            augment=False,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        )
    
    return train_loader, val_loader


def split_jsonl(
    input_file: str,
    train_file: str,
    val_file: str,
    test_file: Optional[str] = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.0,
    seed: int = 42,
):
    """
    Split a JSONL dataset into train/val/test sets.
    
    Args:
        input_file: Input JSONL file
        train_file: Output training JSONL file
        val_file: Output validation JSONL file
        test_file: Optional output test JSONL file
        val_ratio: Fraction for validation
        test_ratio: Fraction for test
        seed: Random seed
    """
    random.seed(seed)
    
    # Load all lines
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Shuffle
    random.shuffle(lines)
    
    # Calculate split sizes
    total = len(lines)
    test_size = int(total * test_ratio)
    val_size = int(total * val_ratio)
    train_size = total - val_size - test_size
    
    # Split
    train_lines = lines[:train_size]
    val_lines = lines[train_size:train_size + val_size]
    test_lines = lines[train_size + val_size:] if test_ratio > 0 else []
    
    # Save
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    
    with open(val_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines))
    
    if test_file and test_lines:
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_lines))
    
    print(f"Split {total} samples into:")
    print(f"  Train: {len(train_lines)} ({train_file})")
    print(f"  Val: {len(val_lines)} ({val_file})")
    if test_lines:
        print(f"  Test: {len(test_lines)} ({test_file})")


def create_split(
    data_file: str,
    train_file: str,
    val_file: str,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Split a dataset into train and validation sets.
    Supports both JSON and JSONL formats.
    
    Args:
        data_file: Input JSON/JSONL file
        train_file: Output training file
        val_file: Output validation file
        val_ratio: Fraction for validation
        seed: Random seed
    """
    if data_file.endswith('.jsonl'):
        split_jsonl(data_file, train_file, val_file, val_ratio=val_ratio, seed=seed)
    else:
        # JSON format
        random.seed(seed)
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Shuffle
        random.shuffle(data)
        
        # Split
        val_size = int(len(data) * val_ratio)
        val_data = data[:val_size]
        train_data = data[val_size:]
        
        # Save
        with open(train_file, 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(val_file, 'w') as f:
            json.dump(val_data, f, indent=2)
        
        print(f"Split {len(data)} samples into:")
        print(f"  Train: {len(train_data)} ({train_file})")
        print(f"  Val: {len(val_data)} ({val_file})")


if __name__ == "__main__":
    # Test dataset loading
    import argparse
    
    parser = argparse.ArgumentParser(description="Test dataset loading")
    parser.add_argument("--data", type=str, help="Path to JSONL/JSON file")
    parser.add_argument("--image-dir", type=str, help="Image directory")
    parser.add_argument("--split", action="store_true", help="Create train/val split")
    args = parser.parse_args()
    
    if args.split and args.data:
        # Create split
        base_dir = os.path.dirname(args.data)
        train_file = os.path.join(base_dir, "train.jsonl")
        val_file = os.path.join(base_dir, "val.jsonl")
        create_split(args.data, train_file, val_file)
    
    elif args.data:
        # Test loading
        from vocabulary import create_default_vocabulary
        
        vocab = create_default_vocabulary()
        
        dataset = PersonBlobDataset(
            data_file=args.data,
            vocab=vocab,
            image_dir=args.image_dir,
        )
        
        print(f"\nDataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample keys: {sample.keys()}")
            print(f"Image shape: {sample['image'].shape}")
            print(f"Input IDs shape: {sample['input_ids'].shape}")

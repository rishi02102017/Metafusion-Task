"""
Controlled Vocabulary for Person Descriptions
Maps structured descriptions to token IDs
"""

import json
import re
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from collections import Counter


# Predefined vocabulary categories for person description
VOCABULARY_CATEGORIES = {
    # Structure words
    "structure": [
        "person", "wearing", "and", "with", "holding", "carrying",
        ",", "unknown", "a", "the", "in", "on",
    ],
    
    # Clothing types
    "clothing_upper": [
        "shirt", "t-shirt", "jacket", "coat", "sweater", "hoodie",
        "blouse", "top", "tank-top", "polo", "vest", "cardigan",
    ],
    
    "clothing_lower": [
        "pants", "jeans", "shorts", "skirt", "trousers", "leggings",
        "dress", "sweatpants", "joggers",
    ],
    
    # Colors
    "colors": [
        "black", "white", "red", "blue", "green", "yellow", "orange",
        "purple", "pink", "brown", "gray", "grey", "beige", "navy",
        "dark", "light", "bright", "striped", "patterned",
    ],
    
    # Objects in hand
    "objects": [
        "phone", "bag", "backpack", "handbag", "purse", "bottle",
        "umbrella", "briefcase", "laptop", "coffee", "cup", "food",
        "book", "newspaper", "keys", "wallet", "cigarette", "nothing",
        "shopping-bag", "suitcase", "luggage",
    ],
    
    # Actions/Postures
    "actions": [
        "standing", "walking", "running", "sitting", "cycling",
        "waiting", "talking", "looking", "using", "crossing",
    ],
    
    # Gender (when visible)
    "gender": [
        "male", "female",
    ],
    
    # Accessories
    "accessories": [
        "glasses", "sunglasses", "hat", "cap", "scarf", "mask",
        "headphones", "watch", "gloves", "helmet",
    ],
    
    # Hair (optional)
    "hair": [
        "hair", "long", "short", "bald",
    ],
    
    # Age approximation
    "age": [
        "young", "middle-aged", "elderly", "adult", "child",
    ],
}


class PersonVocabulary:
    """
    Controlled vocabulary for person descriptions.
    
    Maps tokens to IDs and vice versa.
    Includes special tokens for padding, BOS, EOS, and unknown.
    """
    
    # Special tokens
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"
    
    def __init__(
        self,
        vocab_dict: Optional[Dict[str, int]] = None,
        min_freq: int = 1,
    ):
        """
        Args:
            vocab_dict: Optional pre-built vocabulary
            min_freq: Minimum frequency for token inclusion (when building)
        """
        self.min_freq = min_freq
        
        if vocab_dict is not None:
            self.token2id = vocab_dict
        else:
            self.token2id = self._build_default_vocab()
        
        self.id2token = {v: k for k, v in self.token2id.items()}
        
        # Store special token IDs
        self.pad_id = self.token2id[self.PAD_TOKEN]
        self.bos_id = self.token2id[self.BOS_TOKEN]
        self.eos_id = self.token2id[self.EOS_TOKEN]
        self.unk_id = self.token2id[self.UNK_TOKEN]
    
    def _build_default_vocab(self) -> Dict[str, int]:
        """Build default vocabulary from predefined categories."""
        vocab = {}
        idx = 0
        
        # Add special tokens first
        for token in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]:
            vocab[token] = idx
            idx += 1
        
        # Add all category tokens
        for category, tokens in VOCABULARY_CATEGORIES.items():
            for token in tokens:
                if token not in vocab:
                    vocab[token] = idx
                    idx += 1
        
        return vocab
    
    @classmethod
    def from_corpus(
        cls,
        captions: List[str],
        min_freq: int = 2,
        max_vocab_size: int = 1000,
    ) -> "PersonVocabulary":
        """
        Build vocabulary from a corpus of captions.
        
        Args:
            captions: List of caption strings
            min_freq: Minimum frequency for inclusion
            max_vocab_size: Maximum vocabulary size
        """
        # Tokenize and count
        counter = Counter()
        for caption in captions:
            tokens = cls.tokenize_caption(caption)
            counter.update(tokens)
        
        # Build vocab starting with special tokens
        vocab = {
            cls.PAD_TOKEN: 0,
            cls.BOS_TOKEN: 1,
            cls.EOS_TOKEN: 2,
            cls.UNK_TOKEN: 3,
        }
        idx = 4
        
        # Add predefined vocabulary first (guaranteed inclusion)
        for category, tokens in VOCABULARY_CATEGORIES.items():
            for token in tokens:
                if token not in vocab:
                    vocab[token] = idx
                    idx += 1
        
        # Add frequent tokens from corpus
        for token, count in counter.most_common():
            if len(vocab) >= max_vocab_size:
                break
            if count >= min_freq and token not in vocab:
                vocab[token] = idx
                idx += 1
        
        return cls(vocab_dict=vocab, min_freq=min_freq)
    
    @staticmethod
    def tokenize_caption(caption: str) -> List[str]:
        """
        Tokenize a caption into words.
        
        Handles:
        - Lowercase conversion
        - Punctuation separation
        - Hyphenated words
        """
        caption = caption.lower().strip()
        
        # Keep hyphens within words, separate other punctuation
        caption = re.sub(r'([,.])', r' \1 ', caption)
        
        # Split and filter empty strings
        tokens = [t.strip() for t in caption.split() if t.strip()]
        
        return tokens
    
    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input caption string
            add_bos: Whether to prepend BOS token
            add_eos: Whether to append EOS token
            max_length: Maximum sequence length (including special tokens)
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenize_caption(text)
        
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        
        for token in tokens:
            ids.append(self.token2id.get(token, self.unk_id))
        
        if add_eos:
            ids.append(self.eos_id)
        
        # Truncate if needed
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length - 1] + [self.eos_id]
        
        return ids
    
    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip PAD, BOS, EOS tokens
            clean_up_tokenization_spaces: Whether to clean up spacing
            
        Returns:
            Decoded text string
        """
        special_ids = {self.pad_id, self.bos_id, self.eos_id}
        
        tokens = []
        for id in ids:
            if skip_special_tokens and id in special_ids:
                continue
            token = self.id2token.get(id, self.UNK_TOKEN)
            tokens.append(token)
        
        text = " ".join(tokens)
        
        if clean_up_tokenization_spaces:
            # Fix spacing around punctuation
            text = re.sub(r'\s+([,.])', r'\1', text)
            text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 64,
        padding: bool = True,
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Encode a batch of texts.
        
        Returns:
            Tuple of (padded_ids, lengths)
        """
        encoded = [self.encode(t, max_length=max_length) for t in texts]
        lengths = [len(e) for e in encoded]
        
        if padding:
            max_len = max(lengths)
            encoded = [
                e + [self.pad_id] * (max_len - len(e))
                for e in encoded
            ]
        
        return encoded, lengths
    
    def __len__(self) -> int:
        return len(self.token2id)
    
    def __contains__(self, token: str) -> bool:
        return token.lower() in self.token2id
    
    def save(self, path: str):
        """Save vocabulary to JSON file."""
        with open(path, 'w') as f:
            json.dump({
                "token2id": self.token2id,
                "min_freq": self.min_freq,
            }, f, indent=2)
        print(f"Vocabulary saved to {path} ({len(self)} tokens)")
    
    @classmethod
    def load(cls, path: str) -> "PersonVocabulary":
        """Load vocabulary from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        vocab = cls(
            vocab_dict=data["token2id"],
            min_freq=data.get("min_freq", 1),
        )
        print(f"Vocabulary loaded from {path} ({len(vocab)} tokens)")
        return vocab
    
    def get_vocab_stats(self) -> Dict:
        """Get vocabulary statistics."""
        stats = {
            "total_tokens": len(self),
            "special_tokens": 4,
            "categories": {},
        }
        
        for category, tokens in VOCABULARY_CATEGORIES.items():
            in_vocab = sum(1 for t in tokens if t in self.token2id)
            stats["categories"][category] = {
                "total": len(tokens),
                "in_vocab": in_vocab,
            }
        
        return stats


def create_default_vocabulary() -> PersonVocabulary:
    """Create and return the default vocabulary."""
    vocab = PersonVocabulary()
    print(f"Created default vocabulary with {len(vocab)} tokens")
    return vocab


if __name__ == "__main__":
    # Test vocabulary
    vocab = create_default_vocabulary()
    
    # Test encoding/decoding
    test_captions = [
        "person wearing blue shirt and black pants, holding a phone, walking",
        "male wearing red jacket and jeans, carrying backpack, standing",
        "female wearing white dress, holding umbrella, waiting",
        "person wearing unknown top, unknown gender, running",
    ]
    
    print("\nEncoding/Decoding test:")
    for caption in test_captions:
        encoded = vocab.encode(caption)
        decoded = vocab.decode(encoded)
        print(f"  Original: {caption}")
        print(f"  Encoded:  {encoded}")
        print(f"  Decoded:  {decoded}")
        print()
    
    # Print stats
    print("\nVocabulary stats:")
    stats = vocab.get_vocab_stats()
    print(f"  Total tokens: {stats['total_tokens']}")
    for cat, info in stats["categories"].items():
        print(f"  {cat}: {info['in_vocab']}/{info['total']}")

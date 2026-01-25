"""
Inference Module for Person VLM
"""

from .predict import PersonDescriber, load_model

__all__ = [
    "PersonDescriber",
    "load_model",
]

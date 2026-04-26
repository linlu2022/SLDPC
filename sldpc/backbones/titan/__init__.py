"""TITAN backbone adapter (CLIP-style text encoder) for SLDPC."""

from .model_wrapper import PromptedTitan
from .prompt_learner import TitanPromptLearner
from .text_encoding import encode_text

__all__ = ["PromptedTitan", "TitanPromptLearner", "encode_text"]

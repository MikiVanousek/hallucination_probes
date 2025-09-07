"""
Annotation Pipeline for hallucination detection.

This module provides tools to annotate text completions from language models,
identifying which spans are supported, not supported, or have insufficient information.
"""

from .annotate import annotate_completion
from .run import main, PipelineConfig

__all__ = [
    "annotate_completion",
    "main",
    "PipelineConfig",
]
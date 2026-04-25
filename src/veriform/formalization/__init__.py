"""Autoformalization backends (raw CoT statement -> Lean theorem statement)."""

from .base import BaseFormalizer
from .fine_tuned import (
    StepfunFormalizer,
    KiminaFormalizer,
    GoedelFormalizer,
    HeraldFormalizer,
)

FORMALIZER_REGISTRY = {
    "stepfun": StepfunFormalizer,
    "kimina": KiminaFormalizer,
    "goedel": GoedelFormalizer,
    "herald": HeraldFormalizer,
}

__all__ = [
    "BaseFormalizer",
    "StepfunFormalizer",
    "KiminaFormalizer",
    "GoedelFormalizer",
    "HeraldFormalizer",
    "FORMALIZER_REGISTRY",
]

"""Lean proving backends + theorem extraction utilities."""

from .deepseek_prover import DeepSeekProver
from .theorem_extractor import TheoremExtractor

PROVER_REGISTRY = {
    "deepseek": DeepSeekProver,
}

__all__ = ["DeepSeekProver", "TheoremExtractor", "PROVER_REGISTRY"]

"""Perturbation strategies that modify CoT statements.

Active perturbers:
  - StandardPerturber    : regex-based (default for paper experiments)
  - BrokenMathPerturber  : lookup-based, joins a pre-computed GPT-5.2 batch
                           output (BrokenMath-style prompt) onto DAG nodes
  - DeepSeekPrePerturber : loads pre-generated LLM perturbations from JSON
  - GeminiPerturber      : Google Gemini LLM-based
  - OpenAIPerturber      : OpenAI LLM-based

The Lean-side theorem negation lives in src/veriform/proving/negation.py
(CLAUDE.md improvement #2; backed by Negate.lean metaprogramming).
"""

from .perturbers import (
    StandardPerturber,
    DeepSeekPrePerturber,
    BaseLLMPerturber,
    GeminiPerturber,
    OpenAIPerturber,
)
from .brokenmath_perturber import BrokenMathPerturber

PERTURBER_REGISTRY = {
    "regex": StandardPerturber,
    "brokenmath": BrokenMathPerturber,
    "deepseek_pre": DeepSeekPrePerturber,
    "gemini": GeminiPerturber,
    "openai": OpenAIPerturber,
}

__all__ = [
    "StandardPerturber",
    "BrokenMathPerturber",
    "DeepSeekPrePerturber",
    "BaseLLMPerturber",
    "GeminiPerturber",
    "OpenAIPerturber",
    "PERTURBER_REGISTRY",
]

"""Perturbation strategies that modify CoT statements.

Active perturbers:
  - StandardPerturber  : regex-based (default for paper experiments)
  - DeepSeekPrePerturber : loads pre-generated LLM perturbations from JSON
  - GeminiPerturber    : Google Gemini LLM-based
  - OpenAIPerturber    : OpenAI LLM-based

Future homes (see plan/scaffolds in this package):
  - brokenmath_perturber.py  (paper TODO)
  - lean_negation/           (CLAUDE.md improvement #2)
"""

from .perturbers import (
    StandardPerturber,
    DeepSeekPrePerturber,
    BaseLLMPerturber,
    GeminiPerturber,
    OpenAIPerturber,
)

PERTURBER_REGISTRY = {
    "regex": StandardPerturber,
    "deepseek_pre": DeepSeekPrePerturber,
    "gemini": GeminiPerturber,
    "openai": OpenAIPerturber,
}

__all__ = [
    "StandardPerturber",
    "DeepSeekPrePerturber",
    "BaseLLMPerturber",
    "GeminiPerturber",
    "OpenAIPerturber",
    "PERTURBER_REGISTRY",
]

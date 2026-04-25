"""Negation of natural-language CoT statements (perturbation-time).

Used by StandardPerturber.logical_negation_str to flip the meaning of an
English step before passing it to the formaliser. Today the only backend
is regex-based; the Protocol exists so a future NLP-aware backend can
slot in without touching StandardPerturber.

NOT to be confused with Lean-side theorem negation (which lives in
src/veriform/proving/negation.py and operates on formalised Lean code).
"""

from typing import Protocol, Tuple, runtime_checkable

from .regex_backend import RegexNegationBackend


@runtime_checkable
class NegationBackend(Protocol):
    """Returns ``(possibly_negated_content, did_change)``."""

    def negate(self, content: str) -> Tuple[str, bool]:  # pragma: no cover - protocol
        ...


_REGISTRY = {
    "regex": RegexNegationBackend,
}


def get_backend(name: str = "regex") -> NegationBackend:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown negation backend: {name}. Known: {list(_REGISTRY)}")
    return _REGISTRY[name]()


__all__ = ["NegationBackend", "RegexNegationBackend", "get_backend"]

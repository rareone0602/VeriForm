"""Pluggable backends for negating Lean theorem statements.

Today there is one backend (regex) extracted from StandardPerturber.
A Lean-parser-based backend is planned (see CLAUDE.md improvement #2),
which is why this lives behind a Protocol — drop a new file in here that
implements ``NegationBackend.negate`` and switch via ``get_backend(name)``.
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
    # "lean_parser": LeanParserNegationBackend,  # FUTURE — see CLAUDE.md improvement #2
}


def get_backend(name: str = "regex") -> NegationBackend:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown negation backend: {name}. Known: {list(_REGISTRY)}")
    return _REGISTRY[name]()


__all__ = ["NegationBackend", "RegexNegationBackend", "get_backend"]

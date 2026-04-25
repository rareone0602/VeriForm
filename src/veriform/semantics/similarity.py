"""Cosine similarity between NL and Lean embeddings produced by ProofBridge."""

from typing import Iterable

from .encoder_proofbridge import ProofBridgeEncoder


def nl_lean_similarity(
    nl_statements: Iterable[str],
    lean_snippets: Iterable[str],
    encoder: ProofBridgeEncoder | None = None,
) -> list[float]:
    """Return per-pair cosine similarity in [-1, 1]. Pending encoder hookup."""
    raise NotImplementedError("Awaits ProofBridgeEncoder — see semantics/__init__.py")

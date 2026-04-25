"""Semantic Drift Filter: drop NL/Lean pairs whose embedding similarity falls
below a threshold before computing TPR/FPR.

Implements the ``Semantic Drift Filter`` task in paper/sections/problem.tex
(see Figure proofbridge). Once the ProofBridge wrapper lands, plug this
into evaluation/metrics.py via a pre-aggregation hook.
"""

from typing import List

from .encoder_proofbridge import ProofBridgeEncoder


class SemanticDriftFilter:
    """Mark pairs as semantically drifted (similarity < ``threshold``)."""

    def __init__(self, encoder: ProofBridgeEncoder, threshold: float = 0.5):
        self.encoder = encoder
        self.threshold = threshold

    def is_drifted(self, nl_statements: List[str], lean_snippets: List[str]) -> List[bool]:
        raise NotImplementedError("Awaits ProofBridgeEncoder — see semantics/__init__.py")

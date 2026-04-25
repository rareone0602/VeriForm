"""Semantic-faithfulness module.

Wraps an external NL <-> Lean joint encoder (initially ProofBridge, used
out-of-the-box per the team Discord on 2026-04-24) to score how well the
formaliser preserves the meaning of the input statement, beyond outcome
equivalence.

The motivating critique: TPR/FPR (currently in src/veriform/evaluation/)
treat ``2+2=4 -> 3+3=6`` as equivalent because both prove. ``semantics``
provides the embedding-similarity signal needed for the planned
``Semantic Drift Filter`` and weighted Faithfulness Index (paper
TODO in paper/sections/problem.tex).

Status: scaffold only. None of the three classes are implemented yet.
"""

from .encoder_proofbridge import ProofBridgeEncoder
from .similarity import nl_lean_similarity
from .drift_filter import SemanticDriftFilter

__all__ = ["ProofBridgeEncoder", "nl_lean_similarity", "SemanticDriftFilter"]

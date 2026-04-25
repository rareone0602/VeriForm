"""Wrapper around the ProofBridge joint NL <-> Lean encoder.

We are using ProofBridge out-of-the-box (per team Discord 2026-04-24).
Implementation lands once the ProofBridge group hands over their
artefact / API.
"""

from typing import List


class ProofBridgeEncoder:
    """Encode natural-language strings and Lean code into a shared embedding space."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        # TODO: load ProofBridge model weights once they are released.

    def encode_nl(self, statements: List[str]):
        raise NotImplementedError("ProofBridge encoder integration pending — see semantics/__init__.py")

    def encode_lean(self, snippets: List[str]):
        raise NotImplementedError("ProofBridge encoder integration pending — see semantics/__init__.py")

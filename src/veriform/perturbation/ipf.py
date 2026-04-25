"""Ineffective Perturbation Finder (IPF): decide whether a perturbed CoT
is still semantically equivalent to the original (i.e. perturbation
failed to inject an error).

Paper context: appendix experimental_details.tex defines ``IPF`` as a
boolean filter excluding ``S^perturbed_i`` for which ``IPF == True`` from
the FPR calculation. Discord 2026-04-13: GPT 5.2 currently judges 3.26%
of regex perturbations and 10.45% of LLM perturbations as ineffective.

Currently this module only declares the interface. Wire one of the
implementations into run_benchmark.py once the LLM-judge variant is
trained / prompt-tuned.
"""

from abc import ABC, abstractmethod
from typing import Optional


class IneffectivePerturbationFinder(ABC):
    """``__call__(original, perturbed) -> True`` means the perturbation was ineffective."""

    @abstractmethod
    def __call__(self, original: str, perturbed: str) -> bool: ...


class LLMJudgeIPF(IneffectivePerturbationFinder):
    """LLM-as-judge IPF (paper TODO + Discord 2026-04-13 follow-up)."""

    def __init__(self, model_name: str = "gpt-5.2", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key

    def __call__(self, original: str, perturbed: str) -> bool:
        raise NotImplementedError("LLM-judge IPF prompt + client wiring pending — see ipf.py")

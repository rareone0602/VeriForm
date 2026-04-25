"""LLM-based perturbation following the BrokenMath prompt format.

Paper TODO (paper/sections/experiments.tex): replace the current LLM
perturber with one that uses the exact BrokenMath prompt, then verify
effectiveness with both an LLM judge (see ``ipf.py``) and a human
IMO-medallist sub-check.

Status: scaffold. Subclass BaseLLMPerturber from .perturbers and override
``_call_llm`` once the prompt is finalised.
"""

from typing import Optional

from .perturbers import BaseLLMPerturber


class BrokenMathPerturber(BaseLLMPerturber):
    """LLM perturber using BrokenMath's adversarial prompt template."""

    SYSTEM_PROMPT_TEMPLATE = (
        # TODO: paste the exact BrokenMath prompt here once finalised.
        "<<BROKENMATH_PROMPT_TBD>>"
    )

    def __init__(self, p: float = 0.5, api_key: Optional[str] = None, model_name: str = "gpt-4o"):
        super().__init__(p=p, api_key=api_key, model_name=model_name)
        self.system_prompt = self.SYSTEM_PROMPT_TEMPLATE

    def _call_llm(self, content: str) -> Optional[str]:
        raise NotImplementedError("BrokenMath prompt + client wiring pending — see brokenmath_perturber.py")

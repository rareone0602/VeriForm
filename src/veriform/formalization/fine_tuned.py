
from transformers import AutoTokenizer
from openai import OpenAI
from typing import Tuple
from vllm import LLM, SamplingParams


# Assuming BaseFormalizer is defined in .base
from .base import BaseFormalizer

class HeraldFormalizer(BaseFormalizer):
    """FrenzyMath/Herald_translator. System + user prompt verbatim from
    `worker/translator.py` in frenzymath/herald_translator. Chat formatting
    delegates to the model's tokenizer chat_template (Llama-style: BOS,
    `User: ... Assistant:`, `<|im_end|>` after assistant turn) — the previous
    custom format_dialog had several deviations from that template."""

    MODEL_DIR = "FrenzyMath/Herald_translator"

    def __init__(self, sampling="deterministic"):
        self.system_prompt = "You are an expert at Lean 4 and Mathematics."
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_DIR,
            trust_remote_code=True,
        )
        # On H200 (141GB), 0.2 = ~28GB, which is safe for the 7B model.
        self.model = LLM(
            self.MODEL_DIR,
            dtype="bfloat16",
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.2,
        )

        # Herald's repo doesn't pin sampling; defaults are caller-supplied.
        # We use temp=0 for deterministic (paper) and a mild temperature for
        # the "recommended" mode.
        self.recommended_sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1024,
            repetition_penalty=1.0,
        )
        self.deterministic_sampling_params = SamplingParams(
            temperature=0,
            max_tokens=16384,
        )
        self.sampling_params = (
            self.recommended_sampling_params if sampling == "recommended"
            else self.deterministic_sampling_params
        )

    def _formalize_prompt(self, informal_problems: list[str]) -> Tuple[list[str], list[str]]:
        response = self.model.generate(informal_problems, sampling_params=self.sampling_params)
        lean_codes = []
        raw_outputs = []
        for i in range(len(informal_problems)):
            try:
                generated_text = response[i].outputs[0].text
                raw_outputs.append(generated_text)
                lean_code = self.parse_lean_code(generated_text)
            except Exception:
                lean_code = None
            lean_codes.append(lean_code)
        return lean_codes, raw_outputs

    def initialize_dialog(self):
        self.dialog = [{"role": "system", "content": self.system_prompt}]

    def get_formal_statement_prompt(self, informal_problem: str) -> str:
        # Verbatim from frenzymath/herald_translator worker/translator.py:
        #   "Please translate the natural language statement to Lean4 code with the header\n"
        #   "**Name**\n{informal_name}\n"
        #   "**Informal statement**\n{informal_statement}\n"
        return (
            "Please translate the natural language statement to Lean4 code with the header\n"
            "**Name**\nmy_favorite_theorem\n"
            f"**Informal statement**\n{informal_problem}\n"
        )

    def format_dialog(self, dialog: list[dict]) -> str:
        # Use the tokenizer chat_template shipped in the model repo —
        # it is the canonical Llama-style template and matches the format
        # the model was trained on.
        return self.tokenizer.apply_chat_template(
            dialog, tokenize=False, add_generation_prompt=True
        )


class GoedelFormalizer(BaseFormalizer):
    """Goedel-LM/Goedel-Formalizer-V2-8B. Prompt + sampling verbatim from
    the official model card example. Note the user prompt deliberately has
    no separator between {informal_statement} and 'Think before...': that
    is the literal training-time format."""

    MODEL_DIR = "Goedel-LM/Goedel-Formalizer-V2-8B"

    def __init__(self, sampling="deterministic", base_url="http://localhost:8002/v1"):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_DIR)
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=base_url,
            timeout=3600,
        )
        # Recommended sampling from the official model card.
        self.recommended_kwargs = {
            "temperature": 0.9,
            "top_p": 0.95,
            "max_tokens": 16384,
            "extra_body": {"top_k": 20},
        }
        self.deterministic_kwargs = {
            "temperature": 0.0,
            "max_tokens": 16384,
        }
        self.gen_kwargs = (
            self.recommended_kwargs if sampling == "recommended"
            else self.deterministic_kwargs
        )

    def initialize_dialog(self):
        # Goedel's official example uses NO system role.
        self.dialog = []

    def get_formal_statement_prompt(self, informal_problem: str) -> str:
        # Verbatim from the Goedel-Formalizer-V2-8B model card example —
        # note that {informal_problem} is concatenated directly to
        # "Think before..." with no separator, matching training format.
        return (
            "Please autoformalize the following natural language problem statement in Lean 4. "
            "Use the following theorem name: my_favorite_theorem\n"
            "The natural language statement is: \n"
            f"{informal_problem}"
            "Think before you provide the lean statement."
        )

    def format_dialog(self, dialog: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            dialog, tokenize=False, add_generation_prompt=True
        )

    


class KiminaFormalizer(BaseFormalizer):
    """AI-MO/Kimina-Autoformalizer-7B. System + user prompt + sampling
    verbatim from the official model card example."""

    MODEL_DIR = "AI-MO/Kimina-Autoformalizer-7B"

    def __init__(self, sampling="deterministic", base_url="http://localhost:8002/v1"):
        self.system_prompt = "You are an expert in mathematics and Lean 4."
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_DIR)
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=base_url,
            timeout=3600,
        )
        # Recommended sampling from the official model card. Previously
        # this had drifted (temp 0.7, top_p 0.8, max 16384, top_k 20)
        # which is NOT what Kimina was trained on.
        self.recommended_kwargs = {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 2048,
        }
        self.deterministic_kwargs = {
            "temperature": 0.0,
            "max_tokens": 16384,
        }
        self.gen_kwargs = (
            self.recommended_kwargs if sampling == "recommended"
            else self.deterministic_kwargs
        )

    def initialize_dialog(self):
        self.dialog = [{"role": "system", "content": self.system_prompt}]

    def get_formal_statement_prompt(self, informal_problem: str) -> str:
        # Verbatim from the Kimina-Autoformalizer-7B model card example.
        return (
            "Please autoformalize the following problem in Lean 4 with a header. "
            "Use the following theorem names: my_favorite_theorem.\n\n"
            f"{informal_problem}"
        )

    def format_dialog(self, dialog: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            dialog, tokenize=False, add_generation_prompt=True
        )

    
    
class StepfunFormalizer(BaseFormalizer):
    """stepfun-ai/StepFun-Formalizer-7B. System + user prompt + sampling
    verbatim from the official model card example. CRITICAL: the official
    inference code appends the literal token `<think>` to the chat-template
    output to trigger the model's chain-of-thought; without it performance
    degrades significantly (the base is DeepSeek-R1-distilled)."""

    MODEL_DIR = "stepfun-ai/StepFun-Formalizer-7B"
    HEADER = "import Mathlib\n\nopen Real\n"

    def __init__(self, sampling="deterministic", base_url="http://localhost:8002/v1"):
        self.system_prompt = "You are an expert in mathematics and Lean 4."
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_DIR)
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=base_url,
            timeout=3600,
        )
        # Recommended sampling from the official model card.
        self.recommended_kwargs = {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 16384,
        }
        self.deterministic_kwargs = {
            "temperature": 0.0,
            "max_tokens": 16384,
        }
        self.gen_kwargs = (
            self.recommended_kwargs if sampling == "recommended"
            else self.deterministic_kwargs
        )

    def initialize_dialog(self):
        self.dialog = [{"role": "system", "content": self.system_prompt}]

    def get_formal_statement_prompt(self, informal_problem: str) -> str:
        # Verbatim from the StepFun-Formalizer-7B model card example.
        return (
            "Please autoformalize the following problem in Lean 4 with a header. "
            "Use the following theorem names: my_favorite_theorem.\n\n"
            f"{informal_problem}"
            f"\n\nYour code should start with:\n```Lean4\n{self.HEADER}\n```\n"
        )

    def format_dialog(self, dialog: list[dict]) -> str:
        # The official example appends "<think>" after the chat template
        # to switch the DeepSeek-R1-distilled base into reasoning mode.
        return self.tokenizer.apply_chat_template(
            dialog, tokenize=False, add_generation_prompt=True
        ) + "<think>"
    
    
"""
# Run this before using this script
stepfun: stepfun-ai/StepFun-Formalizer-7B
herald: FrenzyMath/Herald_translator
kimina: AI-MO/Kimina-Autoformalizer-7B
goedel: Goedel-LM/Goedel-Formalizer-V2-8B

CUDA_VISIBLE_DEVICES=2 vllm serve stepfun-ai/StepFun-Formalizer-7B \
    --port 8002 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --trust-remote-code \
    --enable-prefix-caching
"""
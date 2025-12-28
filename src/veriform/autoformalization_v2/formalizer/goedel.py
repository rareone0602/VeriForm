


from typing import Protocol, Optional


from ..dag import DAGModel, Flagging

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import backoff
import re
from .base import BaseFormalizer

class GoedelFormalizer(BaseFormalizer):
    MODEL_DIR = "Goedel-LM/Goedel-Formalizer-V2-8B"

    def __init__(self):
        self.system_prompt = "You are an expert in mathematics and Lean 4."
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_DIR)
        self.model = LLM(
            self.MODEL_DIR,
            dtype="bfloat16", 
            tensor_parallel_size=2, # 8 for 32B, 4 for 7B
            trust_remote_code=True,
            gpu_memory_utilization=0.25,
        )
        self.lean_pattern = re.compile(
            r"```lean4.*?(^theorem.*?)```", 
            re.DOTALL | re.IGNORECASE | re.MULTILINE
        )

        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=16384,
        )

    def initialize_dialog(self):
        self.dialog = []

    def get_formal_statement_prompt(self, informal_problem: str) -> str:
        return (
            f"Please autoformalize the following natural language problem statement in Lean 4. "
            f"Use the following theorem name: my_favorite_theorem\n"
            f"The natural language statement is: \n"
            f"{informal_problem}"
            f"Think before you provide the lean statement."
        )
    
    def parse_lean_code(self, response_text: str) -> str:
        """
        Get the LAST Lean 4 code block from the response text.
        """
        matches = self.lean_pattern.findall(response_text)
        if matches:
            return matches[-1].strip()
        else:
            raise ValueError("No Lean 4 code block found in the response.")
    
    def format_dialog(self, dialog: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=True)

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re
from .base import BaseFormalizer

class KiminaFormalizer(BaseFormalizer):
    MODEL_DIR = "AI-MO/Kimina-Autoformalizer-7B"

    def __init__(self):
        self.system_prompt = "You are an expert in mathematics and Lean 4."
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_DIR)
        self.model = LLM(
            self.MODEL_DIR,
            dtype="bfloat16", 
            tensor_parallel_size=2, # 8 for 32B, 4 for 7B
            gpu_memory_utilization=0.25,
        )
        self.lean_pattern = re.compile(
            r".*(^theorem.*?sorry)", 
            re.DOTALL | re.IGNORECASE | re.MULTILINE
        )
        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=16384,
        )

    def initialize_dialog(self):
        self.dialog = [
            {"role": "system", "content": self.system_prompt}
        ]

    def get_formal_statement_prompt(self, informal_problem: str) -> str:
        prompt = "Please autoformalize the following problem in Lean 4 with a header. Use the following theorem names: my_favorite_theorem.\n\n"
        prompt += informal_problem
        # prompt += f"\n\nYour code should start with:\n```Lean4\n{header}\n```\n"
        return prompt
    
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

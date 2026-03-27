
from transformers import AutoTokenizer
from openai import OpenAI
from typing import Tuple
from vllm import LLM, SamplingParams


# Assuming BaseFormalizer is defined in .base
from .base import BaseFormalizer

class HeraldFormalizer(BaseFormalizer):
    MODEL_DIR = "FrenzyMath/Herald_translator"

    def __init__(self, sampling="deterministic"):
        self.system_prompt = "You are an expert at Lean 4 and Mathematics."
        
        # Aligned with repo: explicit trust_remote_code
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_DIR, 
            trust_remote_code=True
        )
        
        # Aligned with repo: 0.2 utilization is standard for their multi-worker setup
        # On H200 (141GB), 0.2 = ~28GB, which is safe for 7B models.
        self.model = LLM(
            self.MODEL_DIR,
            dtype="bfloat16", 
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.2, 
        )

        # Aligned with repo worker/translator.py defaults
        self.recommended_sampling_params = SamplingParams(
            temperature=0.0,       
            max_tokens=1024,       # Repo default
            repetition_penalty=1.0 
        )
        self.deterministic_sampling_params = SamplingParams(
            temperature=0, 
            max_tokens=16384,
        )
        
        self.sampling_params = self.recommended_sampling_params if sampling == "recommended" else self.deterministic_sampling_params

    def _formalize_prompt(self, informal_problems: list[str]) -> Tuple[list[str], list[str]]:
        response = self.model.generate(informal_problems, sampling_params=self.sampling_params)
    
        lean_codes = []
        raw_outputs = []
        for i in range(len(informal_problems)):
            try:
                # Extract text from the first candidate output
                generated_text = response[i].outputs[0].text
                raw_outputs.append(generated_text)
                lean_code = self.parse_lean_code(generated_text)
            except Exception as e:
                lean_code = None
            lean_codes.append(lean_code)
        return lean_codes, raw_outputs
    
    def initialize_dialog(self):
        # Herald uses a specific system message structure
        self.dialog = [{'role': 'system', 'content': self.system_prompt}]

    def get_formal_statement_prompt(self, informal_problem: str) -> str:
        # Exact string match from Herald's prompt construction
        return (
            f"Please translate the natural language statement to Lean4 code with the header\n"
            f"**Name**\nmy_favorite_theorem\n"
            f"**Informal statement**\n{informal_problem}\n"
        )

    def format_dialog(self, dialog: list[dict]) -> str:
        """
        Replicates 'format_deepseek_msg' from frenzymath/herald_translator/utils.py
        """
        result = ""
        total_step = len(dialog)
        for i, message in enumerate(dialog):
            if message["role"] == "user":
                result += "User:" + message["content"] + "\n\n"
            elif message["role"] == "assistant":
                result += "Assistant" + message["content"] + "<｜end▁of▁sentence｜>"
            elif message["role"] == "system":
                result += message["content"] + "\n\n"
            if i + 1 == total_step and message["role"] == "user":
                result += "Assistant:"                
        return result


class GoedelFormalizer(BaseFormalizer):
    MODEL_DIR = "Goedel-LM/Goedel-Formalizer-V2-8B"

    def __init__(self, sampling="deterministic", base_url="http://localhost:8002/v1"):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_DIR)
        
        self.client = OpenAI(
            api_key="EMPTY", 
            base_url=base_url,
            timeout=3600,
        )
        
        self.recommended_kwargs = {
            "temperature": 0.9, 
            "top_p": 0.95,
            "max_tokens": 16384,
            "extra_body": {"top_k": 20}
        }
        self.deterministic_kwargs = {
            "temperature": 0.0, 
            "max_tokens": 16384,
        }
        self.gen_kwargs = self.recommended_kwargs if sampling == "recommended" else self.deterministic_kwargs

    def initialize_dialog(self):
        self.dialog = []

    def get_formal_statement_prompt(self, informal_problem: str) -> str:
        return (
            f"Please autoformalize the following natural language problem statement in Lean 4. "
            f"Use the following theorem name: my_favorite_theorem\n"
            f"The natural language statement is: \n"
            f"{informal_problem}\n\n"
            f"Think before you provide the lean statement."
        )

    def format_dialog(self, dialog: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=True)

    


class KiminaFormalizer(BaseFormalizer):
    MODEL_DIR = "AI-MO/Kimina-Autoformalizer-7B"

    def __init__(self, sampling="deterministic", base_url="http://localhost:8002/v1"):
        self.system_prompt = "You are an expert in mathematics and Lean 4."
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_DIR)
        
        self.client = OpenAI(
            api_key="EMPTY", 
            base_url=base_url,
            timeout=3600,
        )
        
        self.recommended_kwargs = {
            "temperature": 0.7, 
            "top_p": 0.8,
            "max_tokens": 16384,
            "extra_body": {"top_k": 20}
        }
        self.deterministic_kwargs = {
            "temperature": 0.0, 
            "max_tokens": 16384,
        }
        self.gen_kwargs = self.recommended_kwargs if sampling == "recommended" else self.deterministic_kwargs

    def initialize_dialog(self):
        self.dialog = [{"role": "system", "content": self.system_prompt}]

    def get_formal_statement_prompt(self, informal_problem: str) -> str:
        return (
            "Please autoformalize the following problem in Lean 4 with a header. "
            "Use the following theorem names: my_favorite_theorem.\n\n"
            f"{informal_problem}"
        )

    def format_dialog(self, dialog: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=True)

    
    
class StepfunFormalizer(BaseFormalizer):
    MODEL_DIR = "stepfun-ai/StepFun-Formalizer-7B"
    HEADER = "import Mathlib\n\nopen Real\n"

    def __init__(self, 
                 sampling="deterministic", 
                 base_url="http://localhost:8002/v1"):
        self.system_prompt = "You are an expert in mathematics and Lean 4."
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_DIR)
        self.client = OpenAI(
            api_key="EMPTY", 
            base_url=base_url,
            timeout=3600, # 1 hour timeout
        )
        self.recommended_kwargs = {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 16384,
        }
        
        self.deterministic_kwargs = {
            "temperature": 0.0,
            "max_tokens": 16384,
        }
        
        self.gen_kwargs = self.recommended_kwargs if sampling == "recommended" else self.deterministic_kwargs

    def initialize_dialog(self):
        self.dialog = [{"role": "system", "content": self.system_prompt}]

    def get_formal_statement_prompt(self, informal_problem: str) -> str:
        return (
            "Please autoformalize the following problem in Lean 4 with a header. "
            "Use the following theorem names: my_favorite_theorem.\n\n"
            f"{informal_problem}"
            f"\n\nYour code should start with:\n```Lean4\n{self.HEADER}\n```\n"
        )

    def format_dialog(self, dialog: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=True)
    
    
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
import os
from typing import List, Optional, Any
from .base import BaseFormalizer

# --- Load Prompt Template ---
# As requested, we assume this file exists and contains the prompt text you provided.
formalize_txt_path = os.path.join(os.path.dirname(__file__), 'formalize.txt')
with open(formalize_txt_path, 'r') as f:
    FORMALIZE_PROMPT = f.read()


# --- Compatibility Layer (Mocking vLLM) ---

class MockOutput:
    """Helper to mimic vLLM's RequestOutput structure."""
    def __init__(self, text: str):
        self.text = text

class MockRequestOutput:
    """Helper to mimic vLLM's RequestOutput structure."""
    def __init__(self, text: str):
        self.outputs = [MockOutput(text)]

class OpenAIWrapper:
    """Wraps OpenAI client to mimic vLLM's generate interface."""
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name

    def generate(self, prompt: str | List[dict], sampling_params: Any = None) -> List[MockRequestOutput]:
        # If input is a raw string (from format_dialog), wrap it as user message
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        temperature = getattr(sampling_params, 'temperature', 0.0)
        max_tokens = getattr(sampling_params, 'max_tokens', 1024)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content
            return [MockRequestOutput(content)]
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return [MockRequestOutput("")]

class GeminiWrapper:
    """Wraps Gemini client to mimic vLLM's generate interface."""
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        import google.generativeai as genai
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API Key is missing.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str | List[dict], sampling_params: Any = None) -> List[MockRequestOutput]:
        # Convert list-based dialog to string if necessary, though Gemini handles both.
        # For this formalization task, we usually send a single large string.
        if isinstance(prompt, list):
            final_input = ""
            for msg in prompt:
                final_input += f"{msg['role']}: {msg['content']}\n"
        else:
            final_input = prompt

        temperature = getattr(sampling_params, 'temperature', 0.0)
        max_tokens = getattr(sampling_params, 'max_tokens', 1024)
        
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        try:
            response = self.model.generate_content(
                final_input, 
                generation_config=generation_config
            )
            return [MockRequestOutput(response.text)]
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return [MockRequestOutput("")]


# --- Formalizer Implementations ---

class OpenAIFormalizer(BaseFormalizer):
    def __init__(self, model_name="gpt-4o", sampling="deterministic"):
        # We don't need a system prompt here because the logic is baked into FORMALIZE_PROMPT
        # which is sent as a user message.
        self.system_prompt = "You are a helpful assistant for formalizing mathematics."
        
        self.model = OpenAIWrapper(model_name)
        
        # Mocking vLLM SamplingParams for consistency
        from vllm import SamplingParams
        self.recommended_sampling_params = SamplingParams(
            temperature=0.7, 
            max_tokens=2048
        )
        self.deterministic_sampling_params = SamplingParams(
            temperature=0.0, 
            max_tokens=2048
        )
        self.sampling_params = self.recommended_sampling_params if sampling == "recommended" else self.deterministic_sampling_params

    def initialize_dialog(self):
        # We can start empty or with a system prompt. 
        # Given the strong few-shot prompting in FORMALIZE_PROMPT, 
        # a system prompt is optional but good practice.
        self.dialog = [{"role": "system", "content": self.system_prompt}]

    def get_formal_statement_prompt(self, informal_problem: str) -> str:
        # Strict adherence to the provided template
        return FORMALIZE_PROMPT.format(informal_problem=informal_problem)

    def format_dialog(self, dialog: list[dict]) -> List[dict]:
        # OpenAI expects a list of dicts directly
        return dialog


class GeminiFormalizer(BaseFormalizer):
    def __init__(self, model_name="gemini-1.5-pro", sampling="deterministic"):
        self.system_prompt = "You are a helpful assistant for formalizing mathematics."
        
        self.model = GeminiWrapper(model_name)
        
        from vllm import SamplingParams
        self.recommended_sampling_params = SamplingParams(
            temperature=0.7, 
            max_tokens=2048
        )
        self.deterministic_sampling_params = SamplingParams(
            temperature=0.0, 
            max_tokens=2048
        )
        self.sampling_params = self.recommended_sampling_params if sampling == "recommended" else self.deterministic_sampling_params

    def initialize_dialog(self):
        self.dialog = []

    def get_formal_statement_prompt(self, informal_problem: str) -> str:
        # Strict adherence to the provided template
        return FORMALIZE_PROMPT.format(informal_problem=informal_problem)

    def format_dialog(self, dialog: list[dict]) -> str:
        # Gemini often performs better with the raw string for large few-shot prompts
        # rather than splitting the few-shots into chat history turn-by-turn.
        # We extract the last user message (the prompt) to send.
        
        # If the pipeline appends the prompt to self.dialog, we extract it.
        # Assuming standard usage where the last message is the formatted prompt:
        if dialog and dialog[-1]['role'] == 'user':
            return dialog[-1]['content']
        return ""
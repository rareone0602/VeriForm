import random
import os
import re
import json
from typing import Tuple, Optional
from veriform.preprocessing.dag import DAGModel, Flagging
from veriform.perturbation.nl_negation import NegationBackend, get_backend
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file


class StandardPerturber:
    """
    Regex-based perturbation strategies for DAG nodes.
    """
    def __init__(
        self,
        p: float = 0.5, # Increased default p for visibility
        operator_swap: bool = True,
        value_change: bool = True,
        logical_negation: bool = True,
        negation_backend: Optional[NegationBackend] = None,
    ):
        self.p = p
        self.operator_swap = operator_swap
        self.value_change = value_change
        self.logical_negation = logical_negation
        self.negation_backend = negation_backend or get_backend("regex")
        
        # Robust number pattern: Matches floats, ints, negatives
        # Avoids matching "v1" or "3." at end of sentence by checking boundaries
        self.number_pattern = re.compile(r'(?<![a-zA-Z])(-?\d+(?:\.\d+)?)(?![a-zA-Z])')

    def operator_swap_str(self, content: str) -> Tuple[str, bool]:
        """
        Swap operators with robustness to whitespace.
        Handles '1+1=2' and '1 + 1 = 2'.
        """
        # Map now includes '==' and handles standard '=' with flexible spacing
        op_map = {
            '+': ['-', '*', '/'],
            '-': ['+', '*', '/'],
            '*': ['+', '-', '/'],
            '/': ['+', '-', '*'],
            '^': ['+', '*'],
            '<': ['='],
            '>': ['='],
            # '=': ['<', '>', '≠'],
        }
        
        # Sort keys by length descending to match '==' before '='
        ops = sorted(op_map.keys(), key=len, reverse=True)
        
        # Find all operator candidates
        # We look for operators that are NOT inside words (simple heuristic)
        candidates = []
        for op in ops:
            # Escape op but allow variable spacing for '='
            if op == '=':
                pattern = r'(?<![<>=!])=(?![=])' # strict single '='
            else:
                pattern = re.escape(op)
                
            for match in re.finditer(pattern, content):
                candidates.append((match.start(), match.end(), op))

        if not candidates:
            return content, False

        # Pick a random operator to swap (not just the last one, to add variety)
        # Weight towards the end of string as that's usually where the "result" lies
        candidate = candidates[-1] if random.random() > 0.3 else random.choice(candidates)
        start, end, op_char = candidate
        
        new_op = random.choice(op_map[op_char])
        
        # If it was a tight equals '1=2', keep it tight. If loose '1 = 2', keep loose.
        # This is a bit tricky, so we just blindly insert the new op with standard spacing
        # if the original was '=' to be safe, or raw replacement otherwise.
        if op_char == '=':
            repl = f" {new_op} " 
        else:
            repl = new_op

        return content[:start] + repl + content[end:], True

    def value_change_str(self, content: str) -> Tuple[str, bool]:
        """
        Change numerical values, avoiding 'Theorem 1.2' patterns.
        """
        matches = list(self.number_pattern.finditer(content))
        if not matches:
            return content, False

        # Filter out numbers that look like labels (integers at start of string)
        # or followed by colon e.g. "Problem 1:"
        valid_matches = []
        for m in matches:
            # If number is at very start and followed by logic-less text, skip (heuristic)
            if m.start() < 10 and ":" in content[m.end():m.end()+5]:
                continue
            valid_matches.append(m)

        if not valid_matches:
            return content, False

        # Target the last few numbers (usually the RHS answer)
        target_match = valid_matches[-1]
        
        original_str = target_match.group(1)
        is_float = '.' in original_str
        original_val = float(original_str)

        # Perturbation logic: 
        # Avoid identity (x + 0). Ensure we actually change the value.
        delta = random.choice([-1, 1]) * random.randint(1, max(1, int(abs(original_val) * 0.5)))
        if delta == 0: delta = 1 # Force change
        
        new_val = original_val + delta
        new_str = f"{new_val:.2f}".rstrip('0').rstrip('.') if is_float else str(int(new_val))

        return content[:target_match.start()] + new_str + content[target_match.end():], True

    def logical_negation_str(self, content: str) -> Tuple[str, bool]:
        """Delegate to the configured NegationBackend (regex by default).

        See src/veriform/perturbation/nl_negation/ for backend implementations.
        """
        return self.negation_backend.negate(content)

    def perturb_str(self, content: str) -> Tuple[str, bool]:
        applying_fn = []
        if self.value_change: applying_fn.append(self.value_change_str)
        if self.operator_swap: applying_fn.append(self.operator_swap_str)
        if self.logical_negation: applying_fn.append(self.logical_negation_str)

        # It is intentional to try in order rather than random
        # random.shuffle(applying_fn)

        has_changed = False

        for i, fn in enumerate(applying_fn):
            if i == 2 and has_changed:
                break  
            content, changed = fn(content)
            has_changed = has_changed or changed
            
                
        return content, has_changed

    def perturb(self, dag_input: DAGModel) -> DAGModel:
        """Concrete implementation of perturbation."""
        # Implementation details would go here

        for node in dag_input:
            if node.flag == Flagging.DECLARATIVE:
                node.perturbed_content, node.is_perturbed = node.content, False
                continue
            
            if random.random() < self.p:
                node.perturbed_content, node.is_perturbed = self.perturb_str(node.content)
                # With little probability, it will return a different content
                # We currently negelect that case as it is rare
                # So the actual p is slightly lower than specified
            else:
                node.perturbed_content, node.is_perturbed = node.content, False

        return dag_input


class DeepSeekPrePerturber:
    """
    Loader for DeepSeek Perturbation
    """
    def __init__(
        self,
        p: float = 0.5, # Increased default p for visibility
        file='data/raw/llm_perturbations.json',
        effort='medium'
    ):
        self.p = p
        self.effort = effort
        with open(file, 'r') as f:
            self.data = json.load(f)
    
    def perturb(self, dag_input: DAGModel) -> DAGModel:
        """Concrete implementation of perturbation."""
        # Implementation details would go here
        assert len(dag_input) == len(self.data[dag_input.id])
        for node, pert in zip(dag_input, self.data[dag_input.id]):
            node.perturbed_content, node.is_perturbed = pert[self.effort], True
        return dag_input


class BaseLLMPerturber:
    """
    Base class for LLM-based perturbation. 
    Handles the DAG iteration logic so we don't repeat code.
    """
    def __init__(self, p: float = 0.5, api_key: Optional[str] = None, model_name: str = ""):
        self.p = p
        self.api_key = api_key or os.getenv("API_KEY")
        self.model_name = model_name
        self.system_prompt = (
            "You are a mathematical adversary. Your task is to take a specific "
            "mathematical or logical statement that is TRUE and subtly modify it "
            "so that it becomes FALSE. \n"
            "Rules:\n"
            "1. Maintain the original LaTeX formatting, variable names, and style exactly.\n"
            "2. Make the change subtle (e.g., flip an inequality, change a constant, swap a quantifier).\n"
            "3. Do NOT explain your reasoning.\n"
            "4. Output ONLY the modified statement."
        )

    def _call_llm(self, content: str) -> Optional[str]:
        """Override this in child classes."""
        raise NotImplementedError

    def perturb_str(self, content: str) -> Tuple[str, bool]:
        """
        Attempts to perturb the string via LLM. 
        Returns (original_content, False) if the API call fails or decides not to change.
        """
        try:
            # We add a small retry or delay logic if needed here
            perturbed = self._call_llm(content)
            
            # Sanity check: if LLM returns nothing or the same string
            if not perturbed or perturbed.strip() == content.strip():
                return content, False
                
            return perturbed.strip(), True
        except Exception as e:
            print(f"Warning: LLM perturbation failed for '{content[:20]}...': {e}")
            return content, False

    def perturb(self, dag_input: DAGModel) -> DAGModel:
        """
        Iterates through the DAG and applies perturbation based on probability p.
        """
        
        for node in dag_input:
            # Skip declarative nodes (context/assumptions)
            if hasattr(node, 'flag') and node.flag == Flagging.DECLARATIVE:
                node.perturbed_content, node.is_perturbed = node.content, False
                continue
            
            # Apply perturbation with probability p
            if random.random() < self.p:
                node.perturbed_content, node.is_perturbed = self.perturb_str(node.content)
            else:
                node.perturbed_content, node.is_perturbed = node.content, False

        return dag_input




class GeminiPerturber(BaseLLMPerturber):
    """
    Google Gemini implementation.
    """
    def __init__(self, p: float = 0.5, api_key: Optional[str] = None, model_name: str = "gemini-3-flash-preview"):
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        super().__init__(p, api_key, model_name)
        import google.generativeai as genai
        if not self.api_key:
            raise ValueError("Gemini API Key is missing. Set it in init or env vars.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def _call_llm(self, content: str) -> Optional[str]:
        # Gemini handles system instructions differently in some versions, 
        # but prepending it to the prompt is the robust way for now.
        full_prompt = f"{self.system_prompt}\n\nStatement: {content}\nFalsified Statement:"
        
        response = self.model.generate_content(full_prompt)
        return response.text


class OpenAIPerturber(BaseLLMPerturber):
    """
    OpenAI GPT implementation.
    """
    def __init__(self, p: float = 0.5, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini"):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        super().__init__(p, api_key, model_name)
        from openai import OpenAI
        if not self.api_key:
            raise ValueError("OpenAI API Key is missing. Set it in init or env vars.")
        self.client = OpenAI(api_key=self.api_key)

    def _call_llm(self, content: str) -> Optional[str]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Statement: {content}"}
            ],
        )
        return response.choices[0].message.content



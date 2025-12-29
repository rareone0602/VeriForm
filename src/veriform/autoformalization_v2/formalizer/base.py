from typing import Optional, Protocol
from abc import abstractmethod, ABC
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from ..dag import DAGModel, Flagging

class BaseFormalizer(ABC):
    # Regex Explanation:
    # ^theorem    : Matches 'theorem' at the start of a line (excludes imports).
    # [\s\S]*?    : Matches any char (including newlines) non-greedily.
    # sorry       : Stops exactly at the 'sorry' keyword.
    THEOREM_PATTERN = re.compile(r"(^theorem[\s\S]*?sorry)", re.MULTILINE | re.IGNORECASE)
    
    @abstractmethod
    def get_formal_statement_prompt(self, informal_problem: str) -> str:
        ...

    @abstractmethod
    def initialize_dialog(self):
        ...

    @abstractmethod
    def format_dialog(self, dialog: list[dict]) -> str:
        ...

    def parse_lean_code(self, response_text: str) -> str:
        # Find all valid theorem blocks ending in sorry
        matches = self.THEOREM_PATTERN.findall(response_text)
        if matches:
            # Return the last one, stripped of leading/trailing whitespace
            return matches[-1].strip()
        else:
            raise ValueError(f"No Lean 4 theorem found in response:\n{response_text[:200]}...")
        
    def _get_llm_prompt(self, informal_problem: str) -> str:
        user_prompt = self.get_formal_statement_prompt(informal_problem)
        self.dialog.append({"role": "user", "content": user_prompt})
        prompt = self.format_dialog(self.dialog)
        self.dialog.pop()
        return prompt
    
    def _formalize_prompt(self, informal_problems: list[str]) -> list[str]:
        response = self.model.generate(informal_problems, sampling_params=self.sampling_params)
        lean_codes = []
        for i in range(len(informal_problems)):
            try:
                # Extract text from the first candidate output
                generated_text = response[i].outputs[0].text
                lean_code = self.parse_lean_code(generated_text)
            except Exception as e:
                lean_code = None
            lean_codes.append(lean_code)
        return lean_codes

    def formalize(self, dag: DAGModel, cleanup_dialog: bool = True) -> DAGModel:
        prompts = []
        nodes = []
        for i, node in enumerate(dag.nodes):
            if node.flag == Flagging.DECLARATIVE:
                continue  # Skip declarative nodes
            if cleanup_dialog:
                self.initialize_dialog()
            informal_problem = node.contextualized()

            prompts.append(self._get_llm_prompt(informal_problem))
            nodes.append(node)
            
        if not prompts:
            return dag

        lean_codes = self._formalize_prompt(prompts)
        
        for node, lean_code, i in zip(nodes, lean_codes, range(len(lean_codes))):
            if lean_code is None:
                node.formalized_content = "-- Failed to formalize"
                node.flag = Flagging.AF_FAIL
            else:
                # Replace the placeholder name with a unique step name
                node.formalized_content = lean_code.replace("my_favorite_theorem", f"step_{i}")
                # Reset flag if it was previously failed, or keep as is
                if node.flag == Flagging.AF_FAIL: 
                    node.flag = Flagging.UNKNOWN 

        return dag
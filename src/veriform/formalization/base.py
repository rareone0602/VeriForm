from abc import abstractmethod, ABC
import re
from typing import Tuple
from vllm import SamplingParams
from veriform.preprocessing.dag import DAGModel, Flagging

class BaseFormalizer(ABC):
    # Regex Explanation:
    # ^theorem    : Matches 'theorem' at the start of a line (excludes imports).
    # [\s\S]*?    : Matches any char (including newlines) non-greedily.
    # sorry       : Stops exactly at the 'sorry' keyword.
    THEOREM_PATTERN = re.compile(r"(^theorem[\s\S]*?sorry)", re.MULTILINE | re.IGNORECASE)
    deterministic_sampling_params = SamplingParams(
        temperature=0, 
        max_tokens=16384,
    )
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
    
    def _formalize_prompt(self, informal_problems: list[str]) -> Tuple[list[str], list[str]]:
        # A good rule of thumb for batching: vLLM accepts a list of string prompts directly.
        response = self.client.completions.create(
            model=self.MODEL_DIR, # Ensure this matches the model name served in vLLM
            prompt=informal_problems,
            **self.gen_kwargs
        )

        lean_codes = []
        raw_outputs = []
        
        # Iterate through the returned choices
        for choice in response.choices:
            try:
                generated_text = choice.text.strip()
                raw_outputs.append(generated_text)
                lean_code = self.parse_lean_code(generated_text)
            except Exception as e:
                lean_code = None
            lean_codes.append(lean_code)
            
        return lean_codes, raw_outputs

    def formalize(self, dag: DAGModel, cleanup_dialog: bool = True) -> DAGModel:
        prompts = []
        nodes = []
        for i, node in enumerate(dag.nodes):
            if node.flag == Flagging.DECLARATIVE:
                continue  # Skip declarative nodes
            if cleanup_dialog:
                self.initialize_dialog()
            informal_problem = node.contextualized()

            prompt = self._get_llm_prompt(informal_problem)

            if len(prompt) > 4000:
                node.formalized_content = "-- Failed to formalize"
                node.flag = Flagging.AF_FAIL
                node.formalizer_output = None
                continue

            prompts.append(prompt)
            nodes.append(node)
            
        if not prompts:
            return dag

        lean_codes, raw_outputs = self._formalize_prompt(prompts)
        
        for node, lean_code, raw_output in zip(nodes, lean_codes, raw_outputs):
            if lean_code is None:
                node.formalized_content = "-- Failed to formalize"
                node.flag = Flagging.AF_FAIL
                node.formalizer_output = raw_output
            else:
                # Replace the placeholder name with a unique step name
                node.formalized_content = lean_code
                print(lean_code)
                node.flag = Flagging.UNKNOWN 
                node.formalizer_output = raw_output
        return dag
from ..dag import DAGModel, Flagging
from abc import abstractmethod, ABC

class BaseFormalizer(ABC):
    
    @abstractmethod
    def get_formal_statement_prompt(self, informal_problem: str) -> str:
        ...

    @abstractmethod
    def parse_lean_code(self, response_text: str) -> str:
        ...

    @abstractmethod
    def initialize_dialog(self):
        ...

    @abstractmethod
    def format_dialog(self, dialog: list[dict]) -> str:
        ...

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
                lean_code = self.parse_lean_code(response[i].outputs[0].text)
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
        lean_codes = self._formalize_prompt(prompts)
        for node, lean_code, i in zip(nodes, lean_codes, range(len(lean_codes))):
            if lean_code is None:
                node.formalized_content = "-- Failed to formalize"
                node.flag = Flagging.AF_FAIL
            else:
                node.formalized_content = lean_code
            node.formalized_content = node.formalized_content.replace(
                "my_favorite_theorem", 
                f"step_{i}")
        return dag
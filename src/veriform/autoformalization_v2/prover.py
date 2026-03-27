import re
from typing import List, Protocol, Optional, Tuple
from dataclasses import dataclass, field

import asyncio
import backoff
from abc import abstractmethod
from transformers import AutoTokenizer
import openai
from openai import OpenAI 

from .deepseek.prover.lean.verifier import Lean4ServerScheduler
from .theorem_extractor import TheoremExtractor
from .dag import DAGModel, Flagging

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Prover(Protocol):
    @abstractmethod
    def prove(self, dag: DAGModel) -> DAGModel:
        ...

HEADER = """
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat
"""

LEAN_TEMPLATE = HEADER + """

/- {statement} -/
{lean_code}
""".strip()

LEAN_WRAPPER_TEMPLATE = """
Complete the following Lean 4 code:

```lean4
{lean_code}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()

@dataclass
class TheoremData:
    flag: Flagging = Flagging.UNKNOWN

    tc_request_id: Optional[int] = None

    problem_nl_dir : Optional[str] = None
    prompt_dir: Optional[str] = None
    param_dir: Optional[str] = None
    body_dir: Optional[str] = None
    
    problem_nl_neg: Optional[str] = None
    prompt_neg: Optional[str] = None
    param_neg: Optional[str] = None
    body_neg: Optional[str] = None
    
    proof_candidates_dir: List[Tuple[str, str]] = field(default_factory=list)
    proof_candidates_neg: List[Tuple[str, str]] = field(default_factory=list)


"""
Run this first
CUDA_VISIBLE_DEVICES=2 vllm serve deepseek-ai/DeepSeek-Prover-V2-7B \
    --port 8002 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.5 \
    --max-model-len 16384
"""
class DeepSeekProver:
    MODEL_NAME = "deepseek-ai/DeepSeek-Prover-V2-7B"
    def __init__(self, 
                 base_url: str = "http://localhost:8000/v1", 
                 batch_size: int = 1, 
                 negation_type: str = 'full'):
        self.negation_type = negation_type

        # We keep the tokenizer LOCALLY just for formatting the prompt correctly
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        
        # Initialize the API Client
        self.client = OpenAI(
            api_key="EMPTY", 
            base_url=base_url,
            timeout=7200, # 1 hour timeout
        )
        
        self.lean_pattern = re.compile(r"```lean4?(.*?)```", re.DOTALL | re.IGNORECASE)
        self.batch_size = batch_size

        self.scheduler = Lean4ServerScheduler(max_concurrent_requests=64)
        self.theorem_extractor = TheoremExtractor()

        # Generation config (moved from SamplingParams)
        self.gen_kwargs = {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 8192,
            "n": self.batch_size, # Request 'n' completions per prompt
            # "stop": ["\n\n"] # Add stop tokens if needed
        }
    
    def __del__(self):
        if hasattr(self, 'scheduler'):
            self.scheduler.close()

    def parse_lean_code(self, response_text: str) -> str:
        """
        Get the LAST Lean 4 code block from the response text.
        """
        matches = self.lean_pattern.findall(response_text)
        if matches:
            return matches[-1].strip()
        else:
            raise ValueError("No Lean 4 code block found in the response.")

    def initialize_dialog(self):
        self.dialog = []

    def get_tc_async(self, formalized_problem: str, header: str = HEADER) -> int:
        lean_code = header + "\n\n" + formalized_problem
        request_id = self.scheduler.submit_request(
            dict(code=lean_code, ast=False, tactics=False)
        )
        return request_id
    
    def get_tc_result(self, request_id: int) -> bool:
        result = self.scheduler.get_request_outputs(request_id)
        return result['pass']
    
    def get_refute_theorem(self, formal_problem: str) -> str:
        """
        Generates a negation of the given theorem.
        
        Args:
            formal_problem: The raw Lean 4 code of the original theorem.
            negation_type: 
                - 'strong': 'theorem not_T (x : A) : ¬ P' 
                  (Asserts P is false for ALL inputs).
                - 'full': 'theorem not_T_full : ¬ (∀ (x : A), P)' 
                  (Asserts P is NOT TRUE for all inputs; i.e., a counter-example exists).
        """
        name, param, body = self.theorem_extractor.get_last_theorem(formal_problem)
        
        if name is None or body is None:
            raise ValueError("Cannot extract theorem name/body from formal problem.")
        
        # Clean up params: handle None and whitespace
        param_str = param.strip() if param else ""
        
        if self.negation_type == 'strong':
            # Strategy: Keep params in signature, negate the body only.
            # Good for specific values (e.g. "theorem t : 1 = 2")
            refute_name = f"not_{name}_strong"
            # Re-attach params to the name if they exist
            declaration_params = f" {param_str}" if param_str else ""
            refute_body = f"¬ ({body})"
            
            return f"theorem {refute_name}{declaration_params} : {refute_body} := by sorry"

        elif self.negation_type == 'full':
            # Strategy: Remove params from signature, wrap them in ∀, and negate the whole lot.
            # Good for general laws (e.g. "theorem t (n : Nat) : n > 0")
            refute_name = f"not_{name}_full"
            
            if param_str:
                # We have parameters, so we construct: ¬ (∀ (x : T), body)
                # Note: In Lean 4, '∀ (x : T) (y : U), P' is valid syntax.
                refute_body = f"¬ (∀ {param_str}, {body})"
            else:
                # No parameters? Then Full Negation is identical to Strong Negation.
                refute_body = f"¬ ({body})"

            return f"theorem {refute_name} : {refute_body} := by sorry"
            
        else:
            raise ValueError("Invalid negation type provided.")


    def filter_proof_candidates(self, target_param: Optional[str], target_body: Optional[str], proof_candidates: List[str], comment: str = '') -> List[Tuple[str, str]]:
        valid_candidates = []

        def _normalize(s: Optional[str]) -> str:
            if not s: return ""
            return " ".join(s.split())

        for candidate in proof_candidates:
            try:
                lean_code = self.parse_lean_code(candidate)
            except ValueError:
                continue
            proof_name, proof_param, proof_body = self.theorem_extractor.get_last_theorem(lean_code)
            if _normalize(proof_param) != _normalize(target_param) or _normalize(proof_body) != _normalize(target_body):
                continue
            valid_candidates.append(
                (
                    lean_code,
                    LEAN_TEMPLATE.format(
                        lean_code=lean_code,
                        statement=comment
                    )
                )
            )
        return valid_candidates
        
        
    def get_proof_prompts(self, problem_nl_dir: str, problem_fl_dir: str) -> TheoremData:
        # This method also remains UNCHANGED.
        # We still use self.tokenizer locally to format the chat template into a string.
        name_dir, param_dir, body_dir = self.theorem_extractor.get_last_theorem(problem_fl_dir)
        
        if name_dir is None or body_dir is None:
            return TheoremData(flag=Flagging.AF_FAIL)
        
        user_prompt_dir = LEAN_WRAPPER_TEMPLATE.format(
            lean_code=LEAN_TEMPLATE.format(lean_code=problem_fl_dir, statement=problem_nl_dir)
        )
        problem_fl_neg = self.get_refute_theorem(problem_fl_dir)
        problem_nl_neg = "Negation of: " + problem_nl_dir
        name_neg, param_neg, body_neg = self.theorem_extractor.get_last_theorem(problem_fl_neg)

        user_prompt_neg = LEAN_WRAPPER_TEMPLATE.format(
            lean_code=LEAN_TEMPLATE.format(lean_code=problem_fl_neg, statement=problem_nl_neg)
        )
        
        self.dialog.append({"role": "user", "content": user_prompt_dir})
        prompt_dir = self.tokenizer.apply_chat_template(self.dialog, tokenize=False, add_generation_prompt=True)
        self.dialog.pop()
        self.dialog.append({"role": "user", "content": user_prompt_neg})
        prompt_neg = self.tokenizer.apply_chat_template(self.dialog, tokenize=False, add_generation_prompt=True)

        return TheoremData(
            problem_nl_dir=problem_nl_dir, prompt_dir=prompt_dir, param_dir=param_dir, body_dir=body_dir,
            problem_nl_neg=problem_nl_neg, prompt_neg=prompt_neg, param_neg=param_neg, body_neg=body_neg
        )

    @backoff.on_exception(
        backoff.expo, 
        (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError),
        max_tries=5,
        jitter=backoff.full_jitter
    )
    def _generate_batch(self, prompts: List[str]):
        # No 'await' here. This line blocks until completion.
        return self.client.completions.create(
            model=self.MODEL_NAME,
            prompt=prompts,
            **self.gen_kwargs
        )

    def get_proof_outputs(self, data: List[TheoremData]) -> List[TheoremData]:
        prompts = []
        data_order = []
        
        # 1. Collect all prompts
        for i, item in enumerate(data):
            if item.flag in [Flagging.DECLARATIVE, Flagging.AF_FAIL, Flagging.TC_FAIL]:
                continue
            prompts.extend([item.prompt_dir, item.prompt_neg])
            data_order.append(i)

        if not prompts:
            return data

        # 2. Call the API (Async wrapper to handle IO latency)
        # We run the async function in a blocking way to match your existing synchronous structure
        
        response = self._generate_batch(prompts)
        response_choices = response.choices
        # [dir_1_1, ..., dir_1_b, neg_1_1, ..., neg_1_b, dir_2_1, ..., dir_2_b, neg_2_1, ..., neg_2_b, ...]
        b = self.batch_size 
        # 3. Process results
        
        for i, idx in enumerate(data_order):
            item = data[idx]

            resp_dir = response_choices[2*b*i:2*b*i + b]
            candidates_dir = [choice.text.strip() for choice in resp_dir]

            resp_neg = response_choices[2*b*i+b:2*b*(i+1)]
            candidates_neg = [choice.text.strip() for choice in resp_neg]
            item.proof_candidates_dir = self.filter_proof_candidates(
                target_param=item.param_dir,
                target_body=item.body_dir,
                proof_candidates=candidates_dir,
                comment=item.problem_nl_dir
            )
            item.proof_candidates_neg = self.filter_proof_candidates(
                target_param=item.param_neg,
                target_body=item.body_neg,
                proof_candidates=candidates_neg,
                comment=item.problem_nl_neg
            )
            data[idx] = item
        
        return data
    
    def submit_proof(self, data: List[TheoremData]) -> List[TheoremData]:
        to_submit = []
        data_num = []
        data_order = []

        for i, item in enumerate(data):
            if item.flag == Flagging.AF_FAIL or item.flag == Flagging.TC_FAIL:
                continue
            to_submit.extend(
                [dict(code=full_lean_code, ast=False, tactics=False)
                for _, full_lean_code in item.proof_candidates_dir]
            )
            data_num.append(len(to_submit))
            to_submit.extend(
                [dict(code=full_lean_code, ast=False, tactics=False)
                for _, full_lean_code in item.proof_candidates_neg]
            )
            data_num.append(len(to_submit))
            data_order.append(i)
        request_ids = self.scheduler.submit_all_request(to_submit)
        all_results = self.scheduler.get_all_request_outputs(request_ids)

        for i, idx in enumerate(data_order):
            item = data[idx]
            mid = data_num[2*i]
            results_dir = all_results[data_num[2*i-1]:mid] if i > 0 else all_results[:mid]
            results_neg = all_results[mid:data_num[2*i+1]]

            valid_proof = None
            for (code, _), result in zip(item.proof_candidates_dir, results_dir):
                if result['complete']:
                    item.flag = Flagging.PROVED
                    item.proof_candidates_dir = [(code, '')]
                    valid_proof = code
                    break
            
            if valid_proof is None:
                for (code, _), result in zip(item.proof_candidates_neg, results_neg):
                    if result['complete']:
                        item.flag = Flagging.REFUTED
                        item.proof_candidates_neg = [(code, '')]
                        valid_proof = code
                        break
            
            if valid_proof is None:
                item.flag = Flagging.UNKNOWN

            data[idx] = item
        
        return data


    def prove(self, dag: DAGModel, formalizer_header: str = HEADER) -> DAGModel:
        taskqueue = []
        for node in dag.nodes:
            if node.flag == Flagging.DECLARATIVE or node.flag == Flagging.AF_FAIL or len(node.formalized_content) > 10000:
                continue

            self.initialize_dialog()

            proof_request = self.get_proof_prompts(
                problem_nl_dir=node.perturbed_content,
                problem_fl_dir=node.formalized_content
            )

            if proof_request.flag == Flagging.AF_FAIL:
                node.flag = Flagging.AF_FAIL
                continue
            
            proof_request.tc_request_id = self.get_tc_async(
                formalized_problem=node.formalized_content,
                header=formalizer_header
            )

            tc_testing_id = self.get_tc_async(
                formalized_problem=self.get_refute_theorem(node.formalized_content),
                header=formalizer_header
            )

            taskqueue.append( (node, proof_request, tc_testing_id) )
        
        # Assert that none of the nodes in taskqueue are AF_FAIL
        assert all(node.flag != Flagging.AF_FAIL for node, _, _ in taskqueue)
        # First check tc-fail, mark them
        for node, proof_request, tc_testing_id in taskqueue:
            tc_result = self.get_tc_result(proof_request.tc_request_id)
            tc_testing_result = self.get_tc_result(tc_testing_id)

            if not tc_testing_result and tc_result:
                print(self.get_refute_theorem(node.formalized_content))
                print(node.formalized_content)

            if not tc_result:
                proof_request.flag = node.flag = Flagging.TC_FAIL
                continue
        
        # Prepare data for proof
        proof_attempt = [
            proof_request for _, proof_request, _ in taskqueue
        ]
        proof = self.get_proof_outputs(proof_attempt)
        proof = self.submit_proof(proof)

        for (node, _, _), proof_request in zip(taskqueue, proof):
            node.flag = proof_request.flag
            if proof_request.flag in [Flagging.PROVED, Flagging.REFUTED]:
                if proof_request.flag == Flagging.PROVED:
                    node.formalized_content = proof_request.proof_candidates_dir[0][0]
                else:
                    node.formalized_content = proof_request.proof_candidates_neg[0][0]

        # And then process proof in batches
    
        return dag
import re
from typing import List, Protocol, Optional, Tuple
from dataclasses import dataclass, field

import asyncio
import backoff
from abc import abstractmethod
from transformers import AutoTokenizer
import openai
from openai import AsyncOpenAI  # Changed from vllm

from .deepseek.prover.lean.verifier import Lean4ServerScheduler
from .theorem_extractor import TheoremExtractor
from .dag import DAGModel, Flagging

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
CUDA_VISIBLE_DEVICES=0 vllm serve deepseek-ai/DeepSeek-Prover-V2-7B \
    --port 8000 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.64 \
    --max-model-len 16384
"""
class DeepSeekProver:
    MODEL_NAME = "deepseek-ai/DeepSeek-Prover-V2-7B"
    def __init__(self, base_url: str = "http://localhost:8000/v1", batch_size: int = 1):
        self.system_prompt = "You are an expert in mathematics and Lean 4."
        
        # We keep the tokenizer LOCALLY just for formatting the prompt correctly
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        
        # Initialize the API Client
        self.client = AsyncOpenAI(api_key="EMPTY", base_url=base_url)
        
        self.lean_pattern = re.compile(r"```lean4(.*?)```", re.DOTALL | re.IGNORECASE)
        self.batch_size = batch_size

        self.theorem_pattern = re.compile(r"theorem\s+(\w+)(.*?):\s*(.*?)\s*:=", re.DOTALL)
        self.scheduler = Lean4ServerScheduler()
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
        name, param, body = self.theorem_extractor.get_last_theorem(formal_problem)
        if name is None or body is None:
            raise ValueError("Cannot extract theorem name/body from formal problem.")
        
        refute_name = f"not_{name} {param if param is not None else ''}".strip()
        refute_body = f"¬ ({body})"

        refute_theorem = f"theorem {refute_name} : {refute_body} := by sorry"
        return refute_theorem


    def filter_proof_candidates(self, target_param: Optional[str], target_body: Optional[str], proof_candidates: List[str], comment: str = '') -> List[Tuple[str, str]]:
        valid_candidates = []
        for candidate in proof_candidates:
            try:
                lean_code = self.parse_lean_code(candidate)
            except ValueError:
                continue
            proof_name, proof_param, proof_body = self.theorem_extractor.get_last_theorem(lean_code)
            if proof_param != target_param or proof_body != target_body:
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
    async def _safe_request(self, prompt: str):
        """
        Sends a single request with automatic retries for transient errors.
        """
        return await self.client.completions.create(
            model=self.MODEL_NAME,
            prompt=prompt,
            **self.gen_kwargs
        )

    async def _generate_batch_async(self, prompts: List[str]):
        """Helper to send requests to the API in parallel with robustness."""
        tasks = []
        for prompt in prompts:
            # We call the decorated helper function here
            tasks.append(self._safe_request(prompt))
        
        # return_exceptions=True ensures that if one request permanently fails 
        # (e.g. Context Length Error), it doesn't crash the whole batch immediately.
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Optional: Handle any remaining exceptions (like 400 Bad Request)
        final_results = []
        for res in results:
            if isinstance(res, Exception):
                print(f"Request failed permanently: {res}")
                # You might want to return a dummy object or re-raise
                # creating a dummy object to prevent downstream crashes:
                class DummyChoice: text = ""
                class DummyResponse: choices = [DummyChoice()] * self.batch_size
                final_results.append(DummyResponse())
            else:
                final_results.append(res)
                
        return final_results

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
        response_objects = asyncio.run(self._generate_batch_async(prompts))
        
        # 3. Process results
        # The API returns a list of completion objects.
        # response_objects[0] is for prompts[0], etc.
        
        for i, idx in enumerate(data_order):
            item = data[idx]
            
            # The direct proof response is at index 2*i
            resp_dir = response_objects[2*i]
            candidates_dir = [choice.text for choice in resp_dir.choices]

            # The negation proof response is at index 2*i+1
            resp_neg = response_objects[2*i+1]
            candidates_neg = [choice.text for choice in resp_neg.choices]

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
            if node.flag == Flagging.DECLARATIVE or node.flag == Flagging.AF_FAIL:
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

            taskqueue.append( (node, proof_request) )
        
        # Assert that none of the nodes in taskqueue are AF_FAIL
        assert all(node.flag != Flagging.AF_FAIL for node, _ in taskqueue)
        # First check tc-fail, mark them
        for node, proof_request in taskqueue:
            tc_result = self.get_tc_result(proof_request.tc_request_id)
            if not tc_result:
                proof_request.flag = node.flag = Flagging.TC_FAIL
                continue
        
        # Prepare data for proof
        proof_attempt = [
            proof_request for _, proof_request in taskqueue
        ]
        proof = self.get_proof_outputs(proof_attempt)
        proof = self.submit_proof(proof)

        for (node, _), proof_request in zip(taskqueue, proof):
            node.flag = proof_request.flag
            if proof_request.flag in [Flagging.PROVED, Flagging.REFUTED]:
                if proof_request.flag == Flagging.PROVED:
                    node.formalized_content = proof_request.proof_candidates_dir[0][0]
                else:
                    node.formalized_content = proof_request.proof_candidates_neg[0][0]

        # And then process proof in batches
    
        return dag
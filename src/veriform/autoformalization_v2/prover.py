


from typing import List, Protocol, Optional, Tuple

from .deepseek.prover.lean.verifier import Lean4ServerScheduler
from .dag import DAGModel, Flagging
from abc import abstractmethod
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import re



class TheoremExtractor:
    def __init__(self):
        # EXPLANATION OF REGEX:
        # theorem\s+(\w+)   : Matches 'theorem' and the name.
        # \s+               : Must be some space after name.
        # (?P<sig>          : Start capturing the 'signature' (params + type).
        #   (?:(?!:=).)* : THE KEY FIX. "Match any char that does NOT start ':=', repeatedly."
        # )                 : End capture.
        # \s*:=             : Stop exactly at the definition operator.
        self.theorem_pattern = re.compile(r"theorem\s+(\w+)\s+(?P<sig>(?:(?!:=).)*)\s*:=", re.DOTALL)

    def _split_signature(self, signature: str) -> Tuple[Optional[str], str]:
        """
        Splits the signature (e.g., "(n : Nat) : n = n") into params and body.
        It looks for the first colon that is NOT nested inside (), {}, or [].
        """
        balance = 0
        # Iterate through the string to find the 'top-level' colon
        for i, char in enumerate(signature):
            if char in "([{":
                balance += 1
            elif char in ")]}":
                balance -= 1
            elif char == ":" and balance == 0:
                # We found the separator!
                params = signature[:i].strip()
                body = signature[i+1:].strip()
                # If params is empty strings, make it None for consistency
                return (params if params else None, body)
        
        # Fallback: if no colon found (unlikely in valid Lean), assume entire sig is body
        return None, signature.strip()

    def get_last_theorem(self, code: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extracts the LAST theorem occurrence reliably.
        """
        matches = list(self.theorem_pattern.finditer(code))
        
        if not matches:
            return None, None, None

        # Take the last match
        last_match = matches[-1]
        name = last_match.group(1)
        raw_signature = last_match.group('sig')

        # Use logic to split params and body safely
        params, body = self._split_signature(raw_signature)
        
        return name, params, body


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



class DeepSeekProver:
    MODEL_DIR = "deepseek-ai/DeepSeek-Prover-V2-7B"
    def __init__(self, batch_size: int = 1):
        self.system_prompt = "You are an expert in mathematics and Lean 4."
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_DIR)
        self.model = LLM(
            self.MODEL_DIR,
            dtype="bfloat16", 
            tensor_parallel_size=2,
            gpu_memory_utilization=0.2,
        )
        self.lean_pattern = re.compile(r"```lean4(.*?)```", re.DOTALL | re.IGNORECASE)
        self.batch_size = batch_size

        # 'theorem mathd_algebra_10 : abs ((120 : ℝ) / 100 * 30 - 130 / 100 * 20) = 10 := by ...' => 'theorem mathd_algebra_10 : abs ((120 : ℝ) / 100 * 30 - 130 / 100 * 20) = 10'
        self.theorem_pattern = re.compile(r"theorem\s+(\w+)(.*?):\s*(.*?)\s*:=", re.DOTALL)

        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            max_tokens=8192,
            n=self.batch_size
        )

        self.scheduler = Lean4ServerScheduler(
            max_concurrent_requests=16,
        )

        self.theorem_extractor = TheoremExtractor()
    
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
        
        
    

        
    def prove_single_async(self, problem_nl_dir: str, problem_fl_dir: str) -> Optional[tuple[List[Tuple[str, int]], List[Tuple[str, int]]]]:
        name_dir, param_dir, body_dir = self.theorem_extractor.get_last_theorem(problem_fl_dir)
        
        if name_dir is None or body_dir is None:
            print(problem_fl_dir) # Debugging info
            return None
        
        problem_fl_neg = self.get_refute_theorem(problem_fl_dir)
        problem_nl_neg = "Negation of: " + problem_nl_dir
        name_neg, param_neg, body_neg = self.theorem_extractor.get_last_theorem(problem_fl_neg)
        
        
        user_prompt_dir = LEAN_WRAPPER_TEMPLATE.format(
            lean_code=LEAN_TEMPLATE.format(
                lean_code=problem_fl_dir,
                statement=problem_nl_dir
            )
        )
        user_prompt_neg = LEAN_WRAPPER_TEMPLATE.format(
            lean_code=LEAN_TEMPLATE.format(
                lean_code=problem_fl_neg,
                statement=problem_nl_neg
            )
        )
        
        self.dialog.append({"role": "user", "content": user_prompt_dir})
        prompt_dir = self.tokenizer.apply_chat_template(self.dialog, tokenize=False, add_generation_prompt=True)
        self.dialog.pop()  # Remove last user prompt
        self.dialog.append({"role": "user", "content": user_prompt_neg})
        prompt_neg = self.tokenizer.apply_chat_template(self.dialog, tokenize=False, add_generation_prompt=True)

        response = self.model.generate(
            [prompt_dir, prompt_neg], 
            sampling_params=self.sampling_params)
        # Append model response to dialog

        proof_candidates_dir = self.filter_proof_candidates(
            target_param=param_dir,
            target_body=body_dir,
            proof_candidates=[response[0].outputs[i].text for i in range(self.batch_size)],
            comment=problem_nl_dir
        )
        proof_candidates_neg = self.filter_proof_candidates(
            target_param=param_neg,
            target_body=body_neg,
            proof_candidates=[response[1].outputs[i].text for i in range(self.batch_size)],
            comment=problem_nl_neg
        )

        request_ids = self.scheduler.submit_all_request(
            [
                dict(code=code, ast=False, tactics=False)
                for _, code in proof_candidates_dir
            ] + [
                dict(code=code, ast=False, tactics=False)
                for _, code in proof_candidates_neg
            ]
        )
        requests_ids_dir, requests_ids_neg = (
            request_ids[:len(proof_candidates_dir)],
            request_ids[len(proof_candidates_dir):]
        )
        return (
            list(zip([code for code, _ in proof_candidates_dir], requests_ids_dir)),
            list(zip([code for code, _ in proof_candidates_neg], requests_ids_neg))
        )
        
    def prove_single_result(self, 
                            prove_requests: tuple[List[Tuple[str, int]], List[Tuple[str, int]]],
                            formalized_problem: str) -> tuple[str, Flagging]:
        prove_requests_dir, prove_requests_neg = prove_requests
        mid = len(prove_requests_dir)
        all_ids = [req_id for _, req_id in prove_requests_dir] + [req_id for _, req_id in prove_requests_neg]
        all_results = self.scheduler.get_all_request_outputs(all_ids)
        results_dir, results_neg = (
            all_results[:mid],
            all_results[mid:]
        )
        # Check direct proofs first
        valid_proof = None
        for (code, _), result in zip(prove_requests_dir, results_dir):
            if result['complete']:
                return code, Flagging.PROVED
        
        # Check negation proofs
        for (code, _), result in zip(prove_requests_neg, results_neg):
            if result['complete']:
                return code, Flagging.REFUTED
        
        # If neither direct nor negation proofs succeeded

        return formalized_problem, Flagging.UNKNOWN


    def prove(self, dag: DAGModel, formalizer_header: str = HEADER) -> DAGModel:
        taskqueue = []
        from datetime import datetime
        start_time = datetime.now()
        for node in dag.nodes:
            if node.flag == Flagging.DECLARATIVE or node.flag == Flagging.AF_FAIL:
                continue

            self.initialize_dialog()
            
            tc_request_id = self.get_tc_async(
                formalized_problem=node.formalized_content,
                header=formalizer_header
            )
            
            prove_requests = self.prove_single_async(
                node.perturbed_content, 
                node.formalized_content)
            
            taskqueue.append( (node, tc_request_id, prove_requests) )
        
        print("Time to submit all tasks:", (datetime.now() - start_time).total_seconds())
        # 1470.50615 sec
        start_time = datetime.now()
        for node, tc_request_id, prove_requests in taskqueue:
            tc_result = self.get_tc_result(tc_request_id)
            if not tc_result:
                node.flag = Flagging.TC_FAIL
                continue

            if prove_requests is None:
                node.flag = Flagging.AF_FAIL
                continue

            node.formalized_content, node.flag = self.prove_single_result(prove_requests, node.formalized_content)
        print("Time to process all results:", (datetime.now() - start_time).total_seconds())
        breakpoint()
        return dag
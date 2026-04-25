import argparse
import subprocess
import json
import os
import re
import random
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher
from datetime import datetime
import backoff
import openai

from veriform.data.loaders import ProcessBenchLoader
from veriform.proving.lean_server.prover.lean.verifier import Lean4ServerScheduler
from veriform.proving.theorem_extractor import TheoremExtractor


_TOKEN_RE = re.compile(
    r"""
    (?P<WS>\s+)                                             # whitespace
  | (?P<NUM>(?<![A-Za-z\d])[-]?\d+(?:\.\d+)?(?![A-Za-z]))   # numbers
  | (?P<OP>==|!=|<=|>=|[+\-*/^<>=])                         # operators
  | (?P<WORD>[A-Za-z]+)                                     # words
  | (?P<OTHER>.)                                            # everything else (punct etc.)
    """,
    re.VERBOSE,
)

def _tokenise(s: str) -> List[Tuple[str, str]]:
    """Return list of (type, text) tokens."""
    out: List[Tuple[str, str]] = []
    for m in _TOKEN_RE.finditer(s):
        typ = m.lastgroup or "OTHER"
        out.append((typ, m.group(typ)))
    return out

def _is_candidate(tok_type: str, tok_text: str) -> bool:
    """Tokens we might plausibly want in the replacement dict."""
    if tok_type in ("NUM", "OP"):
        return True
    # Logical negation uses the literal word "not" (case-insensitive)
    if tok_type == "WORD" and tok_text.lower() == "not":
        return True
    return False

# --- Alignment + extraction ---------------------------------------------------

def _pair_candidates(
    old: List[Tuple[str, str]],
    new: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """
    Pair candidate tokens from a replaced region.
    If lengths match: zip in order.
    Otherwise: greedy pairing by type in order.
    """
    old_c = [(t, x) for (t, x) in old if _is_candidate(t, x)]
    new_c = [(t, x) for (t, x) in new if _is_candidate(t, x)]
    if not old_c or not new_c:
        return []

    if len(old_c) == len(new_c):
        return [(o[1], n[1]) for o, n in zip(old_c, new_c) if o[1] != n[1]]

    # Greedy: for each old candidate, find next new candidate with same type
    pairs: List[Tuple[str, str]] = []
    j = 0
    for ot, ox in old_c:
        while j < len(new_c) and new_c[j][0] != ot:
            j += 1
        if j >= len(new_c):
            break
        _, nx = new_c[j]
        j += 1
        if ox != nx:
            pairs.append((ox, nx))
    return pairs

def get_replacement_dict(content: str, perturbed_content: str) -> Dict[str, str]:
    """
    Attempt to recover atomic token replacements performed by StandardPerturber.

    Returns a dict mapping original token text -> perturbed token text,
    e.g. {'2': '3', '+': '-'}.

    Heuristics:
    - tokenise both strings
    - align token texts with SequenceMatcher
    - inspect replace regions (and small insert/delete regions for 'not')
    - extract candidate substitutions for NUM/OP/'not'
    - keep only consistent mappings (skip conflicts)
    """
    a = _tokenise(content)
    b = _tokenise(perturbed_content)

    # Align on token *texts* (types help later; alignment uses just text).
    a_text = [x for (_t, x) in a]
    b_text = [x for (_t, x) in b]
    sm = SequenceMatcher(a=a_text, b=b_text, autojunk=False)

    mapping: Dict[str, str] = {}

    def _add_pair(src: str, dst: str) -> None:
        # Keep only consistent mappings; skip conflicts.
        if src in mapping and mapping[src] != dst:
            return
        mapping[src] = dst

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        old_seg = a[i1:i2]
        new_seg = b[j1:j2]

        if tag == "replace":
            for src, dst in _pair_candidates(old_seg, new_seg):
                _add_pair(src, dst)

        elif tag in ("insert", "delete"):
            # Handle logical negation insert/remove of the word "not"
            # (or very small token insertions/deletions).
            seg = new_seg if tag == "insert" else old_seg
            for t, x in seg:
                if t == "WORD" and x.lower() == "not":
                    # If "not" was inserted, we can’t map from a source token cleanly.
                    # But if it was deleted, we *can* treat it like 'not' -> ''.
                    if tag == "delete":
                        _add_pair(x, "")
                    # If inserted, we ignore (no stable dict key to represent “nothing”).
                    break

    # Optional: prune “obviously not a perturbation” mappings.
    # For instance, mapping long words is unlikely for this perturber.
    mapping = {k: v for k, v in mapping.items() if _is_candidate("NUM", k) or _is_candidate("OP", k) or k.lower() == "not"}

    return mapping

def apply_replacement_dict(content: str, repl: Dict[str, str]) -> str:
    """
    Apply token replacements assuming the perturbator modifies
    only the last (or one of the last) occurrences.

    Each key in `repl` is replaced exactly once, starting from
    the rightmost occurrence.
    """
    out = content

    # Sort by length so that '==' beats '=' and multi-char tokens
    # are handled before single-char ones
    for src, dst in sorted(repl.items(), key=lambda kv: -len(kv[0])):
        if not src:
            continue

        # Use regex to find all literal occurrences
        pattern = re.escape(src)
        matches = list(re.finditer(pattern, out))
        if not matches:
            continue

        # Pick the last occurrence
        m = matches[-1]
        start, end = m.span()

        out = out[:start] + dst + out[end:]

    return out


# content = "Therefore, we can conclude that 1 + 1 = 2."
# pert = "Therefore, we can conclude that 1 - 1 = 3."
# print(get_replacement_dict(content, pert))
# {'+': '-', '2': '3'}



def dominik_correction(
        nl_unperturbed: str,
        nl_perturbed: str,
        fl_unperturbed: str):
    """
    Orchestrates the perturbation injection and verification.
    """
    replacement_dict = get_replacement_dict(
        content=nl_unperturbed,
        perturbed_content=nl_perturbed
    )
    return apply_replacement_dict(
        content=fl_unperturbed,
        repl=replacement_dict
    ), replacement_dict


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


LEAN_PATTERN = re.compile(r"```lean4?(.*?)```", re.DOTALL | re.IGNORECASE)
def parse_lean_code(response_text):
    matches = LEAN_PATTERN.findall(response_text)
    if matches:
        # Return the last one, stripped of leading/trailing whitespace
        return matches[-1].strip()
    else:
        raise ValueError(f"No Lean 4 theorem found in response:\n{response_text[:200]}...")

THEOREM_PATTERN = re.compile(r"(^theorem[\s\S]*?sorry)", re.MULTILINE | re.IGNORECASE)
def parse_lean_theorem(response_text: str) -> str:
    # Find all valid theorem blocks ending in sorry
    matches = THEOREM_PATTERN.findall(response_text)
    if matches:
        # Return the last one, stripped of leading/trailing whitespace
        return matches[-1].strip()
    else:
        raise ValueError(f"No Lean 4 theorem found in response:\n{response_text[:200]}...")

scheduler = Lean4ServerScheduler(max_concurrent_requests=5)
theorem_extractor = TheoremExtractor()

@backoff.on_exception(
        backoff.expo, 
        (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError),
        max_tries=5,
        jitter=backoff.full_jitter
    )
def _generate_batch(prompts: List[str]):
    # No 'await' here. This line blocks until completion.
    return client.completions.create(
        model="deepseek-ai/DeepSeek-Prover-V2-7B",
        prompt=prompts,
        temperature=0.6,
        top_p=0.95,
        max_tokens=8192,
        n=2
    )


def is_provable(taskqueue: List[Tuple[int, str]]) -> List[bool]:
    def _normalize(s: Optional[str]) -> str:
            if not s: return ""
            return " ".join(s.split())
    prompts = [
        LEAN_WRAPPER_TEMPLATE.format(
            lean_code=LEAN_TEMPLATE.format(
                lean_code=task[1], 
                statement="")
        )
        for task in taskqueue
    ]

    response = _generate_batch(prompts)
    choices = []
    correction = []
    indices = []
    for i, candidate in enumerate(response.choices):
        correction.append(False)
        try:
            lean_code = parse_lean_code(candidate.text.strip())
            _, param, body = theorem_extractor.get_last_theorem(lean_code)
            _, param_tgt, body_tgt = theorem_extractor.get_last_theorem(taskqueue[i//2][1])
            if _normalize(param) != _normalize(param_tgt) or _normalize(body) != _normalize(body_tgt):
                raise ValueError("param or body does not match")
            choices.append(lean_code)
            indices.append(i)
        except ValueError as e:
            choices.append(None)


    to_submit = [
        LEAN_TEMPLATE.format(
                lean_code=choices[i], 
                statement="")
                for i in indices]
    request_ids = scheduler.submit_all_request(to_submit)
    all_results = scheduler.get_all_request_outputs(request_ids)
    
    for i, result in zip(indices, all_results):
        correction[i] = correction[i] or result['complete']
    
    return [correction[2 * i] or correction[2 * i + 1] for i in range(len(correction)//2)]

def main():
    processbench = ProcessBenchLoader(file_path='./data/processed/dags.json', num_samples=1179)
    chains = processbench.load()

    random.seed(datetime.now().timestamp())
    random.shuffle(chains)

    for chain in chains:
        save_path = f'data/correction/cots/{chain.chain_id}.json'
        if os.path.exists(save_path):
            continue

        nodes = []
        ret_nodes = []
        for formalizer in ['herald', 'stepfun', 'goedel', 'kimina']:
            json_file = f'data/regex_perturbed/proved/{formalizer}/0/{chain.chain_id}.json'
            perturbed_json_file = f'data/regex_perturbed/proved/{formalizer}/100/{chain.chain_id}.json'
            
            with open(perturbed_json_file, 'r') as f:
                node_list_perturbed = json.load(f)['nodes']
            
            with open(json_file, 'r') as f:
                node_list_unperturbed = json.load(f)['nodes']
            
            for node_u, node_p in zip(node_list_unperturbed, node_list_perturbed):
                node_u['perturbed_content'] = node_p['perturbed_content']

            nodes.append(node_list_unperturbed)
        
        taskqueue = []
        for i in range(len(nodes[0])):
            if nodes[0][i]['flag'] == "Declarative":
                ret_nodes.append(nodes[0][i])
                continue

            node_cands = [j for j in range(4) if nodes[j][i]["flag"] != "AF-Fail"]
            if len(node_cands) == 0:
                ret_nodes.append(nodes[0][i])
                continue
        
            j = node_cands[0]
            node = nodes[j][i]

            node["formalizer_content_by_sorry"] = parse_lean_theorem(node['formalizer_output'])


            fl_perturbed, replacement_dict = dominik_correction(
                node['content'],
                node['perturbed_content'],
                node["formalizer_content_by_sorry"]
            )
            node['replacement_dict'] = replacement_dict
            node['perturbed_formalized_content'] = fl_perturbed
            ret_nodes.append(node)
            taskqueue.append((i, node['perturbed_formalized_content']))
        
        results = is_provable(taskqueue)

        for (i, _), result in zip(taskqueue, results):
            ret_nodes[i]['corrected'] = result

        with open(save_path, 'w') as f:
            json.dump(ret_nodes, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run v2 experiments with perturbed formalization pipeline.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the prover server.")
    args = parser.parse_args()
    from openai import OpenAI 
    client = OpenAI(
            api_key="EMPTY", 
            base_url=f"http://localhost:{args.port}/v1",
            timeout=3600, # 1 hour timeout
        )
    main()

scheduler.close()
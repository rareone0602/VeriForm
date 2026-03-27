import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import random

from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# --- Internal Imports ---
from veriform.autoformalization_v2.dag import DAGModel, Flagging
from veriform.data_collection.dataset_loaders import ProcessBenchLoader
from veriform.autoformalization_v2.theorem_extractor import TheoremExtractor
from veriform.autoformalization_v2.deepseek.prover.lean.verifier import Lean4ServerScheduler



def formatting(lean_code: str) -> str:
    HEADER = """
    import Mathlib
    import Aesop

    set_option maxHeartbeats 0

    open BigOperators Real Nat Topology Rat
    """
    return HEADER + "\n\n" + lean_code

scheduler = Lean4ServerScheduler(max_concurrent_requests=128)
theorem_extractor = TheoremExtractor()

def get_refute_theorem(formal_problem: str) -> str:
    name, param, body = theorem_extractor.get_last_theorem(formal_problem)
    
    if name is None or body is None:
        raise ValueError("Cannot extract theorem name/body from formal problem.")
    
    # Clean up params: handle None and whitespace
    param_str = param.strip() if param else ""
    
    refute_name = f"not_{name}"
        
    if param_str:
        # We have parameters, so we construct: ¬ (∀ (x : T), body)
        # Note: In Lean 4, '∀ (x : T) (y : U), P' is valid syntax.
        refute_body = f"¬ (∀ {param_str}, {body})"
    else:
        # No parameters? Then Full Negation is identical to Strong Negation.
        refute_body = f"¬ ({body})"

    return f"theorem {refute_name} : {refute_body} := by sorry"
        
        



processbench = ProcessBenchLoader(file_path='./data/processed/dags.json', num_samples=1179)
chains = processbench.load()


node_num = 0
af_fail_num = 0
def go_node(node):
    global node_num, af_fail_num
    node_num += 1
    datum = {
        "nl": node.perturbed_content,
        "fl_dir": node.formalized_content,
        "flag_dir": Flagging.UNKNOWN.value,
        "fl_neg": None,
        "flag_neg": Flagging.UNKNOWN.value,
    }
    
    if node.flag == Flagging.DECLARATIVE:
        datum['flag_dir'] = datum['flag_neg'] = Flagging.DECLARATIVE.value
        return datum
    
    if node.flag == Flagging.AF_FAIL:
        af_fail_num += 1
        datum["fl_neg"] = datum["fl_dir"] = "-- Failed to formalize"
        datum["flag_neg"] = datum["flag_dir"] = Flagging.AF_FAIL.value
        return datum
    
    try:
        if node.formalized_content is None:
            breakpoint()
        fl_neg = get_refute_theorem(node.formalized_content)
        datum["fl_neg"] = fl_neg
        return datum
    
    except ValueError as e:
        # For debugging
        print(f"Failed to get refute theorem for \n```\n{node.formalized_content}\n```")
        af_fail_num += 1
        datum["fl_neg"] = datum["fl_dir"] = "-- Failed to formalize"
        datum["flag_neg"] = datum["flag_dir"] = Flagging.AF_FAIL.value
        return datum

json_file = {
    "goedel": {},
    "kimina": {},
    "herald": {},
    "stepfun": {}
}

taskqueue = []
for formalizer_class_name in ["goedel", "kimina", "herald", "stepfun"]:
    for chain in tqdm(chains): # For debug
        datum = []
        
        perturbed_pkl_path = f"data/regrex_perturbed/{formalizer_class_name}/100/{chain.chain_id}.pkl"
        unperturbed_pkl_path = f"data/regrex_perturbed/{formalizer_class_name}/0/{chain.chain_id}.pkl"

        with open(perturbed_pkl_path, "rb") as f:
            perturbed_dag_model = pickle.load(f)

        for i, node in enumerate(perturbed_dag_model):
            assert node.node_id == i
            node_data = {
                "unperturbed": {},
                "perturbed": go_node(node)
            }
            datum.append(node_data)

        with open(unperturbed_pkl_path, "rb") as f:
            unperturbed_dag_model = pickle.load(f)

        for i, node in enumerate(unperturbed_dag_model):
            assert node.node_id == i
            datum[i]["unperturbed"] = go_node(node)

            if datum[i]["unperturbed"] is not None:
                if datum[i]["unperturbed"]["flag_dir"] not in ['Declarative', 'AF-FAIL', 'TC-FAIL']:
                    taskqueue.append((formalizer_class_name, chain.chain_id, node.node_id, "unperturbed", "fl_dir", datum[i]["unperturbed"]["fl_dir"])) 
                if datum[i]["unperturbed"]["flag_neg"] not in ['Declarative', 'AF-FAIL', 'TC-FAIL']:
                    taskqueue.append((formalizer_class_name, chain.chain_id, node.node_id, "unperturbed", "fl_neg", datum[i]["unperturbed"]["fl_neg"]))
            if datum[i]['perturbed'] is not None:
                if datum[i]["perturbed"]['flag_dir'] not in ['Declarative', 'AF-FAIL', 'TC-FAIL']:
                    taskqueue.append((formalizer_class_name, chain.chain_id, node.node_id,   "perturbed", "fl_dir", datum[i]["perturbed"]["fl_dir"]))
                if datum[i]["perturbed"]['flag_neg'] not in ['Declarative', 'AF-FAIL', 'TC-FAIL']:
                    taskqueue.append((formalizer_class_name, chain.chain_id, node.node_id,   "perturbed", "fl_neg", datum[i]["perturbed"]["fl_neg"])) 

        json_file[formalizer_class_name][chain.chain_id] = datum


request_ids = scheduler.submit_all_request([
    dict(code=formatting(lean_code), ast=False, tactics=False)
    for _, _, _, _, _, lean_code in taskqueue
])

results = scheduler.get_all_request_outputs(request_ids)

for (formalizer_class_name, chain_id, node_id, perturbed_or_not, fl_or_neg, _), result in zip(taskqueue, results):
    verdict = result['pass']
    if not verdict:
        json_file[formalizer_class_name][chain_id][node_id][perturbed_or_not][f"flag_{fl_or_neg}"] = Flagging.TC_FAIL.value



with open("data/outputs/all_formalized_results.json", "w") as f:
    json.dump(json_file, f, indent=2)

print(af_fail_num, "/", node_num, "AF-Fail rate:", af_fail_num / node_num)
scheduler.close()
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import random
import re

from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# --- Internal Imports ---
from veriform.preprocessing.dag import DAGModel, Flagging
from veriform.data.loaders import ProcessBenchLoader
from veriform.proving.theorem_extractor import TheoremExtractor
from veriform.proving.lean_server.prover.lean.verifier import Lean4ServerScheduler


scheduler = Lean4ServerScheduler(max_concurrent_requests=64)


file = 'gemini_perturbed'

processbench = ProcessBenchLoader(file_path='./data/processed/dags.json', num_samples=1179)
chains = processbench.load()


import re

LEAN_PATTERN = re.compile(
    r"```(?:lean4?)?[ \t]*(?:\n)?(.*?)(?:\n)?```", 
    re.DOTALL | re.IGNORECASE
)
def parse_lean_code(response_text):
    matches = LEAN_PATTERN.findall(response_text)
    if matches:
        # Return the last one, stripped of leading/trailing whitespace
        return matches[-1].strip()
    else:
        raise ValueError(f"No Lean 4 theorem found in response:\n{response_text}...")


with open(f"data/raw/{file}.json", "r") as f:
    data = json.load(f)


taskqueue = []
lean_codes = []

non_declarative_nodes = 0
parsed_fail = 0
valid_attempt = 0

for chain in tqdm(chains): # For debug
    for i, node in enumerate(data[chain.chain_id]):
        if node['statement_type'] == "declarative":
            continue
        non_declarative_nodes += 1
        try:
            lean_proof = parse_lean_code(node['lean_proof'])
        except ValueError as e:
            node['complete'] = False
            data[chain.chain_id][i]['lean_result'] = {}
            parsed_fail += 1
            continue
        
        valid_attempt += 1
        lean_codes.append(lean_proof)
        taskqueue.append((chain.chain_id, i))

print(non_declarative_nodes, parsed_fail, valid_attempt)
breakpoint()

request_ids = scheduler.submit_all_request([
    dict(code=lean_code, ast=False, tactics=False)
    for lean_code in lean_codes
])
results = scheduler.get_all_request_outputs(request_ids)

for (chain_id, i), result in zip(taskqueue, results):
    data[chain_id][i]['complete'] = result['complete']
    data[chain_id][i]['lean_result'] = result
    

with open(f"data/tmp/{file}.json", "w") as f:
    json.dump(data, f, indent=4)

scheduler.close()
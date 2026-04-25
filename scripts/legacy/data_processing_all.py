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


processbench = ProcessBenchLoader(file_path='./data/processed/dags.json', num_samples=1179)
chains = processbench.load()


node_num = 0
af_fail_num = 0

json_data = {}


for formalizer_class_name in ["goedel", "kimina", "herald", "stepfun"]:
    json_data[formalizer_class_name] = {}
    for chain in tqdm(chains): # For debug
        json_data[formalizer_class_name][chain.chain_id] = []

        efforts = ['low', 'medium', 'high']
        
        json_paths = [
            f"data/llm_perturbed_{effort}/proved/{formalizer_class_name}/100/{chain.chain_id}.json"
            for effort in efforts
        ]

        with open(json_paths[0], "r") as f:
            data = json.load(f)
            for j, node in enumerate(data["nodes"]):
                json_data[formalizer_class_name][chain.chain_id].append({})

        
        for i in range(3):
            with open(json_paths[i], "r") as f:
                effort = efforts[i]
                data = json.load(f)
                assert len(data["nodes"]) == len(json_data[formalizer_class_name][chain.chain_id])
                for j, node in enumerate(data["nodes"]):
                    json_data[formalizer_class_name][chain.chain_id][j][f"llm_perturbed_{effort}"] = node
        
with open("data/tmp/all_perturbed.json", "w") as f:
    json.dump(json_data, f, indent=2)


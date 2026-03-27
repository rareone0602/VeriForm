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

from veriform.autoformalization_v2.dag import DAGModel
from veriform.data_collection.dataset_loaders import ProcessBenchLoader
from veriform.autoformalization_v2.perturber import DeepSeekPrePerturber
import pickle
from pprint import pprint

processbench = ProcessBenchLoader(file_path='./data/processed/dags.json', num_samples=1137)
chains = processbench.load()

input_dir = 'data/regex_perturbed/proved'
save_dir = 'experiments/assets'


xlables = ["AF-Fail", "TC-Fail", "Refuted", "Unknown", "Proved"]
ylables = ["OmniMATH", "OlympiadBench", "MATH", "GSM8K"]

marks = ['unperturbed', 'perturbed']

for formalizer in ['goedel', 'kimina', 'herald', 'stepfun']:
    for i in [0, 1]:
        p = i * 100
        mark = marks[i]
        explore_dir = os.path.join(input_dir, formalizer, str(p))
        
        total_thm = 0
        cnts = {
            prob_type.lower() : {'sum': 0} for prob_type in ylables
        }
        for filename in os.listdir(explore_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(explore_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)

                problem_type = data['chain_id'].split('-')[0].lower()

                for node in data['nodes']:
                    flag = node['flag']
                    if flag != "Declarative":
                        total_thm += 1
                        cnts[problem_type][flag] = cnts[problem_type].get(flag, 0) + 1
                        cnts[problem_type]['sum'] += 1
        print(formalizer, p, total_thm)
        
        with open(f"experiments/assets/{formalizer}_{mark}.dat", 'w') as f:
            f.write("x y val percentage\n")
            for y in range(4):
                problem_type = ylables[y].lower()
                for x in range(5):
                    flag = xlables[x]
                    num = cnts[problem_type].get(flag, 0)
                    total = cnts[problem_type]['sum']
                    f.write(f"{x} {y} {num} {int(0.5 + num / total * 100)}\n")
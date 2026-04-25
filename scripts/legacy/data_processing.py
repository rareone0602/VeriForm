
import json
from tqdm import tqdm

# --- Internal Imports ---
from veriform.data_collection.dataset_loaders import ProcessBenchLoader


processbench = ProcessBenchLoader(file_path='./data/processed/dags.json', num_samples=1179)
chains = processbench.load()


node_num = 0
af_fail_num = 0

json_data = {}


for formalizer_class_name in ["goedel", "kimina", "herald", "stepfun"]:
    json_data[formalizer_class_name] = {}
    for chain in tqdm(chains): # For debug
        json_data[formalizer_class_name][chain.chain_id] = []

        unperturbed_json_path = f"data/regex_perturbed/proved/{formalizer_class_name}/0/{chain.chain_id}.json"
        perturbed_json_path = f"data/regex_perturbed/proved/{formalizer_class_name}/100/{chain.chain_id}.json"
        

        with open(unperturbed_json_path, "r") as f:
            data = json.load(f)
            for j, node in enumerate(data["nodes"]):
                json_data[formalizer_class_name][chain.chain_id].append({})

        
        for i in range(2):
            json_path = [unperturbed_json_path, perturbed_json_path][i]
            perturbation = ["unperturbed", "perturbed"][i]
            with open(json_path, "r") as f:
                data = json.load(f)
                assert len(data["nodes"]) == len(json_data[formalizer_class_name][chain.chain_id])
                for j, node in enumerate(data["nodes"]):
                    json_data[formalizer_class_name][chain.chain_id][j][perturbation] = node
        
with open("data/tmp/regex_perturbed.json", "w") as f:
    json.dump(json_data, f, indent=2)


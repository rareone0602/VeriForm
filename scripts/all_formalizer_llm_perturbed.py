import argparse
import pickle
import os
import random

from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# --- Internal Imports ---
from veriform.data_collection.dataset_loaders import ProcessBenchLoader
from veriform.autoformalization_v2.dag import DAGModel
from veriform.autoformalization_v2.perturber import DeepSeekPrePerturber
from veriform.autoformalization_v2.formalizer import (
    StepfunFormalizer, 
    KiminaFormalizer, 
    GoedelFormalizer, 
    HeraldFormalizer
)

# --- Configuration & Constants ---
# Centralise these lists so you can easily add new datasets or formalizers later.
DATASET_ORDER = ["GSM8K", "MATH", "OlympiadBench", "OmniMATH"]
FLAG_COLUMNS = ['AF-FAIL', 'TC-FAIL', 'REFUTED', 'UNKNOWN', 'PROVED', "TOTAL"]

FORMALIZER_MAP = {
    "stepfun": StepfunFormalizer,
    "kimina": KiminaFormalizer,
    "goedel": GoedelFormalizer,
    "herald": HeraldFormalizer,
}

# Mapping dataset names to row indices for the heatmap
DATASET_TO_ID = {name.lower(): i for i, name in enumerate(DATASET_ORDER)}

def v2_exp(
        p, 
        num_samples, 
        output_dir, 
        prover_batch_size, 
        negation_type, 
        sampling, 
        formalizer_class_name,
        effort):
    lean_dir = output_dir / "lean_programs"
    lean_dir.mkdir(parents=True, exist_ok=True)

    # Initialize data structure
    data = {
        "config": {
            "p": p,
            "num_samples": num_samples,
            "formalizer": formalizer_class_name,
            "negation_type": negation_type,
            "sampling": sampling,
        },
        "results": [],
        "statistics": {
            key: 0 for key in [
                "total", "af_failed_count", "tc_failed_count", 
                "refuted_count", "proved_count", "declarative_count", 
                "unknown_count", "tp_count", "fp_count", "tn_count", "fn_count"
            ]
        }
    }
    # Initialize metrics
    data["statistics"].update({"f1_score": 0.0, "accuracy": 0.0, "recall": 0.0, "precision": 0.0})

    # Setup Pipeline
    processbench = ProcessBenchLoader(file_path='./data/processed/dags.json', num_samples=num_samples)
    chains = processbench.load()
    
    perturber = DeepSeekPrePerturber(
        p=p,
        effort=effort
    )
    
    if formalizer_class_name not in FORMALIZER_MAP:
        raise ValueError(f"Unknown formalizer: {formalizer_class_name}")
    
    formalizer = FORMALIZER_MAP[formalizer_class_name](sampling=sampling)

    # Main Loop
    random.seed(datetime.now().timestamp())
    random.shuffle(chains)

    for chain in tqdm(chains):
        save_path = f"data/llm_perturbed_{effort}/formalized/{formalizer_class_name}/{int(p * 100 + 0.5)}/{chain.chain_id}.pkl"
        
        if os.path.exists(save_path):
            print(f"Skipping existing file: {save_path}")
            continue

        dag = DAGModel(chain)
        dag = perturber.perturb(dag)
        dag = formalizer.formalize(dag)

        # serialize dag to pkl
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(dag, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run v2 experiments with perturbed formalization pipeline.")
    parser.add_argument("--p", type=float, default=0.5, help="Perturbation probability.")
    parser.add_argument("--num_samples", type=int, default=1179, help="Number of samples to process.")
    parser.add_argument("--prover_batch_size", type=int, default=2, help="Batch size for the prover.")
    parser.add_argument("--formalizer", type=str, choices=list(FORMALIZER_MAP.keys()), required=True, help="Choice of formalizer.")
    parser.add_argument("--negation", type=str, default="full", choices=["strong", "full"], help="Choice of negation.")
    parser.add_argument("--sampling",  type=str, default="deterministic", choices=["recommended", "deterministic"],  help="Choice of sampling strategy.")
    parser.add_argument("--effort", type=str, default="medium", choices=["low", "medium", "high"])
    args = parser.parse_args()

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f'experiments/outputs/{date_str}_{args.formalizer}_p{int(args.p * 100):03d}_n{args.num_samples}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    v2_exp(
        p=args.p, 
        num_samples=args.num_samples, 
        output_dir=output_dir,
        prover_batch_size=args.prover_batch_size,
        negation_type=args.negation,
        sampling=args.sampling,
        formalizer_class_name=args.formalizer,
        effort=args.effort
    )

# To run this script, use a command like:
# python scripts/all_formalizer.py --p 1 --formalizer stepfun
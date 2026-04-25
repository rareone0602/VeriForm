import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# --- Internal Imports ---
from veriform.data_collection.dataset_loaders import ProcessBenchLoader
from veriform.autoformalization_v2.dag import DAGModel
from veriform.autoformalization_v2.perturber import StandardPerturber
from veriform.autoformalization_v2.formalizer import (
    StepfunFormalizer, 
    KiminaFormalizer, 
    GoedelFormalizer, 
    HeraldFormalizer
)
from veriform.autoformalization_v2.prover import DeepSeekProver
from veriform.autoformalization_v2.pipeline import StandardPipeline

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

def draw_heatmap(p, formalizer_name, heatmap_data, output_path):
    """
    Generates and saves the theorem status distribution heatmap.
    """
    # 1. Prepare Data
    # The data is already structured by DATASET_ORDER in the main loop,
    # so we can use it directly.
    raw_data = np.array(heatmap_data["data"])
    rows = heatmap_data["rows"]
    cols = heatmap_data["columns"][:-1] # Exclude TOTAL for plotting

    # Separate categories from the total column
    category_data = raw_data[:, :-1] 
    
    # Calculate percentages safe from division by zero
    row_sums = category_data.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums == 0, 1, row_sums)
    pct_data = (category_data / safe_sums) * 100

    # 2. Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pct_data, cmap="Blues", vmin=0, vmax=100)

    # Axis configuration
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols)
    ax.set_yticklabels(rows)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 3. Annotation
    for i in range(len(rows)):
        for j in range(len(cols)):
            pct_val = pct_data[i, j]
            raw_val = category_data[i, j]
            
            text_str = f"{pct_val:.1f}%\n({int(raw_val)})"
            # distinct contrast for readability
            text_color = "white" if pct_val > 50 else "black"
            
            ax.text(j, i, text_str, 
                    ha="center", va="center", 
                    color=text_color, 
                    fontsize=9, fontweight='medium')

    ax.set_title(f"Theorem Status Distribution (p={p}, formalizer={formalizer_name})")
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Percentage (%)", rotation=-90, va="bottom")

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()

def update_statistics(stats, node, is_perturbed):
    """Helper to update the statistics dictionary based on node flags."""
    # Assuming node.flag is an Enum member, we access the Enum class via type(node.flag)
    # or rely on the sibling access if the class isn't imported.
    Flag = type(node.flag) 
    
    if node.flag == Flag.DECLARATIVE:
        stats["declarative_count"] += 1
        return # Declarative nodes are excluded from TP/FP/TN/FN metrics

    stats["total"] += 1
    
    # Update Basic Counts
    if node.flag == Flag.AF_FAIL:
        stats["af_failed_count"] += 1
    elif node.flag == Flag.TC_FAIL:
        stats["tc_failed_count"] += 1
    elif node.flag == Flag.REFUTED:
        stats["refuted_count"] += 1
    elif node.flag == Flag.UNKNOWN:
        stats["unknown_count"] += 1
    elif node.flag == Flag.PROVED:
        stats["proved_count"] += 1
    
    # Update Confusion Matrix Counts
    # PROVED is the 'Negative' class (no bug/perturbation found)
    # FAIL/REFUTED/UNKNOWN are the 'Positive' class (issue found)
    is_positive_flag = (node.flag != Flag.PROVED)

    if is_perturbed:
        if is_positive_flag:
            stats["tp_count"] += 1
        else:
            stats["fn_count"] += 1
    else:
        if is_positive_flag:
            stats["fp_count"] += 1
        else:
            stats["tn_count"] += 1

def calculate_metrics(stats):
    """Recalculates precision, recall, and F1 based on current counts."""
    tp = stats["tp_count"]
    fp = stats["fp_count"]
    fn = stats["fn_count"]
    tn = stats["tn_count"]
    total = stats["total"]

    stats["accuracy"] = (tp + tn) / max(1, total)
    stats["precision"] = tp / max(1, tp + fp)
    stats["recall"] = tp / max(1, tp + fn)
    stats["f1_score"] = 2 * (stats["precision"] * stats["recall"]) / max(1e-8, (stats["precision"] + stats["recall"]))

def v2_exp(p, num_samples, output_dir, prover_batch_size, negation_type, sampling, formalizer_class_name):
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

    # Initialize Heatmap Data (Zeros)
    heatmap_data = {
        "columns": FLAG_COLUMNS,
        "rows": DATASET_ORDER,
        "data": np.zeros((len(DATASET_ORDER), len(FLAG_COLUMNS))).tolist()
    }

    # Setup Pipeline
    processbench = ProcessBenchLoader(file_path='./data/processed/dags.json', num_samples=num_samples)
    chains = processbench.load()
    
    perturber = StandardPerturber(
        p=p, 
        operator_swap=True, 
        value_change=True, 
        logical_negation=True
    )
    
    if formalizer_class_name not in FORMALIZER_MAP:
        raise ValueError(f"Unknown formalizer: {formalizer_class_name}")
    
    formalizer = FORMALIZER_MAP[formalizer_class_name](sampling=sampling)
    prover = DeepSeekProver(batch_size=prover_batch_size, negation_type=negation_type)
    pipeline = StandardPipeline(perturber, formalizer, prover)

    # Main Loop
    for chain in tqdm(chains):
        start_time = datetime.now()
        dag = pipeline(chain)
        lean_program = dag.lean()
        duration = (datetime.now() - start_time).total_seconds()

        # Save individual result
        data['results'].append({
            "duration": duration,
            "chain_id": chain.chain_id,
            "nodes": [{
                "node_id": n.node_id,
                "flag": n.flag.value,
                "is_perturbed": n.is_perturbed,
                "content": n.content,
                "formalizer_output": n.formalizer_output,
                "perturbed_content": n.perturbed_content,
                "formalized_content": n.formalized_content
            } for n in dag.nodes]
        })

        # Save Lean file
        with open(lean_dir / f"{chain.chain_id}.lean", "w") as f:
            f.write(lean_program)
        
        # Update Statistics & Heatmap
        difficulty_key = chain.chain_id.split('-')[0].lower()
        # Default to 0 (GSM8K) if unknown difficulty found to prevent crash
        row_idx = DATASET_TO_ID[difficulty_key] # Make it throw an error if not found

        for node in dag.nodes:
            update_statistics(data["statistics"], node, node.is_perturbed)
            
            # Update heatmap counts if not declarative
            Flag = type(node.flag)
            if node.flag != Flag.DECLARATIVE:
                col_idx = -1
                if node.flag == Flag.AF_FAIL: col_idx = 0
                elif node.flag == Flag.TC_FAIL: col_idx = 1
                elif node.flag == Flag.REFUTED: col_idx = 2
                elif node.flag == Flag.UNKNOWN: col_idx = 3
                elif node.flag == Flag.PROVED:  col_idx = 4
                
                if col_idx != -1:
                    heatmap_data["data"][row_idx][col_idx] += 1
                    heatmap_data["data"][row_idx][5] += 1 # Total column

        # Recalculate derived metrics
        calculate_metrics(data["statistics"])

        # Periodic Save (Persist results during the run)
        with open(output_dir / f"exp_results_{formalizer_class_name}.json", "w") as f:
            json.dump(data, f, indent=4)
            
        draw_heatmap(
            p, 
            formalizer_class_name,
            heatmap_data, 
            output_dir / f"heatmap_{formalizer_class_name}.png"
        )

    del pipeline
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run v2 experiments with perturbed formalization pipeline.")
    parser.add_argument("--p", type=float, default=0.5, help="Perturbation probability.")
    parser.add_argument("--num_samples", type=int, default=1179, help="Number of samples to process.")
    parser.add_argument("--prover_batch_size", type=int, default=2, help="Batch size for the prover.")
    parser.add_argument("--formalizer", type=str, choices=list(FORMALIZER_MAP.keys()), required=True, help="Choice of formalizer.")
    parser.add_argument("--negation", type=str, default="full", choices=["strong", "full"], help="Choice of negation.")
    parser.add_argument("--sampling",  type=str, default="deterministic", choices=["recommended", "deterministic"],  help="Choice of sampling strategy.")
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
        formalizer_class_name=args.formalizer
    )

# To run this script, use a command like:
# python scripts/v2_exp.py --p 0.5 --num_samples 1000
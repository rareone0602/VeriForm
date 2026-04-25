"""Single config-driven entry point for VeriForm experiments.

Replaces the legacy fork of v2_exp.py / all_proof.py / all_formalizer.py.
Pass ``--config configs/foo.yaml`` and optional ``--set key=val`` overrides.
The pipeline stages (dataset, perturbation, formalization, proving,
evaluation) are wired from the registries in src/veriform/.

Examples:
    # End-to-end with default regex perturbation, p=0.5, Stepfun formalizer:
    python scripts/run_benchmark.py --config configs/default.yaml

    # CLI overrides:
    python scripts/run_benchmark.py --config configs/default.yaml \\
        --set formalization.type=goedel perturbation.p=1.0

    # Smoke test (skip GPU prover):
    python scripts/run_benchmark.py --config configs/default.yaml \\
        --set proving.enabled=false dataset.num_samples=5
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from veriform.config import RunConfig
from veriform.data.loaders import ProcessBenchLoader
from veriform.evaluation import (
    calculate_metrics,
    draw_heatmap,
    init_heatmap_data,
    init_statistics,
    update_heatmap_data,
    update_statistics,
)
from veriform.formalization import FORMALIZER_REGISTRY
from veriform.perturbation import PERTURBER_REGISTRY
from veriform.pipeline import StandardPipeline
from veriform.preprocessing.dag import DAGModel
from veriform.proving import PROVER_REGISTRY


def _coerce(value: str) -> Any:
    """Parse a CLI override value as JSON if possible; fall back to str."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _apply_overrides(cfg_dict: dict, overrides: list[str]) -> dict:
    """Apply ``key.path=value`` overrides into a nested dict."""
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override}")
        path, value = override.split("=", 1)
        keys = path.split(".")
        target = cfg_dict
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = _coerce(value)
    return cfg_dict


def _build_dataset_loader(cfg: RunConfig):
    if cfg.dataset.type != "processbench":
        raise NotImplementedError(f"Dataset type {cfg.dataset.type!r} not yet wired")
    return ProcessBenchLoader(
        file_path=cfg.dataset.file_path,
        num_samples=cfg.dataset.num_samples,
        seed=cfg.dataset.seed,
    )


def _build_perturber(cfg: RunConfig):
    cls = PERTURBER_REGISTRY[cfg.perturbation.type]
    if cfg.perturbation.type == "regex":
        return cls(p=cfg.perturbation.p, **cfg.perturbation.params)
    return cls(p=cfg.perturbation.p, **cfg.perturbation.params)


def _build_formalizer(cfg: RunConfig):
    cls = FORMALIZER_REGISTRY[cfg.formalization.type]
    return cls(sampling=cfg.formalization.sampling, **cfg.formalization.params)


def _build_prover(cfg: RunConfig):
    cls = PROVER_REGISTRY[cfg.proving.type]
    return cls(
        base_url=cfg.proving.base_url,
        batch_size=cfg.proving.batch_size,
        negation_type=cfg.proving.negation,
        **cfg.proving.params,
    )


def _make_output_dir(cfg: RunConfig) -> Path:
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    p_pct = int(cfg.perturbation.p * 100 + 0.5)
    n = cfg.dataset.num_samples or "all"
    name = f"{date_str}_{cfg.formalization.type}_p{p_pct:03d}_n{n}"
    out = Path(cfg.output.dir) / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _serialize_dag(dag: DAGModel) -> list:
    return [
        {
            "node_id": n.node_id,
            "flag": n.flag.value,
            "is_perturbed": n.is_perturbed,
            "content": n.content,
            "formalizer_output": n.formalizer_output,
            "perturbed_content": n.perturbed_content,
            "formalized_content": n.formalized_content,
        }
        for n in dag.nodes
    ]


def run(cfg: RunConfig) -> dict:
    output_dir = _make_output_dir(cfg)
    lean_dir = output_dir / "lean_programs"
    lean_dir.mkdir(parents=True, exist_ok=True)

    chains = _build_dataset_loader(cfg).load()
    perturber = _build_perturber(cfg)
    formalizer = _build_formalizer(cfg)
    prover = _build_prover(cfg) if cfg.proving.enabled else None

    if prover is not None:
        pipeline = StandardPipeline(perturber, formalizer, prover)
    else:
        # Run perturb + formalize only, skipping prove
        def pipeline(chain):
            dag = DAGModel(chain)
            dag = perturber.perturb(dag)
            dag = formalizer.formalize(dag)
            return dag

    data = {
        "config": cfg.model_dump(),
        "results": [],
        "statistics": init_statistics(),
    }
    heatmap_data = init_heatmap_data()

    results_path = output_dir / f"exp_results_{cfg.formalization.type}.json"
    heatmap_path = output_dir / f"heatmap_{cfg.formalization.type}.png"

    for chain in tqdm(chains):
        start = datetime.now()
        dag = pipeline(chain)
        duration = (datetime.now() - start).total_seconds()

        node_dicts = _serialize_dag(dag)
        data["results"].append(
            {"duration": duration, "chain_id": chain.chain_id, "nodes": node_dicts}
        )

        if cfg.output.save_lean_files and prover is not None:
            with open(lean_dir / f"{chain.chain_id}.lean", "w") as f:
                f.write(dag.lean())

        for n in dag.nodes:
            update_statistics(data["statistics"], n)
        update_heatmap_data(heatmap_data, chain.chain_id, dag.nodes)
        calculate_metrics(data["statistics"])

        if cfg.output.save_intermediate:
            with open(results_path, "w") as f:
                json.dump(data, f, indent=4, default=str)
            if cfg.output.save_heatmap:
                draw_heatmap(cfg.perturbation.p, cfg.formalization.type, heatmap_data, heatmap_path)

    # Final flush
    with open(results_path, "w") as f:
        json.dump(data, f, indent=4, default=str)
    if cfg.output.save_heatmap:
        draw_heatmap(cfg.perturbation.p, cfg.formalization.type, heatmap_data, heatmap_path)

    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument(
        "--set",
        nargs="+",
        default=[],
        help="Override config values via dotted-key=value pairs (e.g. perturbation.p=1.0)",
    )
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f) or {}
    cfg_dict = _apply_overrides(cfg_dict, args.set)
    cfg = RunConfig(**cfg_dict)
    run(cfg)


if __name__ == "__main__":
    main()

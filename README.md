# Veriform: Verification-Oriented Formalisation Benchmarking

A framework for measuring the faithfulness of autoformalisation systems in correctly handling incorrect reasoning steps from LLM Chain-of-Thoughts. See `paper/main.tex` for the full method.

## Motivation

Current autoformalisation systems exhibit **sycophancy** — they generate correct Lean proofs even when the underlying reasoning is invalid. A faithful AF should:
- Generate correct Lean proofs for correct reasoning steps (validity preservation, TPR)
- Generate **unprovable** Lean statements for incorrect reasoning steps (invalidity preservation, 1 − FPR)

We aggregate both into a Faithfulness Index `FI = 2·TPR·(1−FPR) / (TPR + (1−FPR))`.

## Methodology

1. **Load** ProcessBench reasoning chains as a DAG of statements.
2. **Perturb** statements with probability `p` (regex or LLM strategy).
3. **Autoformalise** each (possibly perturbed) statement to Lean.
4. **Prove** both the statement and its negation with DeepSeek-Prover-V2.
5. **Score** with TPR/FPR/FI per dataset (GSM8K / MATH / OlympiadBench / OmniMATH).

## Project Structure

```
veriform/
├── src/veriform/
│   ├── data/            # ReasoningChain + dataset loaders
│   ├── preprocessing/   # DAGModel, DAG construction
│   ├── perturbation/    # Regex + LLM perturbers; lean_negation/, ipf, brokenmath (scaffolds)
│   ├── formalization/   # Stepfun, Kimina, Goedel, Herald
│   ├── proving/         # DeepSeekProver + bundled Lean kernel server
│   ├── semantics/       # ProofBridge wrapper for semantic faithfulness (scaffold)
│   ├── evaluation/      # TPR/FPR/FI metrics + heatmaps
│   ├── pipeline.py      # StandardPipeline composition
│   └── config.py        # Pydantic RunConfig schema
├── scripts/
│   ├── run_benchmark.py # Single config-driven entry point
│   ├── shell/           # exps_*.sh drivers (vLLM server orchestration)
│   └── legacy/          # Original v2_exp.py / all_proof.py / etc., kept for reference
├── configs/             # YAML configs consumed by run_benchmark.py
├── tests/
├── data/                # Symlink to /scratch — not in git
├── experiments/         # Symlink to /scratch — not in git
└── paper/               # LaTeX source of paper/main.tex
```

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Default experiment (regex perturbation, Stepfun, DeepSeekProver)
python scripts/run_benchmark.py --config configs/default.yaml

# Override individual fields from the CLI
python scripts/run_benchmark.py --config configs/default.yaml \
    --set formalization.type=goedel perturbation.p=1.0

# Smoke test without GPUs (skip the prover stage)
python scripts/run_benchmark.py --config configs/default.yaml \
    --set proving.enabled=false dataset.num_samples=5
```

Available configs:
- `configs/default.yaml` — regex perturbation, paper baseline
- `configs/llm_perturbation.yaml` — pre-generated LLM perturbations
- `configs/brokenmath_perturbation.yaml` — paper TODO placeholder
- `configs/semantic_filter.yaml` — paper TODO placeholder (ProofBridge)

## Features

- **Pluggable perturbers**: regex, pre-generated LLM, or new BrokenMath/Lean-parser strategies via the `PERTURBER_REGISTRY`.
- **Pluggable formalisers**: 4 fine-tuned 7B–8B models out-of-the-box; LLM end-to-end backends ready to plug in.
- **Per-dataset confusion matrix + heatmap** rendered automatically into `experiments/outputs/<run-id>/`.
- **Single config-driven entry point** with CLI override syntax for sweeps.

## Citation

```bibtex
@misc{veriform2026,
  title={Benchmarking Faithfulness of Mathematical Chain-of-Thought Autoformalisation},
  author={...},
  year={2026}
}
```

## License

MIT License

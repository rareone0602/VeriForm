# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

This is the implementation of `paper/main.tex`. Use can also refer to `README.md` but if there is conflicting information, please adhere to `paper/main.tex`.

## Environment

- **Python**: `~/projects/ruqola/VeriForm/veriform/bin/python` (torch 2.9.0+cu128, transformers 5.3, datasets 4.8)
- **Storage**: Home has 100 GB quota, so some large files like `experiments/` and `data/` are symlinks to `/scratch/users/$USER/`. `HF_HOME` is set in the user environment.
- **GPUs**: 3× H200 NVL (143 GB each). They are occupied so far, but you can refactor the part that uses Lean 4.

## Layout (post-refactor)

`src/veriform/` is organised by pipeline stage:

```
src/veriform/
  data/              # ReasoningChain + dataset loaders (ProcessBench, GSM8K, MATH)
  preprocessing/     # DAGModel — turns a ReasoningChain into a DAG of nodes
  perturbation/      # Inject errors into nodes
    perturbers.py            # StandardPerturber (regex), LLM perturbers
    brokenmath_perturber.py  # SCAFFOLD — paper TODO
    ipf.py                   # SCAFFOLD — Ineffective Perturbation Finder
    lean_negation/           # Negation backends (regex today, Lean parser TODO — see #2)
  formalization/     # Autoformalisation backends (Stepfun, Kimina, Goedel, Herald)
  proving/           # DeepSeekProver + theorem extractor + bundled Lean kernel server
    lean_server/             # 5 GB embedded Lean toolchain (.lake/ ignored by git)
  semantics/         # SCAFFOLD — ProofBridge wrapper for semantic faithfulness
  evaluation/        # TPR/FPR/FI metrics + heatmap plotting
  pipeline.py        # StandardPipeline composition
  config.py          # Pydantic RunConfig schema
```

Single entry point: `scripts/run_benchmark.py --config configs/<name>.yaml`. CLI overrides via `--set key.path=value`. Legacy multi-script flow lives in `scripts/legacy/` for reference (will be removed once the new entry point has been used to reproduce the paper results).

## Improvement

1. ~~Directory redesign~~ — done; see "Layout" above. The legacy `src/veriform/autoformalization{,_v2}/` modules have been removed.
2. **Lean negation backend**: the home is `src/veriform/perturbation/lean_negation/`. Today only `regex_backend.py` is implemented (extracted from `StandardPerturber.logical_negation_str`, behaviour-preserving). To replace regex with Lean's own parser/metaprogramming, drop a new file `lean_parser_backend.py` in there implementing the `NegationBackend` Protocol and register it in `__init__.py::_REGISTRY`. **Discuss the design with the user before starting.**
3. Other improvements you can think of. Several scaffolds are in place (semantics/, brokenmath_perturber, ipf) — discuss before filling in.
4. Git commit after you read the codebase and understand the overall structure. You can also create a new branch for your work.

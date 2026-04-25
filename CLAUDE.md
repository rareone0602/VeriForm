# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project goal

Implementation of `paper/main.tex`. If anything in this file or `README.md` conflicts with the paper, the paper wins.

## Environment

- **Python venv**: `~/projects/ruqola/VeriForm/veriform/bin/python` (torch 2.9.0+cu128, transformers 5.3, datasets 4.8, vllm, openai, pydantic 2).
- **Storage**: home has a 100 GB quota. `data/` and `experiments/` are symlinks to `/scratch/users/$USER/VeriForm/...`. `HF_HOME` is set in the user environment so HF model downloads land on scratch.
- **GPUs**: 3× H200 NVL (143 GB each). Occupancy varies; ask before launching anything heavy. `nvidia-smi` to check.
- **Lean toolchain**: pinned to `leanprover/lean4:v4.9.0-rc1` in `src/veriform/proving/lean_server/lean/lean-toolchain`. The elan install on this machine is partial (`~/.elan` is a symlink to `/scratch/...`). The required setup is:
  - `~/.elan/bin/lake` and `~/.elan/bin/lean` must point at the **v4.9.0-rc1** toolchain. If those symlinks are missing, restore them from the v4.9 toolchain at `/scratch/users/$USER/.elan/toolchains/leanprover--lean4---v4.9.0-rc1/bin/`.
  - If only v4.29 toolchains exist on disk (v4.9 lacks `bin/`), download v4.9.0-rc1 from `https://github.com/leanprover/lean4/releases/download/v4.9.0-rc1/lean-4.9.0-rc1-linux.tar.zst` and extract into the v4.9 toolchain dir. Mathlib/`proofwidgets` will not build with v4.29.
  - The `Negate.lean` exe must be built once: `cd src/veriform/proving/lean_server/lean && ~/.elan/bin/lake build negate` (~few minutes if Mathlib already built; 6 s warm startup per LeanNegator session).

## Layout

```
src/veriform/
  data/              ReasoningChain + ProcessBenchLoader (the only active loader)
  preprocessing/     DAGModel — turns a ReasoningChain into a DAG of nodes
  perturbation/
    perturbers.py            StandardPerturber (regex), LLM perturbers
    nl_negation/             Regex-based negation of natural-language CoT (NOT Lean)
    brokenmath_perturber.py  SCAFFOLD — paper TODO
    ipf.py                   SCAFFOLD — Ineffective Perturbation Finder
  formalization/     Stepfun / Kimina / Goedel / Herald (training-time prompts pinned by tests)
  proving/
    deepseek_prover.py       DeepSeekProver
    theorem_extractor.py     Regex-based theorem extraction (still used; unrelated to negation)
    negation.py              LeanNegator: persistent `lake exe negate` subprocess
    lean_server/
      lean/Negate.lean       Lean 4 metaprogramming negator (parser-only, source-slice rebuild)
      lean/Main.lean         (legacy hello-world, kept for the lakefile)
      lean/lakefile.toml     Has BOTH `ruqola` and `negate` exe targets
      prover/                DeepSeek-Prover REPL infrastructure (subprocess scheduler)
      .lake/                 5 GB Mathlib build artefacts; gitignored
  semantics/         SCAFFOLD — ProofBridge wrapper for semantic faithfulness (paper TODO)
  evaluation/        TPR/FPR/FI metrics + heatmap plotting
  pipeline.py        StandardPipeline (perturb → formalize → prove)
  config.py          Pydantic RunConfig schema for run_benchmark.py
```

Entry points (the only two scripts):

- `scripts/run_benchmark.py --config configs/<name>.yaml` — end-to-end (perturb → formalize → prove → metrics) in one process. CLI overrides via `--set key.path=value`. Setting `proving.enabled=false` runs perturb + formalize only.
- `scripts/run_formalize.py --formalizer <name> --p <prob> --port <vllm-port>` — throughput-optimized phase 1 only. Single-process async fan-out (default `--concurrency 64`) against a vLLM OpenAI endpoint, saturates the GPU on 1 vLLM server. Resume-safe (atomic per-chain pickle write). Use this instead of run_benchmark for large sweeps where you want formalize artefacts saved before proving.

A standalone phase 2 script (`run_prove.py`) does **not** exist yet; if you need the legacy two-phase flow (formalize on GPU A, prove on GPU B), it has to be built. The end-to-end `run_benchmark.py` covers single-GPU runs.

## Hard rules

1. **Formaliser prompts must match training-time format exactly.** Each model in `formalization/fine_tuned.py` is pinned to its official model-card example (system prompt, user-prompt template, sampling params, chat template). `tests/test_formalizer_prompts.py` has 13 snapshot tests that fail loudly on drift. Stepfun specifically requires the literal `<think>` token appended after the chat template (DeepSeek-R1-distilled base; without it the model does not enter reasoning mode). Do not "tidy up" these prompts.

2. **Lean theorem negation goes through `LeanNegator`, not regex.** The persistent `lake exe negate` subprocess parses with Lean's own parser and rebuilds the negated theorem from source slices. `DeepSeekProver.get_refute_theorem` is a one-line delegation. Do not re-introduce a regex-based theorem negator.

3. **`data/processed/dags.json` is required input.** The DAG-construction step (raw ProcessBench → DAGs) lives **outside this repo** (a one-shot LLM call, currently GPT/Gemini). The format is row-major list of `{id, problem, dags: {nodes, final_answer, metadata}, split, generator, ...}`. To rebuild from a column-major upstream file (e.g. `data/dag_processbench_final.json`), transpose, parse the `dags` field as JSON per row, and drop any node whose `node_id` doesn't match the expected `step_<N>` sequence (these are LLM glitches like `step_err_1`).

## Pending work / scaffolds

These all have empty implementations behind real module structure; discuss with the user before filling in:

- `src/veriform/semantics/` — ProofBridge encoder wrapper, similarity, drift filter (paper TODO; semantic faithfulness re-weighting of TPR/FPR).
- `src/veriform/perturbation/brokenmath_perturber.py` — BrokenMath-style LLM perturber.
- `src/veriform/perturbation/ipf.py` — LLMJudgeIPF (Ineffective Perturbation Finder; discord 2026-04-13 follow-up).
- A standalone phase 2 `scripts/run_prove.py` analogous to `run_formalize.py`, if you want to split formalize and prove across GPUs.

## Tests

`tests/` contains only fast tests that pass without GPU/vLLM/network:

- `test_theorem_extractor.py` — pure parser logic.
- `test_lean_negation.py` — 15 cases against the live `lake exe negate` (skipped if `~/.elan/bin/lake` or the built `negate` exe is missing).
- `test_formalizer_prompts.py` — pinning snapshots for all 4 formalisers (instantiates tokenisers; Herald uses a mocked `vllm.LLM`).

Run: `veriform/bin/python -m pytest tests/ -o addopts=""` (the `-o` strips coverage flags from `pyproject.toml` because `pytest-cov` isn't installed in the venv).

## Conventions

- Branches: working branch is `refactor/modular-layout`, pushed to `origin/phy` (the user's branch). `main` is the unrefactored baseline.
- Auth for `git push`: use `https://${GITHUB_ACCESS_TOKEN}@github.com/...` one-shot URL; do not write the token into `.git/config`.
- Commit messages: descriptive subject, blank line, body explaining the why. Co-author trailer for Claude commits.
- Discuss large architectural changes with the user before starting (especially anything touching the formaliser prompts, the Lean negation, or the prover).

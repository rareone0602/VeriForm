# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project goal

Implementation of `overleaf/latex/main.tex`. If anything in this file or `README.md` conflicts with the paper, the paper wins. The paper lives in `overleaf/` (a git clone of the Overleaf project, gitignored from the code repo); pre-Overleaf local files preserved at `paper.local-backup-2026-05-13/`.

## Environment

The project was migrated to the **CCDS AAI cluster** (NTU). `~/server_spec.md` is the canonical hardware/queue/policy reference — consult it for nodes, queues, GPU types, and usage policies. Highlights:

- **Cluster**: SLURM-managed, multi-node. Queues: `NA100q` (A100 SXM4), `PA100q`/`PA10080q` (A100 PCIe), `PH100q`/`NH100q` (H100), `RTXA6Kq` (RTX A6000/A6000 Ada), `HPCq` (12 TB-RAM A100 node). 5-GPU-per-user cap, 7-day wall limit. **No jobs on the head node.**
- **Launch jobs via SLURM**: `srun -p <queue> -n 1 --gres=gpu:N <cmd>` or `sbatch`. Check free GPUs with `bash ~/gpu_free.sh`.
- **SLURM GPU binding gotcha** (see `~/server_spec.md` for canonical write-up):
  - On **node04** and **node13**, `CUDA_VISIBLE_DEVICES` is sometimes assigned to an already-occupied GPU. Verify against `nvidia-smi` and silently rebind to a free GPU before launching compute.
  - On any **other** node, if you observe the same collision, append an entry to `~/logs/slurm_gpu_collisions.log` (jobid, node, assigned device, occupying PID/user, timestamp) for batched reporting to `ccdsccr@ntu.edu.sg`. Do **not** silently rebind on unlisted nodes.
- **Storage** (2 TB home quota): `data/`, `experiments/`, and `brokenmath_perturbations/` are symlinks into `/dataset/phy/VeriForm/` (186 TB pool). `~/projects/scratch` is a convenience symlink to `/dataset/phy/`. Keep new large artefacts under `/dataset/phy/VeriForm/` and symlink them in; **never commit large binaries into the home-side tree**.
- **HF cache**: `HF_HOME=/dataset/phy/hf_cache` (already in the user environment). HF model downloads land there.
- **Python venv**: not yet recreated on this cluster — verify before invoking; the old `~/projects/ruqola/VeriForm/veriform/bin/python` path is stale. Ask the user which interpreter to use if unsure.
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

- `scripts/run_prove.py` — phase 2 analog of `run_formalize.py`. Reads phase-1 pickles produced by `run_formalize.py`, runs DeepSeek-Prover-V2 + Lean against each node, writes phase-2 pickles. Use this for the two-phase flow (formalize on GPU A, prove on GPU B); `run_benchmark.py` is still the end-to-end single-GPU path.

## Hard rules

1. **Never run `python` on the head node. Submit everything via `sbatch` / `srun`.** No exceptions, not even "lightweight" or "quick check" python (no `python -c`, no `<<'PY' ... PY` heredocs, no `venv/bin/python` for pickle introspection). On 2026-05-14 a Claude-driven inline pickle load reached ~75 GB RSS and crashed the head node; admin (Jun Hao) emailed an explicit do-not, and the user has revoked any prior "lightweight is fine" allowance — **all python goes through SLURM**. Use `srun -p PA100q --gres=gpu:0 -c 2 --mem=8G -t 0:10 --pty python ...` for interactive checks or `sbatch --wrap='... python ...'` for one-shots. Shell tools (`ls`, `grep`, `awk`, `wc`, `sacct`, `squeue`, `sbatch`, `scontrol`, `Read`/`Edit`) on the head node are fine; anything that loads a model, a torch tensor, a non-trivial pickle, or numpy/pandas data is not.

2. **Formaliser prompts must match training-time format exactly.** Each model in `formalization/fine_tuned.py` is pinned to its official model-card example (system prompt, user-prompt template, sampling params, chat template). `tests/test_formalizer_prompts.py` has 13 snapshot tests that fail loudly on drift. Stepfun specifically requires the literal `<think>` token appended after the chat template (DeepSeek-R1-distilled base; without it the model does not enter reasoning mode). Do not "tidy up" these prompts.

3. **Lean theorem negation goes through `LeanNegator`, not regex.** The persistent `lake exe negate` subprocess parses with Lean's own parser and rebuilds the negated theorem from source slices. `DeepSeekProver.get_refute_theorem` is a one-line delegation. Do not re-introduce a regex-based theorem negator.

4. **`data/parsed/unperturbed/dags.json` is required input.** The DAG-construction step (raw ProcessBench → DAGs) lives **outside this repo** (a one-shot LLM call, currently GPT/Gemini). The format is row-major list of `{id, problem, dags: {nodes, final_answer, metadata}, split, generator, ...}`. To rebuild from a column-major upstream file (e.g. `data/raw/dag_processbench_final.json`), transpose, parse the `dags` field as JSON per row, and drop any node whose `node_id` doesn't match the expected `step_<N>` sequence (these are LLM glitches like `step_err_1`).

   Data layout (reorganised 2026-05-16):
   - `data/raw/` — upstream raw artefacts (column-major JSON, BrokenMath JSONLs).
   - `data/parsed/unperturbed/dags.json` — the cleaned, row-major DAG file.
   - `data/output/<method>-perturbed/{formalized,proved}/<formalizer>/<p_pct>/` — phase-1/2 pickles, one per chain. `<method>` ∈ {`regex`, `brokenmath-soft`, `brokenmath-medium`, `brokenmath-hard`}. `_legacy/` and `_diag/` siblings hold archived runs.

## Pending work / scaffolds

These all have empty implementations behind real module structure; discuss with the user before filling in:

- `src/veriform/semantics/` — ProofBridge encoder wrapper, similarity, drift filter (paper TODO; semantic faithfulness re-weighting of TPR/FPR).
- `src/veriform/perturbation/brokenmath_perturber.py` — BrokenMath-style LLM perturber.
- `src/veriform/perturbation/ipf.py` — LLMJudgeIPF (Ineffective Perturbation Finder; discord 2026-04-13 follow-up).

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

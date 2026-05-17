# H200 cluster bootstrap

Migration from CCDS AAI (NTU) → NUS H200 cluster, 2026-05-17.

The CCDS cluster ran into squatter saturation on all H100 / A100-80GB nodes
(every fast GPU showed 55-80 GB of non-SLURM process memory; SLURM's free
counts were fictional). Phase B was advancing on distributed A100-40GBs but
the central-vLLM architecture this codebase was designed for needs one good
GPU to be worth using.

The NUS H200 cluster gives us one fast homogeneous box. The migration is
designed so a single H200 (141 GB HBM) hosts vLLM and 4 CPU-only prove
processes hit it over HTTP — the architecture in
[[scripts/run_phase2_all.sh]] before refactor, but with the GPU and the
prove processes decoupled.

## What's where

| Asset | Location |
|---|---|
| Code | `git@github.com:rareone0602/VeriForm.git` branch `migration/h200` (this branch) |
| Bucket | `gs://veriform-faithformbench-2026/` (project `research-496610`, region `asia-southeast1`) |
| Lean cache (Mathlib `.lake`, ~4.9 GB → tar.zst ~?? GB) | `gs://veriform-faithformbench-2026/lean-cache/lake-2026-05-17.tar.zst` |
| Phase-1 pickles (formalized) | `gs://veriform-faithformbench-2026/data/output/<method>-perturbed/formalized/**` |
| Phase-2 pickles (proved, in progress on CCDS) | `gs://veriform-faithformbench-2026/data/output/<method>-perturbed/proved/**` — synced hourly from CCDS until cut-over |
| DAG input | `gs://veriform-faithformbench-2026/data/parsed/unperturbed/dags.json` |
| Paper baselines (per-arm flag heatmaps from main.tex) | `gs://veriform-faithformbench-2026/paper/FaithformBench/*.dat` |

## Bootstrap (one-shot)

```bash
# Pre-req: gcloud SDK installed + authed as rareone0602@gmail.com,
# project set to research-496610. The bootstrap script will check and bail
# if not.
git clone -b migration/h200 git@github.com:rareone0602/VeriForm.git
cd VeriForm
bash scripts/bootstrap_h200.sh        # ~15-25 min depending on network
```

The script does, idempotently:

1. **Storage discovery**: locates `$SCRATCH` (or accepts `--scratch /path`),
   symlinks `data/`, `experiments/`, `brokenmath_perturbations/` into it
   (mirrors the CCDS layout under `/dataset/phy/VeriForm/`).
2. **Python venv**: creates `./venv` if absent, installs from
   `requirements.txt`. Sets `TMPDIR`, `TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT`
   off `/tmp` (carry-over from CCDS: vLLM/Triton fill /tmp and crash nodes,
   see [vllm-tmpdir-redirect memory]).
3. **Lean toolchain**: installs elan, pins to `leanprover/lean4:v4.9.0-rc1`
   (matches `src/veriform/proving/lean_server/lean/lean-toolchain`).
4. **Lean cache restore**: downloads the tarred `.lake` from GCS, extracts
   into `src/veriform/proving/lean_server/lean/.lake/`, then runs
   `lake build negate` (should be a no-op rebuild if cache is intact;
   takes ~6 s if so, ~25 min if not).
5. **HF model prefetch**: pulls DSP-V2-7B + 4 formalizers (~25 GB) to
   `$HF_HOME` from huggingface.co. Uses `HF_TOKEN` from `.env` if present.
6. **Data pull**: `gcloud storage rsync -r gs://.../data/ data/` brings
   in pickles + DAG inputs.

After bootstrap completes, run the verification block at the bottom of the
script (looks for the right counts of pickles, the `negate` exe, and a
trivial `lake env repl` round-trip).

## Architecture target on H200

**Single central vLLM**, multiple CPU-only prove processes:

```
              ┌─────────────────────────────┐
   H200 ─────│  vLLM serve DSP-V2-7B         │
   141 GB     │  --tensor-parallel-size 1     │
              │  --port 8001                  │
              │  --max-model-len 6144 (was    │
              │  16384 on CCDS — see note     │
              │  below)                       │
              └────────┬────────────────────┘
                       │ HTTP localhost:8001
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
  run_prove.py    run_prove.py    run_prove.py
  (kimina)        (goedel)        (herald) ...
  --gres=gpu:0    --gres=gpu:0    --gres=gpu:0
```

Why `--max-model-len 6144` (down from CCDS's 16384): each running vLLM
request reserves `max-model-len` tokens of KV cache. On CCDS A100-40GB the
KV ceiling capped concurrency at 15-28 running with 160-294 waiting (see
the per-arm probe in this session's history). Proofs are <2k tokens in
practice; the 16k reservation was 8× wasteful. With H200's 141 GB and a 6k
ceiling, we should sustain 100+ concurrent — enough for all 4 prove
processes without queueing.

## Hard rules (carried over from CCDS)

These come from incidents that took down nodes; they apply regardless of
cluster:

1. **No Python on the head node.** All Python via sbatch/srun. On
   2026-05-14 a Claude-driven inline pickle load reached 75 GB RSS and
   crashed the CCDS head node. Shell tools (`ls`, `grep`, `awk`, `wc`,
   `squeue`, `Read`/`Edit`) on head are fine; anything that loads a model,
   torch tensor, non-trivial pickle, or numpy/pandas data is not.
2. **Nothing in `/tmp`.** Node-local, small. Always export `TMPDIR`,
   `TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT` to a scratch path. Job 492038
   on CCDS died with "No space left on device" mid-vLLM-init when /tmp
   was already full from another tenant. See [no-tmp-logs] and
   [vllm-tmpdir-redirect] memories.
3. **One prove process per node** (CCDS lesson): each Lean4ServerProcess
   worker mmaps ~2000 Mathlib oleans; 16 workers × 1 process is fine,
   16 workers × 3 processes blew system-wide ENFILE (`file-max=131072`).
   On H200 with one central vLLM and 4 CPU-only prove processes, the prove
   processes can co-locate if the H200 cluster's `file-max` is raised. If
   default (131k), spread them across nodes.
4. **Formaliser prompts pinned**: see [tests/test_formalizer_prompts.py].
   13 snapshot tests fail loudly on drift. Stepfun specifically needs the
   literal `<think>` token after chat template.
5. **Lean theorem negation goes through `LeanNegator`**, not regex.
   `lake exe negate` is a persistent subprocess. Do not re-introduce a
   regex theorem negator.

## Migration state on cut-over (2026-05-17 ~18:30 SGT)

Phase B on CCDS still in progress at cut-over time:
- kimina/regex/p=1.0: 399/1179
- goedel/regex/p=1.0: 131/1179
- herald/regex/p=1.0: 555/1179
- stepfun/regex/p=0.0: 213/1177 (p=1.0 already done at 1179)
- stepfun/regex/p=1.0, brokenmath_soft/{stepfun,kimina,goedel,herald}/p=1.0: all 1179 done

After GCS sync from CCDS, H200 will pick up missing chains via per-chain
pickle skip in `run_prove.py:362` (resume-safe).

Phase C (brokenmath_hard / 4 formalizers / 1179 each) hasn't started.

## Verification after bootstrap

Read `scripts/distributional_test.py` and `scripts/paper_comparison.py`.
Run the paper-comparison against `stepfun/regex/p=1.0` (n=12,784 nodes) —
prover-side flags should match the paper within ±1.4 pp. The only known
drift is AF_FAIL +2.3 pp, which is phase-1 (formalizer-side), not Lean.

If the comparison comes back with much larger deltas (>5 pp on any prover
flag), something in the bootstrap is wrong — likely the `.lake` extraction
or a Lean toolchain mismatch. Re-build the negate exe and re-run before
launching Phase B/C.

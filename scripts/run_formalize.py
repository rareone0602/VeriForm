"""Phase 1 — throughput-optimized autoformalization.

Replaces the legacy `scripts/legacy/all_formalizer.py` which processed chains
serially and used `exps_formalizer.sh` to spawn 32 parallel Python processes
to compensate. This single-process driver fans out via `asyncio` against
vLLM's OpenAI-compatible endpoint, so vLLM's continuous batcher sees up to
hundreds of prompts in flight and saturates the GPU.

Why this is faster:
  - Legacy: per chain, one batched `completions.create([~10 prompts])`,
    next chain blocks until the previous chain's batched call returns.
    vLLM rarely sees more than ~10 prompts in flight from a client.
  - Here:   --concurrency K (default 64) chain-tasks issue their batched
    calls concurrently. vLLM continuously batches across them. Effective
    in-flight ≈ K * mean_nodes_per_chain (~640 with defaults), well above
    the saturation point on a single H200.

Sync prefix runs perturbation deterministically before any network call,
so output ordering doesn't depend on async scheduling. Pickles are saved
per chain as soon as that chain finishes, so interrupting and re-running
resumes exactly where it left off.

Example:
    # Goedel formalizer must already be served by vLLM at the given port:
    #   CUDA_VISIBLE_DEVICES=2 vllm serve Goedel-LM/Goedel-Formalizer-V2-8B \\
    #     --port 8002 --tensor-parallel-size 1 --dtype bfloat16 \\
    #     --enable-prefix-caching --max-num-seqs 256

    python scripts/run_formalize.py --formalizer goedel --p 1.0 --port 8002
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pickle
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from veriform.data.loaders import ProcessBenchLoader
from veriform.formalization import FORMALIZER_REGISTRY
from veriform.perturbation.perturbers import StandardPerturber
from veriform.preprocessing.dag import DAGModel, Flagging


PROMPT_LENGTH_CAP = 4000  # mirrors BaseFormalizer.formalize behaviour


def _build_prompts_for_chain(
    dag: DAGModel,
    formalizer,
) -> tuple[list[str], list]:
    """Build (prompts, nodes) for the non-declarative, in-budget nodes of one DAG.

    Mirrors the prompt-building portion of `BaseFormalizer.formalize`. Side
    effects on the DAG: nodes whose prompt is over the length cap are flagged
    AF_FAIL here so we don't even send them.
    """
    prompts: list[str] = []
    nodes: list = []
    for node in dag.nodes:
        if node.flag == Flagging.DECLARATIVE:
            continue
        formalizer.initialize_dialog()
        prompt = formalizer._get_llm_prompt(node.contextualized())
        if len(prompt) > PROMPT_LENGTH_CAP:
            node.formalized_content = "-- Failed to formalize"
            node.flag = Flagging.AF_FAIL
            node.formalizer_output = None
            continue
        prompts.append(prompt)
        nodes.append(node)
    return prompts, nodes


def _apply_responses(nodes: list, response, formalizer) -> None:
    """Mirrors the post-processing portion of `BaseFormalizer._formalize_prompt`
    and `BaseFormalizer.formalize`."""
    for node, choice in zip(nodes, response.choices):
        try:
            generated_text = choice.text.strip()
            lean_code = formalizer.parse_lean_code(generated_text)
        except Exception:
            generated_text = getattr(choice, "text", "") or ""
            lean_code = None
        if lean_code is None:
            node.formalized_content = "-- Failed to formalize"
            node.flag = Flagging.AF_FAIL
        else:
            node.formalized_content = lean_code
            node.flag = Flagging.UNKNOWN
        node.formalizer_output = generated_text


async def _formalize_chain(
    dag: DAGModel,
    prompts: list[str],
    nodes: list,
    save_path: Path,
    client: AsyncOpenAI,
    formalizer,
    sem: asyncio.Semaphore,
) -> str:
    """One async worker: send the chain's batched completion request, save."""
    async with sem:
        response = await client.completions.create(
            model=formalizer.MODEL_DIR,
            prompt=prompts,
            **formalizer.gen_kwargs,
        )
    _apply_responses(nodes, response, formalizer)
    # Atomic write: write to .tmp then rename, so a SIGINT mid-write doesn't
    # leave a corrupt pkl that the resume check then trusts.
    tmp = save_path.with_suffix(save_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(dag, f)
    os.replace(tmp, save_path)
    return dag.id


async def main_async(args: argparse.Namespace) -> None:
    save_dir = Path(args.output_dir) / args.formalizer / f"{int(args.p * 100 + 0.5)}"
    save_dir.mkdir(parents=True, exist_ok=True)

    chains = ProcessBenchLoader(
        file_path=args.dags_path,
        num_samples=args.num_samples,
    ).load()
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(chains)
    print(f"Loaded {len(chains)} chains; saving pickles under {save_dir}")

    perturber = StandardPerturber(
        p=args.p,
        operator_swap=True,
        value_change=True,
        logical_negation=True,
    )
    formalizer_cls = FORMALIZER_REGISTRY[args.formalizer]
    formalizer = formalizer_cls(
        sampling=args.sampling,
        base_url=f"http://localhost:{args.port}/v1",
    )
    client = AsyncOpenAI(api_key="EMPTY", base_url=f"http://localhost:{args.port}/v1", timeout=3600)

    # ---- Sync prefix: perturb every not-yet-saved chain deterministically. ----
    pending: list[tuple[DAGModel, list[str], list, Path]] = []
    skipped = empty = 0
    random.seed(args.seed)  # deterministic perturbation across runs
    for chain in chains:
        save_path = save_dir / f"{chain.chain_id}.pkl"
        if save_path.exists():
            skipped += 1
            continue
        dag = DAGModel(chain)
        dag = perturber.perturb(dag)
        prompts, nodes = _build_prompts_for_chain(dag, formalizer)
        if not prompts:
            with open(save_path, "wb") as f:
                pickle.dump(dag, f)
            empty += 1
            continue
        pending.append((dag, prompts, nodes, save_path))
    print(
        f"resume status: {skipped} already done, {empty} had no formalizable nodes, "
        f"{len(pending)} chains queued for formalization"
    )
    if not pending:
        return

    # ---- Async fan-out. ----
    sem = asyncio.Semaphore(args.concurrency)
    total_prompts = sum(len(p) for _, p, _, _ in pending)
    print(
        f"dispatching {len(pending)} chain-tasks ({total_prompts} prompts) "
        f"with concurrency={args.concurrency}"
    )
    started = time.time()
    tasks = [
        _formalize_chain(dag, prompts, nodes, save_path, client, formalizer, sem)
        for dag, prompts, nodes, save_path in pending
    ]
    completed = 0
    for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="chains"):
        try:
            await fut
            completed += 1
        except Exception as e:
            # Don't kill the whole run on one chain failure — log and continue.
            print(f"WARN: chain failed: {type(e).__name__}: {e}")
    elapsed = time.time() - started
    print(
        f"done: {completed}/{len(pending)} chains in {elapsed:.1f}s "
        f"({completed / max(elapsed, 1e-6):.2f} chains/s, "
        f"{total_prompts / max(elapsed, 1e-6):.1f} prompts/s)"
    )

    await client.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--formalizer", required=True, choices=list(FORMALIZER_REGISTRY))
    p.add_argument("--p", type=float, default=1.0, help="perturbation probability")
    p.add_argument("--port", type=int, default=8002, help="vLLM OpenAI-compatible server port")
    p.add_argument("--num_samples", type=int, default=None, help="cap chains (None = all)")
    p.add_argument("--concurrency", type=int, default=64, help="max concurrent chain requests in flight")
    p.add_argument("--sampling", choices=["recommended", "deterministic"], default="deterministic")
    p.add_argument("--dags_path", default="./data/processed/dags.json")
    p.add_argument("--output_dir", default="./data/regex_perturbed/formalized")
    p.add_argument("--seed", type=int, default=42, help="seed for perturbation + chain ordering")
    p.add_argument("--shuffle", action="store_true", help="shuffle chains for load-balanced restart")
    args = p.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

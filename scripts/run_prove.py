"""Phase 2 — throughput-optimized prover driver.

The prove-phase analog of `scripts/run_formalize.py`. Reads phase-1 pickles
produced by `run_formalize.py` (formalized DAGs with nodes flagged UNKNOWN)
and writes phase-2 pickles in a parallel directory tree, with each node
classified as PROVED / REFUTED / UNKNOWN / TC_FAIL.

Architecture (single process, async fan-out):
  1. Sync prefix per chain: build the proof prompts (which calls
     LeanNegator.negate — single-subprocess, lock-serialised, so we do
     this serially in the main process). Submit the two TC requests
     (orig + negated theorem) to the shared Lean4ServerScheduler.
  2. Async fan-out (Semaphore(args.concurrency)): per chain await the TC
     results, batch the chain's proof prompts into ONE
     completions.create call against the DeepSeek-Prover-V2 vLLM server,
     filter the candidates, submit them to the shared Lean scheduler,
     await Lean results, classify each node, atomically write per-chain
     pickle.

We do NOT modify DeepSeekProver. We construct one instance and call its
helper methods (get_proof_prompts, filter_proof_candidates) and reuse its
gen_kwargs verbatim — the per-prompt sampling distribution must match
what `pipeline.py` produces.

Example:
    # DeepSeek-Prover-V2-7B must already be served by vLLM on the given port:
    #   CUDA_VISIBLE_DEVICES=0 vllm serve deepseek-ai/DeepSeek-Prover-V2-7B \\
    #     --port 8001 --tensor-parallel-size 1 --dtype bfloat16 \\
    #     --max-model-len 16384

    python scripts/run_prove.py --formalizer goedel --p 1.0 --port 8001
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pickle
import random
import time
from pathlib import Path
from typing import Any, List, Tuple

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from veriform.preprocessing.dag import DAGModel, Flagging
from veriform.proving.deepseek_prover import (
    DeepSeekProver,
    HEADER,
    LEAN_TEMPLATE,
    TheoremData,
)
from veriform.proving.lean_server.prover.lean.verifier import Lean4ServerScheduler
from veriform.proving.negation import LeanNegator


def _list_phase1_pickles(in_dir: Path) -> List[Path]:
    return sorted(in_dir.glob("*.pkl"))


def _build_chain_sync(
    dag: DAGModel,
    prover: DeepSeekProver,
    scheduler: Lean4ServerScheduler,
) -> List[Tuple[Any, TheoremData, int, int]]:
    """Sync prefix: per chain, for each non-declarative non-AF_FAIL node:
       - build proof prompts (calls LeanNegator under the hood),
       - submit TC requests for orig + negated theorem.

    Returns a list of (node, proof_request, tc_orig_id, tc_neg_id).
    Mirrors the `taskqueue` construction in `DeepSeekProver.prove`.
    """
    taskqueue: List[Tuple[Any, TheoremData, int, int]] = []
    for node in dag.nodes:
        if (
            node.flag == Flagging.DECLARATIVE
            or node.flag == Flagging.AF_FAIL
            or node.formalized_content is None
            or len(node.formalized_content) > 10000
        ):
            continue

        prover.initialize_dialog()

        try:
            proof_request = prover.get_proof_prompts(
                problem_nl_dir=node.perturbed_content
                if node.perturbed_content is not None
                else node.content,
                problem_fl_dir=node.formalized_content,
            )
        except Exception as e:
            # Negation/parse failure — treat the node as AF_FAIL.
            print(f"WARN: get_proof_prompts failed for chain {dag.id} node {node.node_id}: "
                  f"{type(e).__name__}: {e}")
            node.flag = Flagging.AF_FAIL
            continue

        if proof_request.flag == Flagging.AF_FAIL:
            node.flag = Flagging.AF_FAIL
            continue

        # TC the original theorem.
        tc_orig_id = scheduler.submit_request(
            dict(code=HEADER + "\n\n" + node.formalized_content,
                 ast=False, tactics=False)
        )
        proof_request.tc_request_id = tc_orig_id

        # TC the negated theorem (already produced inside get_proof_prompts,
        # but the prover doesn't expose it; recompute via LeanNegator. This
        # call is the same one .get_proof_prompts already made, so the
        # subprocess is warm — cost ~ms).
        try:
            negated_src = prover.get_refute_theorem(node.formalized_content)
        except Exception as e:
            print(f"WARN: re-negation failed for chain {dag.id} node {node.node_id}: "
                  f"{type(e).__name__}: {e}")
            node.flag = Flagging.AF_FAIL
            continue
        tc_neg_id = scheduler.submit_request(
            dict(code=HEADER + "\n\n" + negated_src,
                 ast=False, tactics=False)
        )

        taskqueue.append((node, proof_request, tc_orig_id, tc_neg_id))
    return taskqueue


async def _await_request(
    scheduler: Lean4ServerScheduler,
    request_id: int,
    loop: asyncio.AbstractEventLoop,
) -> Any:
    """Wrap the blocking get_request_outputs (busy-polls with time.sleep)
    in run_in_executor so it doesn't block the asyncio loop."""
    return await loop.run_in_executor(
        None, scheduler.get_request_outputs, request_id
    )


async def _await_requests(
    scheduler: Lean4ServerScheduler,
    request_ids: List[int],
    loop: asyncio.AbstractEventLoop,
) -> List[Any]:
    if not request_ids:
        return []
    return await asyncio.gather(
        *[_await_request(scheduler, rid, loop) for rid in request_ids]
    )


async def _prove_chain(
    dag: DAGModel,
    taskqueue: List[Tuple[Any, TheoremData, int, int]],
    save_path: Path,
    client: AsyncOpenAI,
    prover: DeepSeekProver,
    scheduler: Lean4ServerScheduler,
    sem: asyncio.Semaphore,
    loop: asyncio.AbstractEventLoop,
) -> Tuple[str, int]:
    """Per-chain async worker. Returns (chain_id, prompts_sent)."""
    async with sem:
        # ---- 1. Wait for TC results (orig + negated) ----
        tc_ids: List[int] = []
        for _, pr, tc_orig, tc_neg in taskqueue:
            tc_ids.append(tc_orig)
            tc_ids.append(tc_neg)
        tc_results = await _await_requests(scheduler, tc_ids, loop)

        # Mark TC_FAIL on nodes whose original theorem fails to typecheck.
        for i, (node, pr, _, _) in enumerate(taskqueue):
            tc_orig_pass = tc_results[2 * i]['pass']
            # We don't act on tc_neg here — DeepSeekProver.prove only logs
            # the discrepancy. We preserve that by simply ignoring it.
            if not tc_orig_pass:
                pr.flag = node.flag = Flagging.TC_FAIL

        # ---- 2. Build the prompt batch for non-TC_FAIL nodes ----
        prompts: List[str] = []
        prompt_owners: List[int] = []  # index into taskqueue
        for i, (_, pr, _, _) in enumerate(taskqueue):
            if pr.flag in (Flagging.AF_FAIL, Flagging.TC_FAIL):
                continue
            prompts.append(pr.prompt_dir)
            prompts.append(pr.prompt_neg)
            prompt_owners.append(i)

        # ---- 3. One batched completion call per chain ----
        if prompts:
            response = await client.completions.create(
                model=DeepSeekProver.MODEL_NAME,
                prompt=prompts,
                **prover.gen_kwargs,
            )
            choices = response.choices
        else:
            choices = []

        # Layout: 2*b choices per owner, [dir_1..dir_b, neg_1..neg_b] per owner.
        b = prover.batch_size
        for k, owner_idx in enumerate(prompt_owners):
            node, pr, _, _ = taskqueue[owner_idx]
            resp_dir = choices[2 * b * k: 2 * b * k + b]
            resp_neg = choices[2 * b * k + b: 2 * b * (k + 1)]
            cands_dir = [c.text.strip() for c in resp_dir]
            cands_neg = [c.text.strip() for c in resp_neg]
            pr.proof_candidates_dir = prover.filter_proof_candidates(
                target_param=pr.param_dir,
                target_body=pr.body_dir,
                proof_candidates=cands_dir,
                comment=pr.problem_nl_dir,
            )
            pr.proof_candidates_neg = prover.filter_proof_candidates(
                target_param=pr.param_neg,
                target_body=pr.body_neg,
                proof_candidates=cands_neg,
                comment=pr.problem_nl_neg,
            )

        # ---- 4. Submit candidate proofs, await Lean results, classify ----
        # Submit per-node so we can track which results belong to which node
        # without the data_num indexing trick that DeepSeekProver.submit_proof
        # uses (and that has a suspicious i==0 edge case at line 305).
        per_node_dir_ids: List[List[int]] = []
        per_node_neg_ids: List[List[int]] = []
        owner_indices: List[int] = []
        for i, (node, pr, _, _) in enumerate(taskqueue):
            if pr.flag in (Flagging.AF_FAIL, Flagging.TC_FAIL):
                continue
            dir_ids = [
                scheduler.submit_request(
                    dict(code=full, ast=False, tactics=False)
                )
                for _, full in pr.proof_candidates_dir
            ]
            neg_ids = [
                scheduler.submit_request(
                    dict(code=full, ast=False, tactics=False)
                )
                for _, full in pr.proof_candidates_neg
            ]
            per_node_dir_ids.append(dir_ids)
            per_node_neg_ids.append(neg_ids)
            owner_indices.append(i)

        # Flatten + await + un-flatten.
        flat_ids: List[int] = []
        for ids in per_node_dir_ids:
            flat_ids.extend(ids)
        for ids in per_node_neg_ids:
            flat_ids.extend(ids)
        flat_results = await _await_requests(scheduler, flat_ids, loop)

        cursor = 0
        dir_results_per_owner: List[List[Any]] = []
        for ids in per_node_dir_ids:
            n = len(ids)
            dir_results_per_owner.append(flat_results[cursor: cursor + n])
            cursor += n
        neg_results_per_owner: List[List[Any]] = []
        for ids in per_node_neg_ids:
            n = len(ids)
            neg_results_per_owner.append(flat_results[cursor: cursor + n])
            cursor += n

        # Classify each non-TC_FAIL node.
        for k, owner_idx in enumerate(owner_indices):
            node, pr, _, _ = taskqueue[owner_idx]
            results_dir = dir_results_per_owner[k]
            results_neg = neg_results_per_owner[k]

            valid_proof = None
            for (code, _), result in zip(pr.proof_candidates_dir, results_dir):
                if result.get('complete'):
                    pr.flag = Flagging.PROVED
                    pr.proof_candidates_dir = [(code, '')]
                    valid_proof = code
                    break
            if valid_proof is None:
                for (code, _), result in zip(pr.proof_candidates_neg, results_neg):
                    if result.get('complete'):
                        pr.flag = Flagging.REFUTED
                        pr.proof_candidates_neg = [(code, '')]
                        valid_proof = code
                        break
            if valid_proof is None:
                pr.flag = Flagging.UNKNOWN

            node.flag = pr.flag
            if pr.flag == Flagging.PROVED:
                node.formalized_content = pr.proof_candidates_dir[0][0]
            elif pr.flag == Flagging.REFUTED:
                node.formalized_content = pr.proof_candidates_neg[0][0]

    # ---- 5. Atomic per-chain write ----
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = save_path.with_suffix(save_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(dag, f)
    os.replace(tmp, save_path)
    return dag.id, len(prompts)


async def main_async(args: argparse.Namespace) -> None:
    p_pct = f"{int(args.p * 100 + 0.5)}"
    in_dir = Path(args.input_dir) / args.formalizer / p_pct
    out_dir = Path(args.output_dir) / args.formalizer / p_pct
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        raise SystemExit(f"input dir not found: {in_dir}")

    pkls = _list_phase1_pickles(in_dir)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(pkls)
    if args.num_samples is not None:
        pkls = pkls[: args.num_samples]
    print(f"Found {len(pkls)} phase-1 pickles in {in_dir}")

    # ---- Shared infra (one of each) ----
    scheduler = Lean4ServerScheduler(
        max_concurrent_requests=args.lean_concurrency,
        timeout=300,
        memory_limit=-1,
        name='verifier',
    )
    negator = LeanNegator()
    negator.start()
    # Pass scheduler + negator in so DeepSeekProver does NOT spawn its own
    # throwaway 64-worker Lean4ServerScheduler (which previously corrupted
    # shared mp state and silently broke the outer scheduler's workers).
    prover = DeepSeekProver(
        base_url=f"http://localhost:{args.port}/v1",
        batch_size=args.batch_size,
        negation_type=args.negation,
        negator=negator,
        scheduler=scheduler,
    )

    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{args.port}/v1",
        timeout=7200,
    )

    # ---- Sync prefix: load each chain, build prompts + submit TC requests. ----
    pending: List[Tuple[DAGModel, List[Tuple[Any, TheoremData, int, int]], Path]] = []
    skipped = empty = 0
    for pkl in pkls:
        save_path = out_dir / pkl.name
        if save_path.exists():
            skipped += 1
            continue
        try:
            with open(pkl, "rb") as f:
                dag: DAGModel = pickle.load(f)
        except Exception as e:
            print(f"WARN: failed to load {pkl}: {type(e).__name__}: {e}")
            continue
        taskqueue = _build_chain_sync(dag, prover, scheduler)
        if not taskqueue:
            # Defensive: ensure the per-chain parent dir exists (the top-level
            # mkdir(parents=True) at script start has been observed to fail
            # silently on the scratch FS for the leaf p_pct dir).
            save_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = save_path.with_suffix(save_path.suffix + ".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(dag, f)
            os.replace(tmp, save_path)
            empty += 1
            continue
        pending.append((dag, taskqueue, save_path))
    print(
        f"resume status: {skipped} already done, {empty} had no provable nodes, "
        f"{len(pending)} chains queued for proving"
    )
    if not pending:
        scheduler.close()
        negator.close()
        await client.close()
        return

    # ---- Async fan-out. ----
    sem = asyncio.Semaphore(args.concurrency)
    loop = asyncio.get_running_loop()
    total_prompts = sum(
        2 * sum(1 for _ in tq)  # 2 prompts (orig + neg) per node in tq
        for _, tq, _ in pending
    )
    print(
        f"dispatching {len(pending)} chain-tasks (~{total_prompts} prompts) "
        f"with concurrency={args.concurrency}, batch_size={args.batch_size}, "
        f"negation={args.negation}, lean_concurrency={args.lean_concurrency}"
    )
    started = time.time()
    tasks = [
        _prove_chain(dag, tq, save_path, client, prover, scheduler, sem, loop)
        for dag, tq, save_path in pending
    ]
    completed = 0
    sent_prompts = 0
    for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="chains"):
        try:
            _, n_prompts = await fut
            completed += 1
            sent_prompts += n_prompts
        except Exception as e:
            print(f"WARN: chain failed: {type(e).__name__}: {e}")
    elapsed = time.time() - started
    print(
        f"done: {completed}/{len(pending)} chains in {elapsed:.1f}s "
        f"({completed / max(elapsed, 1e-6):.2f} chains/s, "
        f"{sent_prompts / max(elapsed, 1e-6):.1f} prompts/s)"
    )

    # ---- Teardown ----
    await client.close()
    scheduler.close()
    negator.close()
    # Prevent prover.__del__ from double-closing the shared scheduler we
    # already closed. __del__ unconditionally calls .scheduler.close(), so
    # swap in a no-op shim instead of None (which would raise AttributeError).
    class _NoOpClose:
        def close(self) -> None: ...
    prover.scheduler = _NoOpClose()  # type: ignore[assignment]
    prover._owns_negator = False


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--formalizer", required=True,
                   help="formalizer name (matches the phase-1 directory)")
    p.add_argument("--p", type=float, default=1.0, help="perturbation probability")
    p.add_argument("--port", type=int, default=8001,
                   help="DeepSeek-Prover-V2 vLLM server port")
    p.add_argument("--num_samples", type=int, default=None,
                   help="cap chains (None = all)")
    p.add_argument("--concurrency", type=int, default=16,
                   help="max concurrent chain-tasks in flight")
    p.add_argument("--batch_size", type=int, default=2,
                   help="n candidates per direction (passed to DeepSeekProver)")
    p.add_argument("--negation", choices=["strong", "full"], default="full",
                   help="Lean negation mode (matches configs/default.yaml)")
    p.add_argument("--lean_concurrency", type=int, default=64,
                   help="max_concurrent_requests for Lean4ServerScheduler")
    p.add_argument("--input_dir", default="./data/output/regex-perturbed/formalized")
    p.add_argument("--output_dir", default="./data/output/regex-perturbed/proved")
    p.add_argument("--seed", type=int, default=42,
                   help="seed for chain ordering when --shuffle is set")
    p.add_argument("--shuffle", action="store_true",
                   help="shuffle chains for load-balanced restart")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

"""Aggregate phase-2 prove pickles into a paper-asset JSON.

Walks `<input_dir>/<formalizer>/<p_pct>/*.pkl` (the layout written by
`scripts/run_prove.py`) and emits one JSON per formalizer matching the
schema of `paper/assets/p1_results_<formalizer>.json`:

  {
    "config":     {p, num_samples, formalizer, negation_type, sampling},
    "results":    [{duration, chain_id, nodes: [...]}, ...],
    "statistics": {total, af_failed_count, ..., f1_score, accuracy,
                   recall, precision}
  }

Reuses `init_statistics` / `update_statistics` / `calculate_metrics`
from `veriform.evaluation.metrics` so the confusion-matrix logic
stays canonical. Optional `--compare paper/assets` prints a one-page
table comparing the new run (new_impl BrokenMath) against the paper
regex baseline aggregated from `<compare>/FaithformBench/<f>_perturbed.dat`
(old_impl Regex), and against the current-pipeline regex re-run pickles
under `--regex_rerun_root` (new_impl Regex; only formalizers with pickles
on disk are shown — typically just stepfun).

Example:
    python scripts/aggregate_results.py \\
        --input_dir data/output/brokenmath-soft-perturbed/proved \\
        --output_dir paper/assets \\
        --suffix brokenmath \\
        --compare paper/assets
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

from veriform.evaluation import (
    calculate_metrics,
    init_statistics,
    update_statistics,
)
from veriform.preprocessing.dag import Flagging


def _node_to_dict(node) -> Dict[str, Any]:
    return {
        "node_id": node.node_id,
        "flag": node.flag.value if isinstance(node.flag, Flagging) else node.flag,
        "is_perturbed": node.is_perturbed,
        "content": node.content,
        "formalizer_output": node.formalizer_output,
        "perturbed_content": node.perturbed_content,
        "formalized_content": node.formalized_content,
    }


def _aggregate_one(
    pkl_dir: Path,
    formalizer: str,
    p_value: float,
    negation_type: str,
    sampling: str,
) -> Dict[str, Any]:
    """Walk one (formalizer, p) directory and build the paper-asset dict."""
    pkls = sorted(pkl_dir.glob("*.pkl"))
    results: List[Dict[str, Any]] = []
    stats = init_statistics()
    for fp in pkls:
        try:
            with open(fp, "rb") as fh:
                dag = pickle.load(fh)
        except Exception as e:
            print(f"WARN: failed to load {fp}: {type(e).__name__}: {e}")
            continue
        nodes_dicts = [_node_to_dict(n) for n in dag.nodes]
        results.append({
            "duration": 0.0,  # we don't track per-chain wall time in the new pipeline
            "chain_id": dag.id,
            "nodes": nodes_dicts,
        })
        for node in dag.nodes:
            update_statistics(stats, node)
    calculate_metrics(stats)
    return {
        "config": {
            "p": p_value,
            "num_samples": len(pkls),
            "formalizer": formalizer,
            "negation_type": negation_type,
            "sampling": sampling,
        },
        "results": results,
        "statistics": stats,
    }


def _sycophancy(stats: Dict[str, Any]) -> float:
    # FPR over the formalizable perturbed denominator, per paper/sections/problem.tex:240-243.
    return stats["proved_count"] / max(stats["total"], 1) * 100.0


def _aggregate_dat(dat_path: Path) -> Dict[str, Any]:
    # FaithformBench/<formalizer>_perturbed.dat columns: x y val percentage.
    # x mapping (verified empirically): 0=AF-Fail, 1=TC-Fail, 2=Refuted, 3=Unknown, 4=Proved.
    counts = {i: 0 for i in range(5)}
    with open(dat_path) as fh:
        next(fh)
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            counts[int(parts[0])] += int(parts[2])
    total = sum(counts.values())
    return {
        "af_failed_count": counts[0],
        "tc_failed_count": counts[1],
        "refuted_count": counts[2],
        "unknown_count": counts[3],
        "proved_count": counts[4],
        "declarative_count": 0,
        "total": total,
        "f1_score": float("nan"),
    }


def _print_comparison(new_assets: Dict[str, Dict], paper_dir: Optional[Path], suffix: str,
                      regex_rerun_root: Optional[Path]) -> None:
    if paper_dir is None:
        return
    print()
    print("=" * 100)
    print(f"COMPARISON: new_impl(BrokenMath) vs old_impl(Regex) vs new_impl(Regex)  (--suffix {suffix})")
    print("=" * 100)
    print(f"{'formalizer':10s} {'src':>14s}  {'n':>5s} {'PROV%':>7s} {'REF%':>7s} {'UNK%':>7s} "
          f"{'TC_F%':>7s} {'AF%':>7s} {'syc%':>7s} {'F1':>7s}")
    print("-" * 100)
    for f in sorted(new_assets):
        new = new_assets[f]
        rows: list = [("new_impl(BM)", new["statistics"], new["config"]["num_samples"])]
        # old_impl Regex: paper-published baseline aggregated from FaithformBench/<f>_perturbed.dat.
        dat_path = paper_dir / "FaithformBench" / f"{f}_perturbed.dat"
        if dat_path.exists():
            rows.append(("old_impl(R)", _aggregate_dat(dat_path), 1179))
        # new_impl Regex: current pipeline running regex perturbation.
        # Only stepfun has the full corpus on disk; other formalizers are skipped if absent.
        if regex_rerun_root is not None:
            regex_dir = regex_rerun_root / f / "100"
            if regex_dir.exists() and any(regex_dir.glob("*.pkl")):
                asset = _aggregate_one(regex_dir, f, 1.0, "full", "deterministic")
                rows.append(("new_impl(R)", asset["statistics"], asset["config"]["num_samples"]))
        for src, s, n in rows:
            tot = max(s["total"], 1)
            f1 = s.get("f1_score", float("nan"))
            f1_str = f"{f1:>6.3f}" if isinstance(f1, (int, float)) and f1 == f1 else f"{'n/a':>6s}"
            print(f"{f:10s} {src:>14s}  {n:>5d} "
                  f"{100*s['proved_count']/tot:>6.2f}% "
                  f"{100*s['refuted_count']/tot:>6.2f}% "
                  f"{100*s['unknown_count']/tot:>6.2f}% "
                  f"{100*s['tc_failed_count']/tot:>6.2f}% "
                  f"{100*s['af_failed_count']/tot:>6.2f}% "
                  f"{_sycophancy(s):>6.2f}% "
                  f"{f1_str}")
        print()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input_dir", type=Path, required=True,
                   help="directory containing <formalizer>/<p_pct>/*.pkl")
    p.add_argument("--output_dir", type=Path, default=Path("paper/assets"),
                   help="where to write p1_<suffix>_<formalizer>.json files")
    p.add_argument("--suffix", default="brokenmath",
                   help="filename suffix (e.g. p1_<suffix>_<formalizer>.json)")
    p.add_argument("--p_pct", default="100",
                   help="sub-directory under each formalizer to aggregate (e.g. 100, 0)")
    p.add_argument("--negation_type", default="full",
                   help="value to record in config.negation_type")
    p.add_argument("--sampling", default="deterministic",
                   help="value to record in config.sampling")
    p.add_argument("--compare", type=Path, default=None,
                   help="if given, print a comparison table against the paper baseline "
                        "(<compare>/FaithformBench/<formalizer>_perturbed.dat) and the new-impl "
                        "regex re-run (<regex_rerun_root>/<formalizer>/100/) where available")
    p.add_argument("--regex_rerun_root", type=Path,
                   default=Path("data/output/regex-perturbed/proved"),
                   help="root of the new-impl regex re-run pickles (per-formalizer subdirs); "
                        "defaults to data/output/regex-perturbed/proved")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    p_value = float(args.p_pct) / 100.0

    new_assets: Dict[str, Dict] = {}
    for formalizer_dir in sorted(args.input_dir.iterdir()):
        if not formalizer_dir.is_dir():
            continue
        formalizer = formalizer_dir.name
        pkl_dir = formalizer_dir / args.p_pct
        if not pkl_dir.exists():
            print(f"skipping {formalizer}: no {pkl_dir}")
            continue
        asset = _aggregate_one(pkl_dir, formalizer, p_value, args.negation_type, args.sampling)
        out_path = args.output_dir / f"p1_{args.suffix}_{formalizer}.json"
        with open(out_path, "w") as fh:
            json.dump(asset, fh, indent=4, default=str)
        s = asset["statistics"]
        print(f"wrote {out_path}  "
              f"(n={asset['config']['num_samples']}, "
              f"sycophancy={_sycophancy(s):.2f}%, F1={s['f1_score']:.3f})")
        new_assets[formalizer] = asset

    _print_comparison(new_assets, args.compare, args.suffix, args.regex_rerun_root)


if __name__ == "__main__":
    main()

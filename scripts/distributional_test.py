"""
Distributional test: compare per-node flag distributions between
pre-refactor (old verifier.py, spawn-per-call) and post-refactor
(PersistentRepl) pickles for the same arm.

We bucket pickles by mtime date and treat (date <= cutoff_old) as OLD and
(date >= cutoff_new) as NEW. Inside each bucket we tally Flagging values
across all nodes in all chains, then run a chi-square test on the
6 x 2 contingency table.

Usage:  python scripts/distributional_test.py <pickle_dir> [cutoff_old YYYY-MM-DD] [cutoff_new YYYY-MM-DD]
Default cutoff_old = 2026-05-13, cutoff_new = 2026-05-16.
"""

from __future__ import annotations

import datetime as dt
import pickle
import sys
from collections import Counter
from pathlib import Path

# Re-establish the import path that lets pickle resolve veriform.* classes
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scipy.stats import chi2_contingency  # noqa: E402
from veriform.preprocessing.dag import Flagging  # noqa: E402


FLAG_ORDER = [
    Flagging.PROVED,
    Flagging.REFUTED,
    Flagging.UNKNOWN,
    Flagging.DECLARATIVE,
    Flagging.AF_FAIL,
    Flagging.TC_FAIL,
]


def iter_nodes(dag):
    """Yield all DAGNode objects in a DAGModel pickle."""
    # DAGModel stores a graph; nodes accessible as `.nodes` attribute or via
    # the underlying ReasoningChain. Try both.
    nodes = getattr(dag, "nodes", None)
    if nodes is None and hasattr(dag, "dag"):
        nodes = getattr(dag.dag, "nodes", None)
    if nodes is None:
        return
    for n in nodes:
        yield n


def main():
    pickle_dir = Path(sys.argv[1])
    cutoff_old = dt.date.fromisoformat(sys.argv[2] if len(sys.argv) > 2 else "2026-05-13")
    cutoff_new = dt.date.fromisoformat(sys.argv[3] if len(sys.argv) > 3 else "2026-05-16")

    print(f"# Distributional test")
    print(f"# dir          : {pickle_dir}")
    print(f"# OLD bucket   : mtime <= {cutoff_old}")
    print(f"# NEW bucket   : mtime >= {cutoff_new}")
    print()

    buckets = {"OLD": Counter(), "NEW": Counter(), "MID": Counter()}
    chain_counts = {"OLD": 0, "NEW": 0, "MID": 0}

    pkls = sorted(pickle_dir.glob("*.pkl"))
    print(f"# scanning {len(pkls)} pickles")

    bad = 0
    for p in pkls:
        d = dt.date.fromtimestamp(p.stat().st_mtime)
        if d <= cutoff_old:
            bucket = "OLD"
        elif d >= cutoff_new:
            bucket = "NEW"
        else:
            bucket = "MID"
        try:
            with open(p, "rb") as f:
                dag = pickle.load(f)
        except Exception as e:  # noqa: BLE001
            bad += 1
            continue
        chain_counts[bucket] += 1
        for n in iter_nodes(dag):
            flag = getattr(n, "flag", None)
            if flag is None:
                continue
            buckets[bucket][flag] += 1

    if bad:
        print(f"# WARNING: {bad} pickles failed to load")

    print()
    print(f"{'flag':<14} {'OLD':>10} {'OLD%':>8} {'MID':>10} {'NEW':>10} {'NEW%':>8}")
    print("-" * 64)
    tot_old = sum(buckets['OLD'].values()) or 1
    tot_mid = sum(buckets['MID'].values()) or 1
    tot_new = sum(buckets['NEW'].values()) or 1
    for f in FLAG_ORDER:
        o = buckets['OLD'][f]; m = buckets['MID'][f]; n = buckets['NEW'][f]
        print(f"{f.name:<14} {o:>10d} {100*o/tot_old:>7.2f}% {m:>10d} {n:>10d} {100*n/tot_new:>7.2f}%")
    print("-" * 64)
    print(f"{'TOTAL nodes':<14} {tot_old:>10d}          {tot_mid:>10d} {tot_new:>10d}")
    print(f"{'chains':<14} {chain_counts['OLD']:>10d}          {chain_counts['MID']:>10d} {chain_counts['NEW']:>10d}")

    # Chi-square on OLD vs NEW (6 x 2 contingency)
    table = [[buckets['OLD'][f], buckets['NEW'][f]] for f in FLAG_ORDER]
    # Drop zero rows to keep chi-square valid
    table_nz = [row for row in table if sum(row) > 0]
    chi2, p, dof, expected = chi2_contingency(table_nz)
    print()
    print(f"chi-square (OLD vs NEW): chi2={chi2:.3f}  dof={dof}  p={p:.4g}")

    # Pairwise effect sizes per flag (raw % delta)
    print()
    print(f"per-flag absolute % drift (NEW − OLD):")
    for f in FLAG_ORDER:
        d_old = 100*buckets['OLD'][f]/tot_old
        d_new = 100*buckets['NEW'][f]/tot_new
        print(f"  {f.name:<14}  {d_new - d_old:+.2f} pp   ({d_old:.2f} → {d_new:.2f})")


if __name__ == "__main__":
    main()

"""
Compare empirical per-dataset flag distributions against the paper's
heatmap tables in overleaf/latex/assets/FaithformBench/*.dat.

Paper buckets: AF-FAIL, TC-FAIL, REFUTED, UNKNOWN, PROVED (DECLARATIVE excluded).
Paper datasets (y rows): OmniMATH, OlympiadBench, MATH, GSM8K.
Counts are per-node, rows sum to 100%.

Usage: python scripts/paper_comparison.py <formalizer> <perturbed|unperturbed>
"""

from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scipy.stats import chi2_contingency  # noqa: E402
from veriform.preprocessing.dag import Flagging  # noqa: E402


REPO = Path(__file__).resolve().parent.parent
PAPER_DAT = REPO / "overleaf" / "latex" / "assets" / "FaithformBench"

PAPER_BUCKETS = ["AF_FAIL", "TC_FAIL", "REFUTED", "UNKNOWN", "PROVED"]
FLAG_BY_NAME = {f.name: f for f in Flagging}
PAPER_DATASETS = ["omnimath", "olympiadbench", "math", "gsm8k"]  # y=0..3

DATASET_PREFIX_MAP = {
    "gsm8k": "gsm8k",
    "math": "math",
    "olympiadbench": "olympiadbench",
    "omnimath": "omnimath",
}


def chain_dataset(stem: str) -> str | None:
    prefix = stem.split("-", 1)[0].lower()
    return DATASET_PREFIX_MAP.get(prefix)


def iter_nodes(dag):
    nodes = getattr(dag, "nodes", None)
    if nodes is None and hasattr(dag, "dag"):
        nodes = getattr(dag.dag, "nodes", None)
    if nodes is None:
        return
    for n in nodes:
        yield n


def load_paper_dat(path: Path) -> dict[str, dict[str, int]]:
    """Returns {dataset_name: {flag_name: count}}."""
    out: dict[str, dict[str, int]] = {d: {b: 0 for b in PAPER_BUCKETS} for d in PAPER_DATASETS}
    with open(path) as f:
        header = f.readline().split()
        for line in f:
            parts = line.split()
            if not parts:
                continue
            x = int(parts[0]); y = int(parts[1])
            val = int(parts[2])
            ds = PAPER_DATASETS[y]
            bucket = PAPER_BUCKETS[x]
            out[ds][bucket] = val
    return out


def tally_empirical(pickle_dir: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {d: {b: 0 for b in PAPER_BUCKETS} for d in PAPER_DATASETS}
    bad = 0
    pkls = sorted(pickle_dir.glob("*.pkl"))
    print(f"# scanning {len(pkls)} pickles in {pickle_dir}")
    for p in pkls:
        ds = chain_dataset(p.stem)
        if ds is None:
            continue
        try:
            with open(p, "rb") as f:
                dag = pickle.load(f)
        except Exception:
            bad += 1
            continue
        for n in iter_nodes(dag):
            flag = getattr(n, "flag", None)
            if flag is None:
                continue
            if flag.name == "DECLARATIVE":  # paper excludes these
                continue
            if flag.name in PAPER_BUCKETS:
                out[ds][flag.name] += 1
    if bad:
        print(f"# WARNING: {bad} pickles failed to load")
    return out


def fmt_row(label: str, counts: dict[str, int]) -> str:
    tot = sum(counts.values()) or 1
    cells = [f"{counts[b]:>5d} ({100*counts[b]/tot:>4.1f}%)" for b in PAPER_BUCKETS]
    return f"{label:<16} | " + " | ".join(cells) + f" |  tot={tot:>5d}"


def main():
    formalizer = sys.argv[1]
    cond = sys.argv[2]  # perturbed|unperturbed
    p_pct = "100" if cond == "perturbed" else "0"
    method = sys.argv[3] if len(sys.argv) > 3 else "regex"  # regex by default

    paper_path = PAPER_DAT / f"{formalizer}_{cond}.dat"
    pkl_dir = REPO / "data" / "output" / f"{method}-perturbed" / "proved" / formalizer / p_pct

    print(f"# formalizer={formalizer}  cond={cond}  method={method}")
    print(f"# paper:     {paper_path}")
    print(f"# empirical: {pkl_dir}")
    print()

    paper = load_paper_dat(paper_path)
    empir = tally_empirical(pkl_dir)

    header = f"{'dataset/source':<16} | " + " | ".join(f"{b:>11s}" for b in PAPER_BUCKETS) + " |  total"
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    pooled_paper = {b: 0 for b in PAPER_BUCKETS}
    pooled_emp = {b: 0 for b in PAPER_BUCKETS}

    for ds in PAPER_DATASETS:
        print(fmt_row(f"PAPER  {ds}", paper[ds]))
        print(fmt_row(f"EMPIR  {ds}", empir[ds]))
        # per-dataset chi-square (5 buckets, drop zero-rows for stability)
        rows = []
        for b in PAPER_BUCKETS:
            row = [paper[ds][b], empir[ds][b]]
            if sum(row) > 0:
                rows.append(row)
        if len(rows) >= 2 and all(sum(r) > 0 for r in rows):
            chi2, pval, dof, _ = chi2_contingency(rows)
            print(f"{'':<16}   chi2={chi2:>7.2f}  dof={dof}  p={pval:.3g}")
        for b in PAPER_BUCKETS:
            pooled_paper[b] += paper[ds][b]
            pooled_emp[b] += empir[ds][b]
        print()

    print("-" * len(header))
    print(fmt_row("PAPER  POOLED", pooled_paper))
    print(fmt_row("EMPIR  POOLED", pooled_emp))
    rows = [[pooled_paper[b], pooled_emp[b]] for b in PAPER_BUCKETS if pooled_paper[b]+pooled_emp[b] > 0]
    chi2, pval, dof, _ = chi2_contingency(rows)
    print(f"{'':<16}   chi2={chi2:.3f}  dof={dof}  p={pval:.4g}")

    # Per-flag absolute % drift (POOLED)
    tot_paper = sum(pooled_paper.values()) or 1
    tot_emp = sum(pooled_emp.values()) or 1
    print()
    print("per-flag absolute pp drift (EMPIR − PAPER, pooled):")
    for b in PAPER_BUCKETS:
        d_paper = 100 * pooled_paper[b] / tot_paper
        d_emp = 100 * pooled_emp[b] / tot_emp
        print(f"  {b:<10}  {d_emp - d_paper:+6.2f} pp   ({d_paper:5.2f} → {d_emp:5.2f})")


if __name__ == "__main__":
    main()

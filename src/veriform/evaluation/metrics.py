"""Per-node statistics, confusion matrix, and per-dataset heatmap counters.

Lifted from scripts/legacy/v2_exp.py and scripts/legacy/all_proof.py. The two
legacy paths used near-identical logic with one operating on `DAGNode` objects
(in-memory) and the other on dict-shaped JSON nodes; both modes are supported
here via the small `_flag_value` adapter.
"""

from typing import Any, Dict, List, Union

import numpy as np

from veriform.preprocessing.dag import Flagging

DATASET_ORDER: List[str] = ["GSM8K", "MATH", "OlympiadBench", "OmniMATH"]
FLAG_COLUMNS: List[str] = ["AF-FAIL", "TC-FAIL", "REFUTED", "UNKNOWN", "PROVED", "TOTAL"]
DATASET_TO_ID: Dict[str, int] = {name.lower(): i for i, name in enumerate(DATASET_ORDER)}

_FLAG_TO_COL = {
    Flagging.AF_FAIL.value: 0,
    Flagging.TC_FAIL.value: 1,
    Flagging.REFUTED.value: 2,
    Flagging.UNKNOWN.value: 3,
    Flagging.PROVED.value: 4,
}


def _flag_value(node: Any) -> str:
    """Accept either a DAGNode (with `.flag.value`) or a dict node (`node['flag']`)."""
    if isinstance(node, dict):
        return node["flag"]
    return node.flag.value


def _is_perturbed(node: Any) -> bool:
    if isinstance(node, dict):
        return node["is_perturbed"]
    return node.is_perturbed


def init_statistics() -> Dict[str, Any]:
    counts = {
        key: 0
        for key in [
            "total",
            "af_failed_count",
            "tc_failed_count",
            "refuted_count",
            "proved_count",
            "declarative_count",
            "unknown_count",
            "tp_count",
            "fp_count",
            "tn_count",
            "fn_count",
        ]
    }
    counts.update({"f1_score": 0.0, "accuracy": 0.0, "recall": 0.0, "precision": 0.0})
    return counts


def init_heatmap_data() -> Dict[str, Any]:
    return {
        "columns": FLAG_COLUMNS,
        "rows": DATASET_ORDER,
        "data": np.zeros((len(DATASET_ORDER), len(FLAG_COLUMNS))).tolist(),
    }


def update_statistics(stats: Dict[str, Any], node: Any) -> None:
    """Accumulate one node's flag into the confusion matrix counters."""
    flag = _flag_value(node)

    if flag == Flagging.DECLARATIVE.value:
        stats["declarative_count"] += 1
        return

    stats["total"] += 1

    if flag == Flagging.AF_FAIL.value:
        stats["af_failed_count"] += 1
    elif flag == Flagging.TC_FAIL.value:
        stats["tc_failed_count"] += 1
    elif flag == Flagging.REFUTED.value:
        stats["refuted_count"] += 1
    elif flag == Flagging.UNKNOWN.value:
        stats["unknown_count"] += 1
    elif flag == Flagging.PROVED.value:
        stats["proved_count"] += 1

    is_positive_flag = flag != Flagging.PROVED.value
    if _is_perturbed(node):
        if is_positive_flag:
            stats["tp_count"] += 1
        else:
            stats["fn_count"] += 1
    else:
        if is_positive_flag:
            stats["fp_count"] += 1
        else:
            stats["tn_count"] += 1


def update_heatmap_data(heatmap: Dict[str, Any], chain_id: str, nodes: List[Any]) -> None:
    """Increment per-dataset, per-flag counts for one chain's nodes."""
    difficulty_key = chain_id.split("-")[0].lower()
    row_idx = DATASET_TO_ID[difficulty_key]

    for node in nodes:
        flag = _flag_value(node)
        if flag == Flagging.DECLARATIVE.value:
            continue
        col_idx = _FLAG_TO_COL.get(flag, -1)
        if col_idx != -1:
            heatmap["data"][row_idx][col_idx] += 1
            heatmap["data"][row_idx][5] += 1


def calculate_metrics(stats: Dict[str, Any]) -> None:
    """Recalculate derived metrics in place."""
    tp = stats["tp_count"]
    fp = stats["fp_count"]
    fn = stats["fn_count"]
    tn = stats["tn_count"]
    total = stats["total"]

    stats["accuracy"] = (tp + tn) / max(1, total)
    stats["precision"] = tp / max(1, tp + fp)
    stats["recall"] = tp / max(1, tp + fn)
    stats["f1_score"] = (
        2 * (stats["precision"] * stats["recall"])
        / max(1e-8, (stats["precision"] + stats["recall"]))
    )

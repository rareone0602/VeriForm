"""LLM-based perturbation following the BrokenMath prompt format.

This is a *lookup* perturber: the actual GPT-5.2 batch call is run offline
(re-adapted BrokenMath prompt) and the per-step results are persisted to a
JSONL whose path is passed at construction time. At ``perturb()`` time we
just join the JSONL records onto DAG nodes by ``(chain_id, node_id)``.

JSONL row shape (OpenAI batch API output):
    {
      "custom_id":  "<chain_id>_step_<N>"   # 1-indexed step
      "response": {
        "body": {
          "choices": [
            {"message": {"role": "assistant", "content": "<perturbed step>"}},
            ...
          ]
        }
      },
      ...
    }

`is_perturbed` is True only when the BrokenMath output differs from the
node's original content (after stripping). No-op rows (model returned the
original) leave ``is_perturbed=False`` so they don't pollute the
perturbed arm of the TP/FP/FN/TN matrix in evaluation/metrics.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from veriform.preprocessing.dag import DAGModel, Flagging


class BrokenMathPerturber:
    """Lookup perturber over a pre-computed BrokenMath JSONL."""

    def __init__(
        self,
        jsonl_path: str | Path = "brokenmath_perturbations/new_perturbations_gpt5.2_medium.jsonl",
        # Accepted for registry-uniformity with StandardPerturber/__init__ signature;
        # ignored — BrokenMath is full-coverage by construction.
        p: float = 1.0,
        **kwargs,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self._lookup: Dict[Tuple[str, int], str] = {}
        with open(self.jsonl_path) as f:
            for line in f:
                row = json.loads(line)
                cid = row["custom_id"]
                # custom_id format: "<chain_id>_step_<N>" with N 1-indexed.
                # Use rsplit so chain_ids that themselves contain '_step_' (none today,
                # but be safe) don't get mis-split on the wrong delimiter.
                chain_id, step = cid.rsplit("_step_", 1)
                node_id = int(step) - 1
                # Defensive: a few batch rows may be malformed (refusal, missing
                # choices). In those cases we leave the node un-mapped.
                try:
                    content = row["response"]["body"]["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    continue
                if not isinstance(content, str):
                    continue
                self._lookup[(chain_id, node_id)] = content

    def perturb(self, dag: DAGModel) -> DAGModel:
        for node in dag.nodes:
            if node.flag == Flagging.DECLARATIVE:
                # DECLARATIVE nodes are never formalized/proved; leave untouched.
                continue
            new_content = self._lookup.get((dag.id, node.node_id))
            if new_content is None:
                # No JSONL entry for this (chain, node). Pass through untouched.
                node.perturbed_content = node.content
                node.is_perturbed = False
                continue
            node.perturbed_content = new_content
            # Mark perturbed only when the text actually changed. Honest semantics
            # for the TP/FP/FN/TN matrix downstream — no-op rows shouldn't count
            # as positives in the perturbed arm.
            node.is_perturbed = (new_content.strip() != (node.content or "").strip())
        return dag

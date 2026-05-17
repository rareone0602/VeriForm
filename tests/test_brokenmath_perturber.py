"""Pinning tests for the BrokenMath lookup perturber.

Confirms:
  - JSONL rows are joined onto DAG nodes by (chain_id, node_id) where
    custom_id "<chain_id>_step_<N>" maps to node_id = N - 1 (1-indexed
    in JSONL, 0-indexed in DAG).
  - is_perturbed is True iff the BrokenMath output differs from
    node.content; identical-content rows leave is_perturbed=False.
  - DECLARATIVE nodes are passed through unchanged.
  - Nodes without a matching JSONL entry are left untouched.

Tests use the real JSONL on disk (cheap to load — ~12k rows, no model)
to catch any silent regression in the file format / parser.
"""

import unittest
from pathlib import Path

from veriform.data.reasoning_chain import ReasoningChain, ReasoningStep, StepType
from veriform.perturbation.brokenmath_perturber import BrokenMathPerturber
from veriform.preprocessing.dag import DAGModel, Flagging


JSONL_PATH = Path(__file__).resolve().parent.parent / "brokenmath_perturbations" / "new_perturbations_gpt5.2_medium.jsonl"


def _make_chain(chain_id: str, contents):
    """Build a synthetic ReasoningChain with N steps, then a DAG over it."""
    steps = [
        ReasoningStep(step_id=i, content=c, step_type=StepType.CALCULATION)
        for i, c in enumerate(contents)
    ]
    chain = ReasoningChain(chain_id=chain_id, problem_statement="(test)", steps=steps)
    return DAGModel(chain)


@unittest.skipUnless(JSONL_PATH.exists(), f"JSONL not present at {JSONL_PATH}")
class TestBrokenMathPerturber(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.perturber = BrokenMathPerturber(jsonl_path=str(JSONL_PATH))

    def test_lookup_size_matches_corpus(self):
        # 12,784 formalizable steps in the DAG corpus; one perturbation per step.
        # If the JSONL is regenerated with a different shape this guard fires.
        self.assertEqual(len(self.perturber._lookup), 12784)

    def test_known_chain_perturbations_land_on_right_nodes(self):
        # math-548 has 9 perturbations from rows 0..8 (custom_ids step_1..step_9).
        # We use the actual JSONL contents to drive the assertion so the test
        # stays in sync if the underlying data changes.
        cid = "math-548"
        rows = {nid: self.perturber._lookup[(cid, nid)]
                for (k, nid) in self.perturber._lookup if k == cid for _ in [None]}
        # That comprehension is awkward — rebuild straight:
        rows = {nid: txt for (k, nid), txt in self.perturber._lookup.items() if k == cid}
        self.assertGreaterEqual(len(rows), 1)

        # Build a synthetic DAG with original content per node — different from
        # the perturbed text, so is_perturbed should flip True after perturb.
        n = max(rows) + 1
        contents = [f"original step {i}" for i in range(n)]
        dag = _make_chain(cid, contents)
        dag = self.perturber.perturb(dag)

        for nid, expected_txt in rows.items():
            node = dag.nodes[nid]
            self.assertEqual(node.perturbed_content, expected_txt)
            self.assertTrue(node.is_perturbed,
                            f"node {nid} should be marked perturbed (text differs)")

    def test_noop_perturbation_leaves_is_perturbed_false(self):
        # Pick a real JSONL row and use its perturbed text as the original
        # content — perturber should detect no change and leave is_perturbed=False.
        (cid, nid), txt = next(iter(self.perturber._lookup.items()))
        contents = [""] * (nid + 1)
        contents[nid] = txt  # identical to BrokenMath output
        dag = _make_chain(cid, contents)
        dag = self.perturber.perturb(dag)
        node = dag.nodes[nid]
        self.assertEqual(node.perturbed_content, txt)
        self.assertFalse(node.is_perturbed,
                         "identical-text perturbation must not flip is_perturbed")

    def test_unmapped_chain_passes_through(self):
        # A chain_id that doesn't exist in the JSONL — nothing to apply,
        # everything stays untouched.
        dag = _make_chain("does-not-exist-9999", ["a", "b", "c"])
        dag = self.perturber.perturb(dag)
        for node in dag.nodes:
            self.assertEqual(node.perturbed_content, node.content)
            self.assertFalse(node.is_perturbed)

    def test_declarative_nodes_skipped(self):
        # DECLARATIVE nodes must not be touched even when a JSONL entry exists.
        cid = "math-548"  # known chain in JSONL
        contents = ["x", "y", "z"]
        dag = _make_chain(cid, contents)
        # Mark node 1 declarative.
        dag.nodes[1].flag = Flagging.DECLARATIVE
        # Pre-condition: node 1 has perturbed_content=None (DAG default).
        self.assertIsNone(dag.nodes[1].perturbed_content)
        dag = self.perturber.perturb(dag)
        # Even though math-548_step_2 exists in the JSONL, the declarative node
        # must remain untouched.
        self.assertIsNone(dag.nodes[1].perturbed_content)
        self.assertFalse(dag.nodes[1].is_perturbed)


if __name__ == "__main__":
    unittest.main()

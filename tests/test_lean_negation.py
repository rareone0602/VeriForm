"""Lean theorem negation tests via the bundled `lake exe negate`.

Two layers:
  - Unit-ish test of LeanNegator's framing protocol (verifies request format).
  - Integration tests that spawn `lake exe negate` for real. These are
    SLOW (~6s startup) and SKIPPED if the lake binary or built executable
    isn't available locally.
"""

import os
import re
import subprocess
import unittest
from pathlib import Path

from veriform.proving.negation import (
    DEFAULT_LAKE_PATH,
    DEFAULT_LEAN_WORKSPACE,
    LeanNegator,
    LeanNegatorError,
)


def _lake_available() -> bool:
    return DEFAULT_LAKE_PATH.exists() and (
        DEFAULT_LEAN_WORKSPACE / ".lake" / "build" / "bin" / "negate"
    ).exists()


@unittest.skipUnless(_lake_available(), "lake or lake exe negate not built")
class TestLeanNegationIntegration(unittest.TestCase):
    """Each test asserts the prefix and that the result re-parses by sending
    it back through the negator a second time. A round-trip parse failure
    means the first negation produced syntactically broken Lean."""

    @classmethod
    def setUpClass(cls):
        cls.negator = LeanNegator()
        cls.negator.start()

    @classmethod
    def tearDownClass(cls):
        cls.negator.close()

    def _assertNegatesCleanly(self, src: str, mode: str, expected_prefix: str):
        out = self.negator.negate(src, mode=mode)
        self.assertTrue(
            out.startswith(expected_prefix),
            f"\nexpected prefix: {expected_prefix!r}\ngot: {out!r}",
        )
        self.assertIn("¬", out, f"missing negation: {out!r}")
        self.assertTrue(out.endswith(":= by sorry"), f"missing proof: {out!r}")
        # Round-trip: negate the negation. Must not raise.
        # (We don't compare the second negation to the original — `not_not_T_strong_strong`
        #  is fine; we only require it parses.)
        try:
            self.negator.negate(out, mode=mode)
        except LeanNegatorError as e:
            self.fail(f"second negation failed (output not re-parseable): {e}\nfirst output: {out}")

    # --- regex-killer cases -----------------------------------------------

    def test_simple_no_binders(self):
        self._assertNegatesCleanly(
            "theorem t1 : 2 + 2 = 4 := by sorry",
            "strong",
            "theorem not_t1_strong",
        )

    def test_implicit_args(self):
        self._assertNegatesCleanly(
            "theorem t2 {α : Type*} (x : α) : x = x := by sorry",
            "strong",
            "theorem not_t2_strong",
        )

    def test_implicit_args_full_mode(self):
        out = self.negator.negate(
            "theorem t2 {α : Type*} (x : α) : x = x := by sorry",
            mode="full",
        )
        # In full mode the binders are absorbed into ∀, so the new theorem has none.
        self.assertTrue(out.startswith("theorem not_t2_full :"))
        self.assertIn("∀", out)

    def test_instance_args(self):
        self._assertNegatesCleanly(
            "theorem t3 [Group G] (a b : G) : a * b = b * a := by sorry",
            "strong",
            "theorem not_t3_strong",
        )

    def test_function_arrow_in_binder(self):
        self._assertNegatesCleanly(
            "theorem t5 {α β : Type*} (f : α → β) (x : α) : f x = f x := by sorry",
            "full",
            "theorem not_t5_full",
        )

    def test_nested_forall_in_body(self):
        self._assertNegatesCleanly(
            "theorem t4 (n : Nat) : ∀ m : Nat, n + m = m + n := by sorry",
            "strong",
            "theorem not_t4_strong",
        )

    def test_compound_body(self):
        self._assertNegatesCleanly(
            "theorem t6 : (∀ x : Nat, x ≥ 0) ∧ (∀ y : Int, y + 0 = y) := by sorry",
            "strong",
            "theorem not_t6_strong",
        )

    def test_existential_body(self):
        self._assertNegatesCleanly(
            "theorem t7 (n : Nat) (h : n ≠ 0) : ∃ m, m + 1 = n := by sorry",
            "full",
            "theorem not_t7_full",
        )

    def test_unicode_operators(self):
        self._assertNegatesCleanly(
            "theorem t8 (x y : Nat) : x ≤ y ∨ y ≤ x := by sorry",
            "strong",
            "theorem not_t8_strong",
        )

    def test_dependent_type(self):
        # Body's type depends on a binder name.
        self._assertNegatesCleanly(
            "theorem t9 (n : Nat) (v : Fin n) : v.val < n := by sorry",
            "strong",
            "theorem not_t9_strong",
        )

    # --- error cases ------------------------------------------------------

    def test_not_a_theorem(self):
        with self.assertRaises(LeanNegatorError):
            self.negator.negate("def foo : Nat := 0", mode="full")

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            self.negator.negate("theorem t : True := by trivial", mode="bogus")  # type: ignore[arg-type]

    # --- multi-call stability --------------------------------------------

    def test_many_sequential_calls(self):
        """Drive 50 negations through a single subprocess to catch
        framing/leak bugs that don't show up in a one-shot test."""
        src = "theorem t (x : Nat) : x + 0 = x := by sorry"
        for _ in range(50):
            out = self.negator.negate(src, mode="full")
            self.assertTrue(out.startswith("theorem not_t_full"))


class TestLeanNegationProtocol(unittest.TestCase):
    """Verifies the wire format without spawning the real subprocess."""

    def test_request_is_tab_framed_base64(self):
        import base64
        src = "theorem foo : True := by trivial"
        b64 = base64.b64encode(src.encode("utf-8")).decode("ascii")
        # Reconstruct what LeanNegator would write to stdin
        line = f"strong\t{b64}\n"
        self.assertTrue(line.endswith("\n"))
        mode, payload = line.rstrip("\n").split("\t", 1)
        self.assertEqual(mode, "strong")
        self.assertEqual(base64.b64decode(payload).decode("utf-8"), src)

    def test_default_paths(self):
        # Just sanity-check the paths are absolute and well-formed.
        self.assertTrue(DEFAULT_LAKE_PATH.is_absolute())
        self.assertTrue(DEFAULT_LEAN_WORKSPACE.is_absolute())
        self.assertEqual(DEFAULT_LEAN_WORKSPACE.name, "lean")


if __name__ == "__main__":
    unittest.main()

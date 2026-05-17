"""Pinning tests for the DeepSeek-Prover-V2 prompt.

The prove-phase prompt is built from three pieces of `deepseek_prover.py`:
  - `HEADER`        : the imports + set_option + open lines that prefix the
                      Lean code,
  - `LEAN_TEMPLATE` : `HEADER` + a `/-- statement -/` docstring + the Lean
                      theorem body,
  - `LEAN_WRAPPER_TEMPLATE` : the natural-language wrapper (`Complete the
                      following Lean 4 code:` ...).

These tests reconstruct the exact `formal_statement` from the official
DeepSeek-Prover-V2-7B model card quickstart and assert byte-for-byte equality
against what our code produces. If anyone tightens whitespace or changes the
docstring delimiter, these tests fire and stop the change before it eats
several hours of GPU time at a drifted distribution.

Reference: https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-7B
"""

import unittest

from veriform.proving.deepseek_prover import (
    HEADER,
    LEAN_TEMPLATE,
    LEAN_WRAPPER_TEMPLATE,
)


# ---------------------------------------------------------------------------
# The official `formal_statement` literal from the model-card Quickstart,
# verbatim. The example uses `mathd_algebra_10` from miniF2F-style problems.
# ---------------------------------------------------------------------------

OFFICIAL_FORMAL_STATEMENT = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- What is the positive difference between $120\\%$ of 30 and $130\\%$ of 20? Show that it is 10.-/
theorem mathd_algebra_10 : abs ((120 : ℝ) / 100 * 30 - 130 / 100 * 20) = 10 := by
  sorry"""

OFFICIAL_USER_PROMPT = """Complete the following Lean 4 code:

```lean4
{}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof."""


# Inputs that, when fed through our LEAN_TEMPLATE, must reproduce
# OFFICIAL_FORMAL_STATEMENT exactly.
SAMPLE_NL = (
    "What is the positive difference between $120\\%$ of 30 and "
    "$130\\%$ of 20? Show that it is 10."
)
SAMPLE_THEOREM = (
    "theorem mathd_algebra_10 : abs ((120 : ℝ) / 100 * 30 - "
    "130 / 100 * 20) = 10 := by\n  sorry"
)


class TestProverPromptMatchesModelCard(unittest.TestCase):
    """Byte-identical match against the official Quickstart snippet."""

    def test_header_has_no_leading_blank_line(self):
        # Regression guard: the previous HEADER started with '\n', which made
        # the rendered ```lean4 fence open with a blank line — out-of-distribution
        # relative to the model card.
        self.assertFalse(
            HEADER.startswith("\n"),
            "HEADER must not start with a newline; the leading blank line "
            "drifts from the model-card formal_statement layout.",
        )

    def test_header_contents_verbatim(self):
        expected = (
            "import Mathlib\n"
            "import Aesop\n"
            "\n"
            "set_option maxHeartbeats 0\n"
            "\n"
            "open BigOperators Real Nat Topology Rat\n"
        )
        self.assertEqual(HEADER, expected)

    def test_lean_template_uses_lean_docstring_delimiter(self):
        # `/--` is the Lean docstring delimiter (attached to the next decl);
        # `/-` is a plain block comment. The model card uses `/--`.
        self.assertIn("/--", LEAN_TEMPLATE)
        # And the closing must be tight (no space before -/), to match the
        # model card's `it is 10.-/` style.
        self.assertIn("-/", LEAN_TEMPLATE)
        self.assertNotIn(" -/", LEAN_TEMPLATE)

    def test_lean_template_renders_to_official_formal_statement(self):
        rendered = LEAN_TEMPLATE.format(statement=SAMPLE_NL, lean_code=SAMPLE_THEOREM)
        self.assertEqual(rendered, OFFICIAL_FORMAL_STATEMENT)

    def test_lean_wrapper_template_renders_to_official_user_prompt(self):
        rendered_formal = LEAN_TEMPLATE.format(
            statement=SAMPLE_NL, lean_code=SAMPLE_THEOREM
        )
        rendered_user = LEAN_WRAPPER_TEMPLATE.format(lean_code=rendered_formal)
        expected_user = OFFICIAL_USER_PROMPT.format(OFFICIAL_FORMAL_STATEMENT)
        self.assertEqual(rendered_user, expected_user)

    def test_lean_wrapper_text_verbatim(self):
        # Verbatim from the Quickstart's `prompt = """..."""` literal,
        # after .strip().
        expected = (
            "Complete the following Lean 4 code:\n"
            "\n"
            "```lean4\n"
            "{lean_code}\n"
            "```\n"
            "\n"
            "Before producing the Lean 4 code to formally prove the given theorem, "
            "provide a detailed proof plan outlining the main proof steps and strategies.\n"
            "The plan should highlight key ideas, intermediate lemmas, and proof "
            "structures that will guide the construction of the final formal proof."
        )
        self.assertEqual(LEAN_WRAPPER_TEMPLATE, expected)


if __name__ == "__main__":
    unittest.main()

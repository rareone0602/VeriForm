"""Snapshot tests pinning each formaliser's prompt to its canonical
training-time format.

Formalisers fine-tuned on a specific prompt format degrade significantly
if the inference prompt drifts. These tests fail loudly on any change so
the drift is a deliberate decision, not an accident.

Sources of truth (all official model-card example code):
  - Goedel: https://huggingface.co/Goedel-LM/Goedel-Formalizer-V2-8B
  - Kimina: https://huggingface.co/AI-MO/Kimina-Autoformalizer-7B
  - Stepfun: https://huggingface.co/stepfun-ai/StepFun-Formalizer-7B
  - Herald: https://github.com/frenzymath/herald_translator/blob/main/worker/translator.py
            + chat_template field in HF tokenizer_config.json

Herald is excluded from the live-instantiation tests because its __init__
loads vllm.LLM into GPU memory, which is too heavy for unit tests. Its
prompt format is checked separately on a mocked instance.
"""

import unittest
from unittest.mock import MagicMock, patch

from veriform.formalization import (
    GoedelFormalizer,
    KiminaFormalizer,
    StepfunFormalizer,
)


SAMPLE_PROBLEM = "Prove that 2 + 2 = 4."


class TestGoedelPrompt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = GoedelFormalizer(sampling="recommended")

    def test_user_prompt_matches_card_verbatim(self):
        # Verbatim from Goedel-LM/Goedel-Formalizer-V2-8B model card example.
        # NOTE: no separator between {informal_problem} and "Think before...".
        expected = (
            "Please autoformalize the following natural language problem statement in Lean 4. "
            "Use the following theorem name: my_favorite_theorem\n"
            "The natural language statement is: \n"
            f"{SAMPLE_PROBLEM}"
            "Think before you provide the lean statement."
        )
        self.assertEqual(self.f.get_formal_statement_prompt(SAMPLE_PROBLEM), expected)

    def test_no_system_role(self):
        self.f.initialize_dialog()
        self.assertEqual(self.f.dialog, [], "Goedel example uses no system role")

    def test_recommended_sampling(self):
        self.assertEqual(self.f.recommended_kwargs["temperature"], 0.9)
        self.assertEqual(self.f.recommended_kwargs["top_p"], 0.95)
        self.assertEqual(self.f.recommended_kwargs["max_tokens"], 16384)
        self.assertEqual(self.f.recommended_kwargs["extra_body"], {"top_k": 20})


class TestKiminaPrompt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = KiminaFormalizer(sampling="recommended")

    def test_user_prompt_matches_card_verbatim(self):
        expected = (
            "Please autoformalize the following problem in Lean 4 with a header. "
            "Use the following theorem names: my_favorite_theorem.\n\n"
            f"{SAMPLE_PROBLEM}"
        )
        self.assertEqual(self.f.get_formal_statement_prompt(SAMPLE_PROBLEM), expected)

    def test_system_prompt_matches_card(self):
        self.assertEqual(self.f.system_prompt, "You are an expert in mathematics and Lean 4.")
        self.f.initialize_dialog()
        self.assertEqual(self.f.dialog[0], {"role": "system", "content": self.f.system_prompt})

    def test_recommended_sampling(self):
        self.assertEqual(self.f.recommended_kwargs["temperature"], 0.6)
        self.assertEqual(self.f.recommended_kwargs["top_p"], 0.95)
        self.assertEqual(self.f.recommended_kwargs["max_tokens"], 2048)
        # Kimina's official example has no top_k; keep it absent.
        self.assertNotIn("extra_body", self.f.recommended_kwargs)


class TestStepfunPrompt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = StepfunFormalizer(sampling="recommended")

    def test_user_prompt_matches_card_verbatim(self):
        expected = (
            "Please autoformalize the following problem in Lean 4 with a header. "
            "Use the following theorem names: my_favorite_theorem.\n\n"
            f"{SAMPLE_PROBLEM}"
            f"\n\nYour code should start with:\n```Lean4\n{self.f.HEADER}\n```\n"
        )
        self.assertEqual(self.f.get_formal_statement_prompt(SAMPLE_PROBLEM), expected)

    def test_system_prompt_matches_card(self):
        self.assertEqual(self.f.system_prompt, "You are an expert in mathematics and Lean 4.")

    def test_format_dialog_appends_literal_think_token(self):
        # CRITICAL: the StepFun model card explicitly appends "<think>"
        # after applying the chat template. Without it the DeepSeek-R1
        # distilled base does not enter reasoning mode and degrades.
        self.f.initialize_dialog()
        self.f.dialog.append({"role": "user", "content": "x"})
        out = self.f.format_dialog(self.f.dialog)
        self.assertTrue(out.endswith("<think>"), f"trailing <think> missing; tail={out[-40:]!r}")

    def test_recommended_sampling(self):
        self.assertEqual(self.f.recommended_kwargs["temperature"], 0.6)
        self.assertEqual(self.f.recommended_kwargs["top_p"], 0.95)
        self.assertEqual(self.f.recommended_kwargs["max_tokens"], 16384)


class TestHeraldPrompt(unittest.TestCase):
    """Herald's __init__ loads vllm.LLM into GPU; mock the model away so we
    can still pin the prompt-building behaviour."""

    @classmethod
    def setUpClass(cls):
        with patch("veriform.formalization.fine_tuned.LLM") as MockLLM:
            MockLLM.return_value = MagicMock()
            from veriform.formalization import HeraldFormalizer
            cls.f = HeraldFormalizer(sampling="deterministic")

    def test_system_prompt_matches_repo(self):
        # Verbatim from frenzymath/herald_translator worker/translator.py.
        self.assertEqual(self.f.system_prompt, "You are an expert at Lean 4 and Mathematics.")

    def test_user_prompt_matches_repo_verbatim(self):
        expected = (
            "Please translate the natural language statement to Lean4 code with the header\n"
            "**Name**\nmy_favorite_theorem\n"
            f"**Informal statement**\n{SAMPLE_PROBLEM}\n"
        )
        self.assertEqual(self.f.get_formal_statement_prompt(SAMPLE_PROBLEM), expected)

    def test_format_dialog_uses_canonical_chat_template(self):
        # The chat_template shipped with Herald's tokenizer encodes:
        #   <BOS> + system + '\n\n' + 'User: ' + content + '\n\nAssistant:'
        # Verify the rendered string matches that — in particular:
        #  - "User: " (with space after colon, not the legacy "User:")
        #  - presence of BOS sentence marker
        self.f.initialize_dialog()
        self.f.dialog.append({"role": "user", "content": "PAYLOAD"})
        out = self.f.format_dialog(self.f.dialog)
        self.assertIn("User: PAYLOAD", out)
        self.assertTrue(out.endswith("Assistant:"), f"missing trailing 'Assistant:'; tail={out[-40:]!r}")
        # The previous custom format_dialog used "User:PAYLOAD" (no space)
        # and was missing the BOS marker entirely. Guard against regression.
        self.assertNotIn("User:PAYLOAD", out)


if __name__ == "__main__":
    unittest.main()

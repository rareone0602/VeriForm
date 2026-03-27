import re
import unittest
from typing import Optional, Tuple
from veriform.autoformalization_v2.theorem_extractor import TheoremExtractor

# ==========================================
# TEST SUITE
# ==========================================

class TestTheoremExtraction(unittest.TestCase):
    def setUp(self):
        self.extractor = TheoremExtractor()

    def test_basic_theorem(self):
        # The user's specific failing case (without params)
        text = "theorem step_0 : 256 = 2^8 := by\n  have h : Nat := sorry"
        name, params, body = self.extractor.get_last_theorem(text)
        self.assertEqual(name, "step_0")
        self.assertIsNone(params)
        self.assertEqual(body, "256 = 2^8")

    def test_params_with_colons(self):
        # Tricky case: Colon inside parenthesis
        text = "theorem algebra (x : ℝ) : x + 0 = x := by sorry"
        name, params, body = self.extractor.get_last_theorem(text)
        self.assertEqual(name, "algebra")
        self.assertEqual(params, "(x : ℝ)")
        self.assertEqual(body, "x + 0 = x")

    def test_body_with_colons_and_quantifiers(self):
        # Tricky case: Body has colons, no params
        text = "theorem quant : ∀ x : Nat, x = x := by sorry"
        name, params, body = self.extractor.get_last_theorem(text)
        self.assertEqual(name, "quant")
        self.assertIsNone(params)
        self.assertEqual(body, "∀ x : Nat, x = x")

    def test_multiple_theorems(self):
        # Ensure we get the LAST one
        text = """
        theorem t1 : 1=1 := rfl
        theorem t2 (a : Nat) : a = a := rfl
        """
        name, params, body = self.extractor.get_last_theorem(text)
        self.assertEqual(name, "t2")
        self.assertEqual(params, "(a : Nat)")

    def test_complex_proof_block(self):
        # Ensure regex doesn't eat the proof if it contains :=
        text = """
        theorem tough_one (n : Nat) : n > 0 := by
          let x := 10
          have y : Nat := 5
          sorry
        """
        name, params, body = self.extractor.get_last_theorem(text)
        self.assertEqual(name, "tough_one")
        self.assertEqual(params, "(n : Nat)")
        self.assertEqual(body, "n > 0")
    
    def test_tautology(self):
        text = "theorem step_0 : 256 = 2^8 := by\n  sorry"
        name, params, body = self.extractor.get_last_theorem(text)
        self.assertEqual(name, "step_0")
        self.assertIsNone(params)
        self.assertEqual(body, "256 = 2^8")
    
    def test_let(self):
        text = "theorem my_favorite_thetical : \n  let monday_rain : \u211a := 2\n  let twice_monday_rain : \u211a := 2 * monday_rain\n  let tuesday_rain : \u211a := twice_monday_rain + 1\n  tuesday_rain = 4 / 2 := by sorry"
        name, params, body = self.extractor.get_last_theorem(text)
        self.assertEqual(name, "my_favorite_thetical")
        self.assertIsNone(params)
        self.assertEqual(body, "let monday_rain : \u211a := 2\n  let twice_monday_rain : \u211a := 2 * monday_rain\n  let tuesday_rain : \u211a := twice_monday_rain + 1\n  tuesday_rain = 4 / 2")
    
    def test_comment(self):
        # We need to preserve newlines for the 'let' bindings to be valid
        text = """
theorem my_favorite_theorem :  
  ∃ (x y : ℕ),  
    x ≤ 9 ∧ y ≤ 9 ∧  -- x and y are digits
    let birth_year := 1900 + 10 * x + y
    let age_in_2014 := 2014 - birth_year
    let rightmost_two_digits := 10 * x + y
    age_in_2014 = 2 * rightmost_two_digits ∧  -- age is twice the rightmost two digits
    birth_year = 1938 ∧  -- the solution
    age_in_2014 = 76 ∧  -- verification: 2014 - 1938 = 76
    rightmost_two_digits = 38  -- verification: 10*3 + 8 = 38
  := by sorry
"""     
        name, params, body = self.extractor.get_last_theorem(text)
        self.assertEqual(name, "my_favorite_theorem")
        self.assertIsNone(params)
        
        # The expected body must exactly match the cleaned multi-line string.
        # Note: The comments are gone, but the empty lines/newlines remain.
        expected_body = """∃ (x y : ℕ),  
    x ≤ 9 ∧ y ≤ 9 ∧  
    let birth_year := 1900 + 10 * x + y
    let age_in_2014 := 2014 - birth_year
    let rightmost_two_digits := 10 * x + y
    age_in_2014 = 2 * rightmost_two_digits ∧  
    birth_year = 1938 ∧  
    age_in_2014 = 76 ∧  
    rightmost_two_digits = 38"""
        
        # Compare stripped versions to ignore leading/trailing whitespace of the block
        self.assertEqual(body.strip(), expected_body.strip())
    

    def test_comments_and_strings_masks(self):
        # Ensure we don't trigger on keywords inside strings
        text = 'theorem tricky : String := "This theorem : is not real := by sorry" := rfl'
        name, params, body = self.extractor.get_last_theorem(text)
        self.assertEqual(name, "tricky")
        # Params should be None (colon detected correctly)
        self.assertIsNone(params)
        # Body should be 'String'
        self.assertEqual(body, "String")
        # If it parsed the string as code, it would have stopped at the first := inside the quotes


    def test_comment_removal(self):
        """Test that the cleaner removes comments but keeps structure/strings."""
        text = 'theorem t : (a : Nat) -- comment\n : a = a /- block -/ := rfl'
        name, params, body = self.extractor.get_last_theorem(text)
        self.assertEqual(name, 't')
        self.assertEqual(params, '(a : Nat)')
        # Internal spaces might be wider due to replacement, stripping handles ends
        self.assertTrue('block' not in body) 
        self.assertEqual(body.strip(), 'a = a')

    def test_nested_let_in_signature_and_body(self):
        text = """
theorem my_favorite_thetical (S U P E R B : ℝ × ℝ) 
  (h_regular : -- The hexagon is regular
    dist S U = dist U P ∧ 
    let angle := fun (A B C : ℝ × ℝ) => 
      let BA := A - B
      let BC := C - B
      Real.arccos ((BA.1 * BC.1) / (dist A B))
    angle S U P = 2 * Real.pi / 3)
  : -- Conclusion: angle SEB = 120°
    let angle := fun (A B C : ℝ × ℝ) => 
      let BA := A - B
      let BC := C - B
      Real.arccos ((BA.1 * BC.1) / (dist A B))
    angle S E B = 2 * Real.pi / 3 := by sorry
"""
        name, params, body = self.extractor.get_last_theorem(text)
        self.assertEqual(name, "my_favorite_thetical")
        self.assertIn("h_regular", params)
        self.assertNotIn("The hexagon is regular", params) # Comment check
        self.assertIn("angle S E B =", body)
        self.assertNotIn("Conclusion:", body) # Comment check

    def test_nested_comments_balanced(self):
        """
        Modified test case: Ensure the nested comments are actually balanced.
        Original failing string: /- comment /- deep /-*/ -/
        Fixed string: /- comment /- deep -/ -/
        """
        text = """
        theorem nested /- comment /- deep -/ -/ : 1 = 1 := rfl
        """
        name, params, body = self.extractor.get_last_theorem(text)
        self.assertEqual(name, "nested")
        self.assertEqual(body, "1 = 1")

    def test_string_masking(self):
        # Ensure keywords inside strings are ignored
        text = 'theorem tricky : String := "This theorem : is not real := by sorry" := rfl'
        name, params, body = self.extractor.get_last_theorem(text)
        self.assertEqual(name, "tricky")
        self.assertIsNone(params)
        self.assertEqual(body, "String")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

# python -m unittest tests.test_theorem_extractor.TestTheoremExtraction
"""
Lean code verification utilities.
"""

import subprocess
import tempfile
import os
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

from click import prompt

from veriform.autoformalization.prompts import create_deepseek_prover_prompt


@dataclass
class VerificationResult:
    """Result of verifying Lean code."""

    success: bool
    error_message: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    has_sorry: bool = False
    has_admit: bool = False

    @property
    def is_provable(self) -> bool:
        """Check if the code is provable (no sorry/admit and no errors)."""
        return self.success and not self.has_sorry and not self.has_admit

@dataclass
class LeanCodeVerificationResult:
    """Result of verifying multiple Lean codes."""

    goals: List[str]
    complete: bool
    sorries: int
    errors: List

class LeanVerifierOld:
    """Verifier for Lean 4 code."""

    def __init__(self, lean_executable: str = "lean", timeout: int = 30):
        """
        Initialize the Lean verifier.

        Args:
            lean_executable: Path to Lean executable
            timeout: Timeout in seconds for verification
        """
        self.lean_executable = lean_executable
        self.timeout = timeout

    def verify(self, lean_code: str) -> VerificationResult:
        """
        Verify Lean code.

        Args:
            lean_code: The Lean code to verify

        Returns:
            VerificationResult indicating success or failure
        """
        # Check for sorry and admit
        has_sorry = "sorry" in lean_code
        has_admit = "admit" in lean_code

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.lean',
            delete=False
        ) as f:
            f.write(lean_code)
            temp_file = f.name

        try:
            # Run Lean on the file
            result = subprocess.run(
                [self.lean_executable, temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            success = result.returncode == 0

            return VerificationResult(
                success=success,
                error_message=None if success else "Lean verification failed",
                stdout=result.stdout,
                stderr=result.stderr,
                has_sorry=has_sorry,
                has_admit=has_admit
            )

        except subprocess.TimeoutExpired:
            return VerificationResult(
                success=False,
                error_message=f"Verification timeout after {self.timeout}s",
                has_sorry=has_sorry,
                has_admit=has_admit
            )
        except FileNotFoundError:
            return VerificationResult(
                success=False,
                error_message=f"Lean executable not found: {self.lean_executable}",
                has_sorry=has_sorry,
                has_admit=has_admit
            )
        except Exception as e:
            return VerificationResult(
                success=False,
                error_message=f"Verification error: {str(e)}",
                has_sorry=has_sorry,
                has_admit=has_admit
            )
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except:
                pass

    def verify_file(self, file_path: str) -> VerificationResult:
        """
        Verify a Lean file.

        Args:
            file_path: Path to the Lean file

        Returns:
            VerificationResult
        """
        with open(file_path, 'r') as f:
            lean_code = f.read()

        return self.verify(lean_code)



class LeanVerifier:
    def __init__(self, lean_executable: str = "lean", timeout: int = 30):
        """
        Initialize the Lean verifier.

        Args:
            lean_executable: Path to Lean executable
            timeout: Timeout in seconds for verification
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .deepseek.prover.lean.verifier import Lean4ServerScheduler
        import torch
        from .templates import LEAN_WRAPPER_TEMPLATE

        self.lean_executable = lean_executable
        self.timeout = timeout
        self.batch_size = 4
        prover_model_id = "deepseek-ai/DeepSeek-Prover-V2-7B"  # or DeepSeek-Prover-V2-671B
        self.tokenizer = AutoTokenizer.from_pretrained(prover_model_id)
        self.prover = AutoModelForCausalLM.from_pretrained(prover_model_id, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True)
        self.prover.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.scheduler = Lean4ServerScheduler(
            max_concurrent_requests=self.batch_size,
        )

    def __del__(self):
        self.scheduler.close()

    def prove_batch(self, lean_code: str) -> List[str]:
        """
        Prove Lean code.

        Args:
            lean_code: The Lean code to prove

        Returns:
            VerificationResult indicating success or failure
        """
        import torch
        from .prompts import parse_deepseek_prover_response, create_deepseek_prover_prompt
        from .templates import LEAN_WRAPPER_TEMPLATE

        chat = [
            {"role": "user", "content": create_deepseek_prover_prompt(lean_code)},
        ]

        inputs = self.tokenizer.apply_chat_template([chat] * self.batch_size, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.prover.device)
        attention_mask = torch.ones_like(inputs)
        outputs = self.prover.generate(inputs, max_new_tokens=8192, attention_mask=attention_mask, temperature=0.7)
        batch_decoded = self.tokenizer.batch_decode(outputs)
        lean_codes = []
        for response in batch_decoded:
            lean_code = parse_deepseek_prover_response(response)
            lean_code_wrapped = LEAN_WRAPPER_TEMPLATE.format(lean_code=lean_code, statement="")
            lean_codes.append(lean_code_wrapped)
        return lean_codes
    
    def get_code_verification_result(self, lean_codes: List[str]) -> List[LeanCodeVerificationResult]:
        results = self.scheduler.submit_all_request([
            dict(code=lean_code, ast=False, tactics=False)
            for lean_code in lean_codes
        ])
        verification_results = self.scheduler.get_all_request_outputs(results)
        code_verification_results = []
        for verification_result in verification_results:
            goals = [
                sorries['goal'] for sorries in verification_result['sorries']
            ]
            complete = verification_result['complete']
            sorries = len(verification_result['sorries'])
            errors = verification_result['errors']
            code_verification_results.append(LeanCodeVerificationResult(
                goals=goals,
                complete=complete,
                sorries=sorries,
                errors=errors
            ))
        return code_verification_results

    def verify(self, lean_code: str) -> VerificationResult:
        """
        Verify Lean code.

        Args:
            lean_code: The Lean code to verify

        Returns:
            VerificationResult indicating success or failure
        """
        from .templates import LEAN_WRAPPER_TEMPLATE
        lean_code_wrapped = LEAN_WRAPPER_TEMPLATE.format(lean_code=lean_code, statement="")
        
        # Submit to Lean Interpreter
        verification_result = self.get_code_verification_result([lean_code_wrapped])[0]

        if len(verification_result.errors) > 0:
            return VerificationResult(
                success=False,
                error_message="; ".join(verification_result.errors),
                stdout="",
                stderr="; ".join(verification_result.errors),
                has_sorry="sorry" in lean_code,
                has_admit="admit" in lean_code
            )

        if verification_result.sorries == 0:
            return VerificationResult(
                success=True,
                error_message=None,
                stdout="",
                stderr="",
                has_sorry=False,
                has_admit=False
            )
        else:
            lean_codes = self.prove_batch(lean_code_wrapped)
            proven_results = self.get_code_verification_result(lean_codes)
            # Check if any proof attempt complete
            for verification_result in proven_results:
                if verification_result.complete:
                    return VerificationResult(
                        success=True,
                        error_message=None,
                        stdout="",
                        stderr="",
                        has_sorry=False,
                        has_admit=False
                    )
            
            return VerificationResult(
                success=False,
                error_message="; ".join(verification_result.errors),
                stdout="",
                stderr="; ".join(verification_result.errors),
                has_sorry="sorry" in lean_code,
                has_admit="admit" in lean_code
            )

    def verify_file(self, file_path: str) -> VerificationResult:
        """
        Verify a Lean file.

        Args:
            file_path: Path to the Lean file

        Returns:
            VerificationResult
        """
        with open(file_path, 'r') as f:
            lean_code = f.read()

        return self.verify(lean_code)



class MockLeanVerifier(LeanVerifier):
    """Mock verifier for testing without Lean installed."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def verify(self, lean_code: str) -> VerificationResult:
        """
        Mock verification that checks for basic syntax issues.

        This is a simple heuristic-based checker for testing purposes.
        """
        has_sorry = "sorry" in lean_code
        has_admit = "admit" in lean_code

        # Simple checks
        errors = []

        if not any(keyword in lean_code for keyword in ["theorem", "lemma", "def", "example"]):
            errors.append("No theorem/lemma/def/example declaration found")

        if lean_code.count("(") != lean_code.count(")"):
            errors.append("Mismatched parentheses")

        if lean_code.count("{") != lean_code.count("}"):
            errors.append("Mismatched braces")

        success = len(errors) == 0

        return VerificationResult(
            success=success,
            error_message="; ".join(errors) if errors else None,
            stdout="",
            stderr="; ".join(errors) if errors else "",
            has_sorry=has_sorry,
            has_admit=has_admit
        )

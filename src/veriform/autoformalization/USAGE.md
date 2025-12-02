# Autoformalization Framework Documentation
## 1. System Overview
This framework provides an interface to convert natural language reasoning steps (from a chain of thought) into formal Lean 4 code. The system is designed to be agnostic to the underlying engine (OpenAI, Anthropic, vLLM, or custom symbolic methods).

### Core Architecture
The system relies on a central abstract base class Autoformalization which processes a ReasoningChain step-by-step. The output of every operation is standardized via the `FormalizationResult` data class.

## 2. The Integration Contract: `FormalizationResult`
To integrate a new method, your implementation must return a `FormalizationResult` object for every step processed. This is the strict contract between the formalizer and the rest of the pipeline.

### The Data Structure
Defined in formalizer.py:
```Python
@dataclass
class FormalizationResult:
    step_id: str                  # Unique ID of the step being verified
    lean_code: str                # The extracted valid Lean 4 code (or empty string)
    raw_response: str             # The full raw output from the LLM/Solver for debugging
    success: bool                 # True if code was extracted/verified, False otherwise
    error_message: Optional[str]  # Captures API errors or extraction failures
    metadata: Dict[str, Any]      # Arbitrary data (e.g., attempt counts, confidence scores)
```
**Developer Note**: Even if your method fails to generate code, you should return a `FormalizationResult` with `success=False` rather than raising an exception, to ensure the rest of the chain can still be processed or analyzed.
## 3. How to Integrate New Methods
There are two ways to extend the framework, depending on the complexity of your new method.
### Strategy A: The LLM Wrapper (Simple)
*Best for: Adding support for Llama 3, Gemini, or a new hosted API.*
You only need to implement the abstract method `_call_llm`. The base class handles retries, exponential backoff, and loop management.
1. Create a class inheriting from Autoformalization.
2. Implement `_call_llm(self, prompt: str, system_prompt: str) -> str.`
```Python
class MyNewLLM(Autoformalization):
    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        # 1. format messages
        # 2. call your API
        # 3. return the string content
        return api_response_string
```
### Strategy B: Full Custom Logic (Advanced)
*Best for: Symbolic solvers, Rule-based systems, or multi-agent workflows.*
If your method doesn't fit the standard "Prompt -> String Response" paradigm (e.g., you need to run a subprocess, query a database, or run a local prover), you should override `formalize_step`.
```Python
class SymbolicFormalizer(Autoformalization):
    def formalize_step(self, step, context_steps, problem_statement):
        # Your custom logic here
        # ... logic to run symbolic engine ...
        
        return FormalizationResult(
            step_id=step.step_id,
            lean_code="theorem ...",
            raw_response="Symbolic engine output",
            success=True,
            metadata={"engine": "Isabelle-Bridge"}
        )
```
## 4. Understanding Prompt Flow & Templates
The framework (formalizer.py) includes specific logic for handling how prompts are constructed in `create_autoformalization_prompt`.
### The "Safe" vs. "Custom" Switch
The `formalize_step` method changes behavior based on the prompt object type:
1. Single-Pass (Standard):
    - If prompt is a `str`: The system calls `_call_llm` once.
    - It expects the response to contain a Lean code block.
2. Two-Pass (Autoformalization + Prover):
    - If prompt is a `tuple` (length 2):
        - Pass 1: Uses `prompt[0]` to translate Natural Language $\to$ Formal Statement.
        - Pass 2: Uses `prompt[1]` to take the Formal Statement $\to$ Proof (filling sorry).
    - Trigger: This mode is currently triggered when the `template="safe"` argument is passed during initialization.
### Formatting Toolkit
When creating new templates in templates.py, use the strict placeholders required by `SafeFormalTemplate`:
* <problem>: The global problem statement.
* <previous_steps>: The history of the chain.
* <current_step>: The specific text to formalize now.
## 5. Registration Guide
To make your new class accessible via the `factory` pattern:
1. Open formalizer.py.
2. Locate `get_formalizer` function (end of file).
3. Add your key-value pair to the dictionary:
```Python
def get_formalizer(...):
    formalizers = {
        "openai": OpenAIFormalizer,
        "anthropic": AnthropicFormalizer,
        "my_new_method": MyCustomFormalizer, # <--- Add this
        # ...
    }
```
## 6. Implementation Checklist
Use this checklist before pushing a new formalizer:
- [ ] Inheritance: Does the class inherit from Autoformalization?
- [ ] Contract: Does it return `FormalizationResult` (never None or plain strings)?
- [ ] Metadata: Are you populating metadata with useful debug info (e.g., retries, tokens used)?
- [ ] Error Handling: Does it catch API exceptions internally and populate last_error?
- [ ] Lean Extraction: If using the base `formalize_step`, does your LLM output format match LEAN_CODE_EXTRACTION_PATTERNS (markdown blocks)?

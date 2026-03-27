from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


MODEL_DIR = "deepseek-ai/DeepSeek-Prover-V2-7B"  # or DeepSeek-Prover-V2-671B
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

formal_statement = """
import Mathlib
import Aesop
set_option maxHeartbeats 0
open BigOperators Real Nat Topology Rat


theorem negation_rule {α: Type} {p : α → Prop} (h : ¬ ∀ x, ¬ p x) : ∃ x, p x := by sorry
""".strip()

prompt = """
Complete the following Lean 4 code:

```lean4
{}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()

chat = [
  {"role": "user", "content": prompt.format(formal_statement)},
]

model = LLM(
        MODEL_DIR, 
        tensor_parallel_size=1, # 8 for 32B, 4 for 7B
        gpu_memory_utilization=0.2, 
    )
inputs = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    max_tokens=16384,
    n=1
)
import time
start = time.time()
output = model.generate(inputs, sampling_params=sampling_params)
output_text = output[0].outputs[0].text
print(output_text)
print(time.time() - start)

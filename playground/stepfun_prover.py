from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_name = "stepfun-ai/StepFun-Prover-Preview-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = LLM(
    model=model_name,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.3,
)
formal_problem = """
import Mathlib

open Real

theorem my_favorite_theorem :
    IsLeast {t : ℝ | ∃ x y z : ℝ, 0 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ z ≤ 4 ∧ y^2 - x^2 = 2 ∧ z^2 - y^2 = 2 ∧ t = |x - y| + |y - z|} (4 - 2 * sqrt 3) := by
""".strip()

system_prompt = "You will be given an unsolved Lean 4 problem. Think carefully and work towards a solution. At any point, you may use the Lean 4 REPL to check your progress by enclosing your partial solution between <sketch> and </sketch>. The REPL feedback will be provided between <REPL> and </REPL>. Continue this process as needed until you arrive at a complete and correct solution."

user_prompt = f"```lean4\n{formal_problem}\n```"

dialog = [
  {"role": "system", "content": system_prompt},
  {"role": "user", "content": user_prompt}
] 

prompt = tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=True)

sampling_params = SamplingParams(
    temperature=0.999,
    top_p=0.95,
    top_k=-1,
    max_tokens=16384,
    stop_token_ids=[151643, 151666], # <｜end▁of▁sentence｜>, </sketch>
    include_stop_str_in_output=True,
)

output = model.generate(prompt, sampling_params=sampling_params)
output_text = output[0].outputs[0].text
print(output_text)

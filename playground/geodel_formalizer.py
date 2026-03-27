from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import torch
import re
import time

torch.manual_seed(30)

model_id = "Goedel-LM/Goedel-Formalizer-V2-8B"
model = LLM(model=model_id, 
          dtype="bfloat16", 
          trust_remote_code=True,
          gpu_memory_utilization=0.25)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

problem_name = "my_favorite_theorem"
informal_statement_content = "Prove that 3 cannot be written as the sum of two cubes."

user_prompt_content = (
    f"Please autoformalize the following natural language problem statement in Lean 4. "
    f"Use the following theorem name: {problem_name}\n"
    f"The natural language statement is: \n"
    f"{informal_statement_content}"
    f"Think before you provide the lean statement."
)

chat = [
    {"role": "user", "content": user_prompt_content},
]


inputs = tokenizer.apply_chat_template(
    chat, 
    tokenize=False, 
    add_generation_prompt=True
)

print(inputs)

sampling_params = SamplingParams(
    max_tokens=16384,
    temperature=0.9,
    top_k=20,
    top_p=0.95,
)

start = time.time()
outputs = model.generate(inputs, sampling_params=sampling_params)

model_output_text = outputs[0].outputs[0].text

def extract_code(text_input):
    """Extracts the last Lean 4 code block from the model's output."""
    try:
        matches = re.findall(r'```lean4\n(.*?)\n```', text_input, re.DOTALL)
        return matches[-1].strip() if matches else "No Lean 4 code block found."
    except Exception:
        return "Error during code extraction."

extracted_code = extract_code(model_output_text)

print(time.time() - start)
print("output:\n", model_output_text)
print("lean4 statement:\n", extracted_code)

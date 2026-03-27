from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def get_formal_statement_prompt(informal_problem: str, header: str = "import Mathlib\n") -> str:
    prompt = "Please autoformalize the following problem in Lean 4 with a header. Use the following theorem names: my_favorite_theorem.\n\n"
    prompt += informal_problem
    prompt += f"\n\nYour code should start with:\n```Lean4\n{header}\n```\n"
    return prompt

MODEL_DIR = "stepfun-ai/StepFun-Formalizer-7B"

if __name__ == "__main__":

    system_prompt = "You are an expert in mathematics and Lean 4."
    informal_problem = "I save 100 dollars every month. If I start with 500 dollars in my savings account, how much money will I have in the account after 6 months? The answer is 500 + 6 * 100 = 1100 dollars."
    header = "import Mathlib\n\nopen Real\n"
    user_prompt = get_formal_statement_prompt(informal_problem, header)

    dialog = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ] 

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    prompt = tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=True) + "<think>"
    print(f"prompt: {prompt}")

    model = LLM(
        MODEL_DIR, 
        tensor_parallel_size=2, # 8 for 32B, 4 for 7B
        gpu_memory_utilization=0.75,
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=16384,
        n=1
    )

    responses = model.generate(prompt, sampling_params)
    print(f"response: {responses[0].outputs[0].text}")

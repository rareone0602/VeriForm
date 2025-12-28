from vllm import LLM, SamplingParams
import re
from .base import BaseFormalizer

class Util: # From Herald repo
    @staticmethod
    def chat_template_to_prompt(messages: list[dict], model: str) -> str:
        """
        Chat template for deepseek and internlm
        """
        result = ""
        total_step = len(messages)
        for i, message in enumerate(messages):
            if model == "internlm":
                result += "<|im_start|>" + message["role"] + "\n" + message["content"]
                if i + 1 != total_step:
                    result += "<|im_end|>\n"
                elif message["role"] == "user":
                    result += "<|im_end|>\n<|im_start|>assistant\n"

            elif model == "deepseek":
                if message["role"] == "user":
                    result += "User:" + message["content"] + "\n\n"
                elif message["role"] == "assistant":
                    result += "Assistant" + message["content"] + "<｜end▁of▁sentence｜>"
                elif message["role"] == "system":
                    result += message["content"] + "\n\n"
                if i + 1 == total_step and message["role"] == "user":
                    result += "Assistant:"
            else:
                raise NotImplementedError
        return result

    @staticmethod
    def get_openai_messages(prompt: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": prompt}]

    @staticmethod
    def remove_informal_prefix(formal_statement: str) -> str:
        pattern = r"/-- [\s\S]*? -/\n"
        cleaned_text = re.sub(pattern, "", formal_statement, flags=re.DOTALL)
        return cleaned_text

    @staticmethod
    def extract_bold_text(output):
        match = re.search(r"\|\|(.*?)\|\|", output)
        if match:
            return match.group(1)
        return "null"

    @staticmethod
    def jsonltojson(path: str, output_path: str):
        data = []
        with open(path) as fp:
            for l in fp.read().splitlines():
                data.append(json.loads(l))
        with open(output_path, "w") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=4)


class HeraldFormalizer(BaseFormalizer):
    MODEL_DIR = "FrenzyMath/Herald_translator"

    def __init__(self):
        self.system_prompt = "You are an expert at Lean 4 and Mathematics."
        self.model = LLM(
            self.MODEL_DIR,
            dtype="bfloat16", 
            tensor_parallel_size=2, # 8 for 32B, 4 for 7B
            trust_remote_code=True,
            gpu_memory_utilization=0.25,
        )
        self.lean_pattern = re.compile(
            r".*(^theorem.*?sorry)",
            re.DOTALL | re.IGNORECASE | re.MULTILINE
        )

        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=16384,
        )

    def initialize_dialog(self):
        self.dialog = [{'role': 'system', 'content': self.system_prompt},]

    def get_formal_statement_prompt(self, informal_problem: str) -> str:
        return (
            f"Please translate the natural language statement to Lean4 code with the header\n"
            f"**Name**\nmy_favorite_theorem\n"
            f"**Informal statement**\n{informal_problem}\n"
        )
    
    def parse_lean_code(self, response_text: str) -> str:
        """
        Get the LAST Lean 4 code block from the response text.
        """
        matches = self.lean_pattern.findall(response_text)
        if matches:
            return matches[-1].strip()
        else:
            raise ValueError("No Lean 4 code block found in the response.")
    
    def format_dialog(self, dialog: list[dict]) -> str:
        return Util.chat_template_to_prompt(dialog, model="deepseek")
    
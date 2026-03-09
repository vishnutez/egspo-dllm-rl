import re
from gsm8k import GSM8KDataset
from datasets import load_dataset
from parsers import Parser, test_solution
import warnings

MBPP_SYSTEM_PROMPT = """You are a coding expert. You will be given a coding problem to solve. Solve it step by step. Ensure you wrap the answer in ```python````

Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
```python
Your code here
```
</answer>" 
"""


class MBPPDataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=MBPP_SYSTEM_PROMPT,
        subsample=-1,
        output_dir=None,
    ):
        if num_examples > 0:
            warnings.warn("num_examples must be 0 for MBPP. Overriding num_examples to 0.")
        super().__init__(tokenizer, 0, add_reasoning, system_prompt, subsample, output_dir)

    def load_test_dataset(self):
        self.dataset = load_dataset("mbpp", "sanitized", split="test")

    def parse_answer_and_score(self, generated_texts, ground_truths, **kwargs):
        preds = []
        num_correct = 0
        for generated_text, unit_test in zip(generated_texts, ground_truths):
            program = Parser.extract_answer_code(generated_text)
            if program is None:
                preds.append(program)
                continue
            # NOTE: Don't need to extract func name like human_eval since prompt specifies it
            # If model doesn't follow the instruction, it shouldn't pass
            test_code = program + "\n\n" + unit_test
            preds.append(test_code)
            num_correct += test_solution(test_code, self.output_dir)
        return preds, num_correct, len(preds)

    def __getitem__(self, idx):
        unit_tests = "\n".join(self.dataset[self.subsample[idx].item()]["test_list"])
        question = (
            self.dataset[self.subsample[idx].item()]["prompt"]
            + "\n\n Your code should pass the following tests:\n\n"
            + unit_tests
        )
        answer = "\n".join(self.dataset[self.subsample[idx].item()]["test_imports"]) + f"\n\n{unit_tests}"
        prompt = self.create_prompt(question)
        return prompt, question, answer
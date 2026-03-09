import re
from gsm8k import GSM8KDataset
from datasets import load_dataset
from parsers import Parser, test_solution, extract_human_eval_prompt
import warnings

HUMANEVAL_SYSTEM_PROMPT = """You are a coding expert. You will be given a coding problem to solve. Solve it step by step.

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


class HumanEvalDataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=HUMANEVAL_SYSTEM_PROMPT,
        subsample=-1,
        output_dir=None,
    ):
        if num_examples > 0:
            warnings.warn("num_examples must be 0 for HumanEval. Overriding num_examples to 0.")
        super().__init__(tokenizer, 0, add_reasoning, system_prompt, subsample, output_dir)

    def load_test_dataset(self):
        self.dataset = load_dataset("openai/openai_humaneval", split="test")

    def parse_answer_and_score(self, generated_texts, ground_truths, **kwargs):
        preds = []
        num_correct = 0
        for generated_text, ground_truth in zip(generated_texts, ground_truths):
            program = Parser.extract_answer_code(generated_text)
            unit_test = ground_truth[ground_truth.index("def") :]
            if program is None:
                preds.append(program)
                continue
            solution_match = re.search(r"def (\w+)\(", program)
            if not solution_match:
                preds.append(program)
                continue
            defined_func = solution_match.group(1)
            test_code = program + "\n\n" + unit_test + "\n\n" + f"check({defined_func})"
            preds.append(test_code)
            num_correct += test_solution(test_code, self.output_dir)
        return preds, num_correct, len(preds)

    def __getitem__(self, idx):
        question = extract_human_eval_prompt(self.dataset[self.subsample[idx].item()]["prompt"])
        answer = self.dataset[self.subsample[idx].item()]["test"]
        prompt = self.create_prompt(question)
        return prompt, question, answer
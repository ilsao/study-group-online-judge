import hashlib
import json
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer

from labs.lab1 import gpt2_small


def _format_question(row: dict, include_answer: bool = False) -> str:
    options = " ".join(
        f"({chr(ord('A') + index)}) {choice}"
        for index, choice in enumerate(row["choices"])
    )
    answer = chr(ord("A") + row["answer"]) if include_answer else ""
    return f'{row["question"]}\n{options}\nAnswer: {answer}'


def _format_prompt(row: dict, exemplars: list[dict]) -> str:
    subject = row["subject"].replace("_", " ")
    questions = [
        *(_format_question(example, include_answer=True) for example in exemplars),
        _format_question(row),
    ]
    return (
        f"The following are multiple choice questions about {subject}.\n\n"
        + "\n\n".join(questions)
    )


def mmlu_eval() -> dict[str, str]:
    """Return GPT-2's A/B/C/D prediction for every MMLU test question.

    Load ``cais/mmlu`` at revision
    ``c30699e8356da336a370243923dbaf21066bb9fe``. For each subject, use
    its first four ``dev`` questions as exemplars and evaluate its ``test``
    questions. Format the prompt as specified in the Lab 2 assignment. If a
    prompt exceeds GPT-2's context window, retain its final 1024 tokens.

    Each key is the SHA-256 of a compact UTF-8 JSON object with keys
    ``index``, ``subject``, ``question``, and ``choices`` (sorted keys,
    ``ensure_ascii=False``, compact separators). ``index`` is the zero-based
    row number of the pinned ``all`` test split. This disambiguates repeated
    questions, including 27 identical subject/question/choice rows. Each
    value is one of A/B/C/D, selected from the corresponding next-token
    logits. No question labels should be used to choose a prediction.
    """

    ex_dataset = load_dataset(
        "cais/mmlu",
        "all",
        revision="c30699e8356da336a370243923dbaf21066bb9fe",
        split="dev"
    )

    eval_dataset = load_dataset(
        "cais/mmlu",
        "all",
        revision="c30699e8356da336a370243923dbaf21066bb9fe",
        split="test"
    )

    exemplars = defaultdict(list)
    for row in ex_dataset:
        if len(exemplars[row["subject"]]) < 4:
            exemplars[row["subject"]].append(row)

    res = {}
    keys = []
    prompts = []

    eval_size = len(eval_dataset)
    for i in range(eval_size):
        row = eval_dataset[i]
        key = hashlib.sha256(
            json.dumps(
                {
                    "index": i,
                    "subject": row["subject"],
                    "question": row["question"],
                    "choices": row["choices"],
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        prompt = _format_prompt(row, exemplars[row["subject"]])
        keys.append(key)
        prompts.append(prompt)

    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    option_ids = tokenizer(
        ['A', 'B', 'C', 'D'],
        return_tensors="pt"
    )["input_ids"]
    option_ids = option_ids.reshape(1, -1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = gpt2_small.from_pretrained().to(device).eval()
    option_ids = option_ids.to(device)

    with torch.inference_mode():
        for start in range(0, eval_size, 8):
            batch = prompts[start:start + 8]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )
            logits = model(
                encoded["input_ids"].to(device),
                encoded["attention_mask"].to(device),
            )[:, -1, :]
            batch_option_ids = option_ids.expand(len(batch), -1)
            option_logits = torch.gather(
                logits, dim=-1, index=batch_option_ids
            )
            answer_indexes = torch.argmax(option_logits, dim=-1).tolist()

            for offset, answer_index in enumerate(answer_indexes):
                res[keys[start + offset]] = "ABCD"[answer_index]

    return res

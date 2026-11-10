"""Prompt templates proposed for the two candidate bases of the judge, and their checks.

Input of the judge: the whole context and one sentence, nothing else. Output: one label
among three. The score is read at the last position of the prompt, from the logits of the
first token of each label: score = P(Supported) among the three.

Maximum input length: 4096 tokens for template, context and sentence together. Beyond it
the sentence is kept whole and the context is truncated at its end. No windowing.

    python analysis/prompt_templates.py labels            first tokens of the labels, both bases
    python analysis/prompt_templates.py lengths FILE...   input lengths of JSONL files (context, sentence)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

MAX_LENGTH = 4096
# Written as the judge outputs them. In the data files the same labels are in upper case.
LABELS = ("Supported", "Unsupported", "Contradicted")

BASES = {
    "qwen3": {"repo": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca"},
    "pleias": {"repo": "PleIAs/Pleias-RAG-350M", "revision": "db001b29a33532583d14979c3c38ef04b6d59352"},
}

INSTRUCTION = (
    "Judge whether the sentence is supported by the context. Answer with one label.\n"
    "Supported: the context states or directly implies the sentence, or the sentence makes no claim (an announcement or a courtesy formula).\n"
    "Unsupported: the context neither states nor contradicts the sentence.\n"
    "Contradicted: the context contradicts the sentence."
)

# Qwen3-0.6B: its own chat format, thinking disabled. The label follows the generation prompt.
QWEN3_USER = "Context:\n{context}\n\nSentence:\n{sentence}"

# Pleias-RAG-350M: its native markers. The instruction and the sentence form the query, the
# context is the single source, and the label follows <|answer_start|>.
PLEIAS_PROMPT = (
    "<|query_start|>" + INSTRUCTION + "\nSentence: {sentence}<|query_end|>\n"
    "<|source_start|><|source_id|>1 {context}<|source_end|>\n"
    "<|answer_start|>"
)


def load_tokenizer(base: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(BASES[base]["repo"], revision=BASES[base]["revision"])


def render(base: str, tokenizer: Any, context: str, sentence: str) -> str:
    """The prompt, up to the position where the label is read."""
    if base == "pleias":
        return PLEIAS_PROMPT.format(context=context, sentence=sentence)
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": QWEN3_USER.format(context=context, sentence=sentence)},
    ]
    return str(
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    )


def encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_input(
    base: str, tokenizer: Any, context: str, sentence: str, max_length: int = MAX_LENGTH
) -> tuple[list[int], bool]:
    """Token ids of the prompt and whether the context was truncated at its end."""
    ids = encode(tokenizer, render(base, tokenizer, context, sentence))
    if len(ids) <= max_length:
        return ids, False
    context_ids = encode(tokenizer, context)
    keep = len(context_ids) - (len(ids) - max_length)
    while keep > 0:
        shorter = tokenizer.decode(context_ids[:keep], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        ids = encode(tokenizer, render(base, tokenizer, shorter, sentence))
        if len(ids) <= max_length:
            return ids, True
        keep -= max(1, len(ids) - max_length)
    raise ValueError(f"template and sentence alone exceed {max_length} tokens")


def label_first_tokens(base: str, tokenizer: Any, labels: tuple[str, ...] = LABELS) -> dict[str, dict[str, Any]]:
    """First token of each label as it is tokenised right after the prompt."""
    prompt = render(base, tokenizer, "A context.", "A sentence.")
    prompt_ids = encode(tokenizer, prompt)
    result = {}
    for label in labels:
        ids = encode(tokenizer, prompt + label)
        if ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"{base}: adding {label!r} changes the tokens of the prompt")
        own = ids[len(prompt_ids) :]
        result[label] = {"ids": own, "first_id": own[0], "pieces": [tokenizer.decode([i]) for i in own]}
    return result


def check_labels() -> None:
    variants = {
        "capitalised (proposed)": LABELS,
        "upper case": tuple(label.upper() for label in LABELS),
        "lower case": tuple(label.lower() for label in LABELS),
    }
    for base in BASES:
        tokenizer = load_tokenizer(base)
        overhead = len(encode(tokenizer, render(base, tokenizer, "", "")))
        print(f"{base} ({BASES[base]['repo']}): template of {overhead} tokens without context and sentence")
        for name, labels in variants.items():
            tokens = label_first_tokens(base, tokenizer, labels)
            first = [tokens[label]["first_id"] for label in labels]
            distinct = len(set(first)) == len(first)
            detail = "; ".join(f"{label} = {tokens[label]['pieces']}" for label in labels)
            print(f"   {name}: first tokens {'distinct' if distinct else 'NOT DISTINCT'} {first} | {detail}")


def check_lengths(paths: list[Path]) -> None:
    tokenizers = {base: load_tokenizer(base) for base in BASES}
    for path in paths:
        records = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        groups = Counter(record.get("dataset", "all") for record in records)
        for base, tokenizer in tokenizers.items():
            overhead = len(encode(tokenizer, render(base, tokenizer, "", "")))
            cache: dict[str, int] = {}
            estimate = []
            for record in records:
                context = record["context"]
                if context not in cache:
                    cache[context] = len(encode(tokenizer, context))
                estimate.append(overhead + cache[context] + len(encode(tokenizer, record["sentence"])))
            # exact count for every input close to or beyond the limit, estimate elsewhere (it is exact to one token)
            exact = list(estimate)
            for index, value in enumerate(estimate):
                if value > MAX_LENGTH - 8:
                    record = records[index]
                    exact[index] = len(
                        encode(tokenizer, render(base, tokenizer, record["context"], record["sentence"]))
                    )
            longest = max(range(len(records)), key=estimate.__getitem__)
            record = records[longest]
            longest_exact = len(encode(tokenizer, render(base, tokenizer, record["context"], record["sentence"])))
            beyond = Counter(
                record.get("dataset", "all") for record, value in zip(records, exact, strict=True) if value > MAX_LENGTH
            )
            ordered = sorted(estimate)
            print(
                f"{path.name}, {base}: {len(records)} inputs, median {ordered[len(ordered) // 2]}, "
                f"longest {longest_exact} tokens, beyond {MAX_LENGTH}: {sum(beyond.values())}"
                + (
                    f" { ({name: f'{beyond[name]} of {groups[name]}' for name in groups if beyond[name]}) }"
                    if beyond
                    else ""
                )
            )
            if longest_exact > MAX_LENGTH:
                ids, truncated = build_input(base, tokenizer, record["context"], record["sentence"])
                tail = encode(tokenizer, record["sentence"])
                print(
                    f"   longest input after truncation: {len(ids)} tokens, truncated {truncated}, "
                    f"sentence kept whole: {tokenizer.decode(tail) in tokenizer.decode(ids)}"
                )


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "labels":
        check_labels()
    elif len(sys.argv) >= 3 and sys.argv[1] == "lengths":
        check_lengths([Path(argument) for argument in sys.argv[2:]])
    else:
        raise SystemExit(__doc__)

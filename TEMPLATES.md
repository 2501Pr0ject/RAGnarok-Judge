# Prompt templates of the two candidate bases

Status: declared on 2026-11-09 and written into `TARGETS.md`, section 3; instruction amended
on 2026-11-10 (see below). Reference implementation and checks: `prompt_templates.py` in this
folder.

## Principles

- Input: the whole context and the sentence, nothing else. No question, no previous
  sentence, no other part of the answer.
- Output: one label among three, `Supported`, `Unsupported`, `Contradicted`. In the data
  files the same labels are written in upper case.
- Score: read at the last position of the prompt, in one forward pass, from the logits of the
  first token of each label. The three logits are converted to float32 before the softmax, so
  that the score is not limited to the few hundred distinct values a bfloat16 softmax would
  give, which would multiply the ties at the threshold. Score = P(Supported) among the three.
  Nothing is generated.
- Maximum input length: 4096 tokens for template, context and sentence together. Beyond it
  the sentence is kept whole and the context is truncated at its end. No windowing.
- The instruction is the same for both bases. It says that a sentence making no claim is
  `Supported`, and names what that covers: an announcement or a courtesy formula, which is
  how such sentences are labelled in the training set. It does not say that an abstention is
  `Supported`: a sentence saying that the context does not answer the question makes a claim
  that is true or false according to the question, which the judge does not receive. RJ gets
  no instruction about such sentences and scores them as it does any sentence; no judge of the
  protocol can check them, and `TARGETS.md` (sections 3 and 6) reports behaviour on them, not
  verification. Until 2026-11-10 the clause read "or the sentence makes no claim" without
  naming its scope.

## Instruction, common to both bases

```
Judge whether the sentence is supported by the context. Answer with one label.
Supported: the context states or directly implies the sentence, or the sentence makes no claim (an announcement or a courtesy formula).
Unsupported: the context neither states nor contradicts the sentence.
Contradicted: the context contradicts the sentence.
```

## Qwen3-0.6B

Its own chat format, built by the tokenizer's chat template with thinking disabled. System
message: the instruction. User message:

```
Context:
{context}

Sentence:
{sentence}
```

The prompt ends with the generation prompt of the template, and the label follows it:

```
<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
Context:
{context}

Sentence:
{sentence}<|im_end|>
<|im_start|>assistant
<think>

</think>

{label}<|im_end|>
```

## Pleias-RAG-350M

Its native markers. The instruction and the sentence form the query, the context is the
single source, and the label follows `<|answer_start|>`, without the reasoning steps the
base generates natively:

```
<|query_start|>{instruction}
Sentence: {sentence}<|query_end|>
<|source_start|><|source_id|>1 {context}<|source_end|>
<|answer_start|>{label}<|answer_end|>
```

## Checks

First token of each label, tokenised right after the prompt:

| Base | Template, tokens | Supported | Unsupported | Contradicted | First tokens distinct |
|---|---|---|---|---|---|
| Qwen3-0.6B | 84 | `Supported` (34636) | `Unsupported` (41884) | `Contr`, `ad`, `icted` (41437) | yes |
| Pleias-RAG-350M | 83 | `Supported` (57889) | `Un`, `supported` (2076) | `Cont`, `rad`, `icted` (3099) | yes |

The first tokens are also distinct in upper case and in lower case for both tokenizers. Upper
case was not kept: in Qwen3 the first token of `SUPPORTED` is the single letter `S`.

Input lengths with these templates, in tokens (inputs only, no label, no score), measured on
2026-11-10 with the amended instruction; the counts beyond 4096 are those of the first
measurement:

| File | Inputs | Qwen3-0.6B: median, longest | Pleias-RAG-350M: median, longest | Beyond 4096 |
|---|---|---|---|---|
| train | 54655 | 529, 2540 | 543, 2508 | 0 |
| dev | 6011 | 547, 1949 | 561, 1990 | 0 |
| test | 10385 | 532, 2181 | 543, 2277 | 0 |
| out of domain, five datasets | 8424 | 279, 28182 | 280, 27154 | 282 and 293 |

Out of domain the inputs beyond 4096 tokens are in WiCE (31 and 33 of 358), ClaimVerify (141
and 142 of 1088) and ExpertQA (110 and 118 of 3702). On the longest input the truncation
gives exactly 4096 tokens with the sentence whole, for both bases.

## Not checked

- No base was loaded: these are checks of tokenisation, not of behaviour.
- The cost runs of 2026-11-09 used shorter provisional templates (43 and 21 tokens). The
  templates above add about 41 and 62 tokens per input.

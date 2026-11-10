"""MiniCheck-Flan-T5-Large with bounded forward passes, for the out-of-domain benchmark.

``MiniCheckJudge`` sends the chunks of a document to the model in batches of 16, each batch
padded to its longest input, and never releases the device cache. The attention memory of one
forward pass grows with ``rows * padded_length ** 2``: 16 chunks padded to 2048 tokens need
64 times what one input of 2048 tokens needs. On documents of several thousand words the
process does not fit in 16 GB, and the buffers of the largest batch stay allocated for the
calls that follow.

The classes below change how the forward passes are cut, and nothing else:

- chunks, texts, batches of 16, tokenization, truncation and padding are those of the stock
  judge: every row has exactly the input the stock judge gives it;
- the rows of a batch go through the model in groups such that
  ``rows * padded_length ** 2 <= 2048 ** 2``, the attention size of one input of maximum
  length: 7 rows at 760 tokens, 4 at 1024, 1 at 2048;
- the device cache is released after each batch of 16, so after each call as well.

Used through the ``python:<module>:<Class>`` judge of ``ragnarok judge-bench``, with this
folder on ``PYTHONPATH``:

    PYTHONPATH=ood ragnarok judge-bench --judge python:minicheck_bounded:MiniCheckBoundedMps ...

The class name and the rule are part of the configuration: a run with these classes has its
own ``config_hash`` and never resumes a run of the stock judge.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ragnarok_ai.judges.minicheck import FLAN_LABEL_TOKEN_IDS, FLAN_PREFIX, OFFICIAL_BATCH_SIZE, MiniCheckJudge

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ragnarok_ai.judges.contract import JudgeConfig

CHECKPOINT = "flan-t5-large"
# rows * padded_length ** 2 of one forward pass: what one input of 2048 tokens needs
MAX_ATTENTION_CELLS = 2048 * 2048
FORWARD_RULE = (
    f"official batches of {OFFICIAL_BATCH_SIZE} for tokenization and padding; "
    f"rows * padded_length ** 2 <= {MAX_ATTENTION_CELLS} per forward pass; "
    f"device cache released after each official batch"
)


def forward_rows(padded_length: int, max_cells: int = MAX_ATTENTION_CELLS) -> int:
    """Rows of one forward pass for inputs padded to *padded_length* tokens; never below one."""
    return max(1, max_cells // (padded_length * padded_length))


class BoundedSupport:
    """The stock backend, ``support`` computed in bounded forward passes."""

    def __init__(self, stock: Any, max_cells: int = MAX_ATTENTION_CELLS) -> None:
        if stock.checkpoint.architecture != "seq2seq":
            msg = f"BoundedSupport handles the Flan-T5 checkpoint only, got {stock.checkpoint.name}"
            raise ValueError(msg)
        self.stock = stock
        self.max_cells = max_cells
        self.forward_passes = 0

    def split_sentences(self, text: str) -> list[str]:
        return self.stock.split_sentences(text)

    def splitter_id(self) -> str:
        return self.stock.splitter_id()

    def count_tokens(self, text: str) -> int:
        return self.stock.count_tokens(text)

    def release(self) -> None:
        """Give the cached device buffers back to the system."""
        kind = self.stock.device.split(":", 1)[0]
        if kind == "mps":
            self.stock._torch.mps.empty_cache()
        elif kind == "cuda":
            self.stock._torch.cuda.empty_cache()

    def support(self, chunks: Sequence[str], claim: str) -> tuple[list[float], int, bool]:
        """Same result as the stock ``support``: P(supported) per chunk, tokens sent, truncation."""
        stock = self.stock
        torch = stock._torch
        spec = stock.checkpoint
        tokenizer = stock.tokenizer
        eos = tokenizer.eos_token
        texts = [FLAN_PREFIX + eos.join([chunk, claim]) for chunk in chunks]
        probabilities: list[float] = []
        tokens = 0
        truncated = False
        for start in range(0, len(texts), OFFICIAL_BATCH_SIZE):
            batch = texts[start : start + OFFICIAL_BATCH_SIZE]
            full = tokenizer(batch, padding=False, truncation=False)["input_ids"]
            truncated = truncated or any(len(ids) > spec.max_model_len for ids in full)
            tokens += sum(min(len(ids), spec.max_model_len) for ids in full)
            inputs = tokenizer(batch, max_length=spec.max_model_len, truncation=True, padding=True, return_tensors="pt")
            rows = forward_rows(int(inputs["input_ids"].size(1)), self.max_cells)
            parts = []
            try:
                with torch.inference_mode():
                    for first in range(0, len(batch), rows):
                        input_ids = inputs["input_ids"][first : first + rows].to(stock.device)
                        attention_mask = inputs["attention_mask"][first : first + rows].to(stock.device)
                        decoder_input_ids = torch.zeros((input_ids.size(0), 1), dtype=torch.long).to(stock.device)
                        logits = stock.model(
                            input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids
                        ).logits.squeeze(1)
                        parts.append(logits[:, torch.tensor(FLAN_LABEL_TOKEN_IDS)].cpu())
                        self.forward_passes += 1
                        del input_ids, attention_mask, decoder_input_ids, logits
                    probs = torch.nn.functional.softmax(torch.cat(parts), dim=-1)
            finally:
                self.release()
            probabilities.extend(float(p) for p in probs[:, 1].tolist())
        return probabilities, tokens, truncated


class _MiniCheckBounded(MiniCheckJudge):
    DEVICE = "cpu"

    def __init__(self) -> None:
        super().__init__(checkpoint=CHECKPOINT, unit="answer", device=self.DEVICE)
        self.backend = BoundedSupport(self.backend)

    def _answer_config(self, criterion: str) -> JudgeConfig:
        base = super()._answer_config(criterion)
        return replace(base, judge_specific=tuple(sorted((*base.judge_specific, ("forward_passes", FORWARD_RULE)))))


class MiniCheckBoundedMps(_MiniCheckBounded):
    """MiniCheck-Flan-T5-Large on MPS, forward passes bounded, cache released after each batch."""

    DEVICE = "mps"


class MiniCheckBoundedCpu(_MiniCheckBounded):
    """MiniCheck-Flan-T5-Large on CPU, forward passes bounded."""

    DEVICE = "cpu"

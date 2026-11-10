"""HHEM-2.1-Open with a maximum input length, for the out-of-domain benchmark.

``ragnarok judge-bench`` builds HHEM without any length limit and has no option to set one.
On documents of several thousand tokens the judge cannot run: attention memory grows with
the square of the input length. These classes give HHEM the limit used for the candidate
judge: 4096 tokens of HHEM's own tokenizer, the sentence kept whole and the context
truncated at its end (the judge's own ``max_input_tokens`` option).

Used through the ``python:<module>:<Class>`` judge of ``ragnarok judge-bench``, with this
folder on ``PYTHONPATH``:

    PYTHONPATH=ood ragnarok judge-bench --judge python:hhem_l4096:HHEM4096Mps ...

The MPS class releases the allocator cache after each call, which lowers the peak memory and
does not change the scores. The CPU class is the fallback if MPS runs out of memory; on
RAGTruth the two devices gave scores identical to within 3e-6.
"""

from __future__ import annotations

from typing import Any

from ragnarok_ai.judges.hhem import HHEMJudge

MAX_INPUT_TOKENS = 4096


class HHEM4096Mps(HHEMJudge):
    """HHEM on MPS, inputs limited to 4096 tokens, cache released after each call."""

    def __init__(self) -> None:
        super().__init__(unit="answer", device="mps", max_input_tokens=MAX_INPUT_TOKENS)
        import torch

        self._torch = torch

    async def judge(self, criterion: str, *, question: str, context: str, answer: str) -> Any:
        try:
            return await super().judge(criterion, question=question, context=context, answer=answer)
        finally:
            self._torch.mps.empty_cache()


class HHEM4096Cpu(HHEMJudge):
    """HHEM on CPU, inputs limited to 4096 tokens."""

    def __init__(self) -> None:
        super().__init__(unit="answer", device="cpu", max_input_tokens=MAX_INPUT_TOKENS)

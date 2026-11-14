"""The retained judge of RAGnarok Judge, for the ``judge-bench`` harness (TRAINING.md, section 6).

Qwen3-0.6B at its pinned revision, weights in bfloat16 on MLX, carrying the LoRA adapters of
the selected checkpoint, kept separate and not merged: the object selected on dev, measured
for cost and evaluated on test and out of domain. One forward pass per sentence, the three
label logits read at the last position (``judge.py``), and the score is P(Supported) among
the three. The device cache is released after each call. The question is not part of the
input (TARGETS.md, section 3).

    PYTHONPATH=training RJ_ADAPTERS=RUN_DIR/checkpoint-3462 HF_HUB_OFFLINE=1 ragnarok judge-bench --judge python:rj_judge:RJQwen3 --data DATA.jsonl --out OUT_DIR

``RJ_ADAPTERS`` names the checkpoint folder written by ``train.py``. The SHA-256 of its
``adapters.safetensors`` must be ``ADAPTERS_SHA256``: the judge refuses to score with other
weights. Contexts beyond L = 4096 tokens are truncated at their end, the sentence kept whole
(``analysis/prompt_templates.py``), and each call reports it in ``trace.context_truncated``.

``Scorer`` needs MLX only; ``RJQwen3`` and ``judge_config`` also need ``ragnarok-ai``, which
the harness environment provides (requirements-frozen.txt).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import judge as rj  # training/judge.py; puts analysis/ on sys.path
import prompt_templates as pt

BASE = "qwen3"
CHECKPOINT = "qwen3-lr3e-4/checkpoint-3462"
ADAPTERS_SHA256 = "44cdeecac3ae75c9bb53d538a14968667f9af2fa730a9dfae9477b6f2b1bed31"
LABELS = rj.LABELS  # SUPPORTED, UNSUPPORTED, CONTRADICTED: the order of the three logits
POSITIVE = "SUPPORTED"
CRITERIA = frozenset({"faithfulness"})
JUDGE_CLASS = "rj_judge.RJQwen3"
# The template of the base before the sample is inserted: system prompt and user message.
TEMPLATE_SHA256 = hashlib.sha256(json.dumps([pt.INSTRUCTION, pt.QWEN3_USER]).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_adapters(folder: Path, base: str = BASE, expected: str | None = ADAPTERS_SHA256) -> tuple[str, dict]:
    """SHA-256 of the adapters of ``folder`` and its saved state; refuses other weights."""
    sha = file_sha256(folder / "adapters.safetensors")
    if expected is not None and sha != expected:
        raise ValueError(f"{folder}: adapters.safetensors has SHA-256 {sha}, the retained judge has {expected}")
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    if state["config"]["base"] != base:
        raise ValueError(f"{folder}: trained on {state['config']['base']}, not on {base}")
    return sha, state


class Scorer:
    """The base in bfloat16 with the separate adapters of ``folder``; three probabilities per call."""

    def __init__(self, folder: Path, base: str = BASE, expected_sha256: str | None = ADAPTERS_SHA256) -> None:
        import mlx.core as mx

        self.base = base
        self.adapters_sha256, state = checked_adapters(folder, base, expected_sha256)
        self.updates = state["updates_done"]
        self.model, self.tokenizer = rj.load_base(base)
        rj.add_lora(self.model, state["config"]["seed"])
        rj.load_adapters(self.model, folder)
        self.model.eval()
        self.labels = mx.array(rj.label_ids(base, self.tokenizer))

    def probabilities(self, context: str, sentence: str) -> tuple[list[float], int, bool]:
        """P(Supported), P(Unsupported), P(Contradicted); prompt length; whether the context was cut."""
        import mlx.core as mx

        ids, truncated = pt.build_input(self.base, self.tokenizer, context, sentence, pt.MAX_LENGTH)
        probs = mx.softmax(rj.label_logits(self.model, mx.array([ids]), self.labels), axis=-1)[0]
        values = [float(v) for v in probs.tolist()]
        mx.clear_cache()
        return values, len(ids), truncated


def verdict_policy() -> Any:
    from ragnarok_ai.judges.contract import LabelMap

    return LabelMap((("SUPPORTED", "PASS"), ("UNSUPPORTED", "FAIL"), ("CONTRADICTED", "FAIL")))


def judge_config(criterion: str, adapters_sha256: str = ADAPTERS_SHA256) -> Any:
    """The run configuration of the retained judge; ``dev_attempts.py`` writes the same hash."""
    from ragnarok_ai import __version__
    from ragnarok_ai.judges.contract import JudgeConfig

    return JudgeConfig(
        judge_class=JUDGE_CLASS,
        judge_kind="classifier",
        model=pt.BASES[BASE]["repo"],
        criterion=criterion,
        verdict_policy=verdict_policy(),
        ragnarok_ai_version=str(__version__),
        backend="mlx",
        model_digest=f"{pt.BASES[BASE]['revision']}+lora:{adapters_sha256}",
        options=(
            ("adapters", "separate"),
            ("cache_released_after_call", True),
            ("dtype", "bfloat16"),
            ("max_input_tokens", pt.MAX_LENGTH),
        ),
        prompt_format=None,
        prompt_template_sha256=TEMPLATE_SHA256,
        judge_specific=(
            ("checkpoint", CHECKPOINT),
            ("label_tokens", "34636,41884,41437"),
            ("lora_rank", rj.LORA_RANK),
            ("lora_scale", rj.LORA_SCALE),
            ("question_used", False),
            ("score", "P(Supported) among the three labels"),
        ),
        source_commit=None,
    )


class RJQwen3:
    """``JudgeProtocol`` of the retained judge, built without arguments by ``judge-bench``."""

    def __init__(self) -> None:
        folder = os.environ.get("RJ_ADAPTERS")
        if not folder:
            raise ValueError("RJ_ADAPTERS must name the checkpoint folder of the retained judge")
        self.scorer = Scorer(Path(folder))

    @property
    def criteria(self) -> frozenset[str]:
        return CRITERIA

    def config(self, criterion: str) -> Any:
        if criterion not in CRITERIA:
            raise ValueError(f"Unknown criterion {criterion!r}; expected one of {sorted(CRITERIA)}")
        return judge_config(criterion, self.scorer.adapters_sha256)

    async def judge(self, criterion: str, *, question: str, context: str, answer: str) -> Any:  # noqa: ARG002
        from ragnarok_ai.evaluators.judge import JudgeResult
        from ragnarok_ai.judges.contract import CallTrace, ClassProbabilities

        config = self.config(criterion)
        started = time.perf_counter()
        probs, tokens, truncated = self.scorer.probabilities(context, answer)
        output = ClassProbabilities(labels=LABELS, probs=tuple(probs), positive_label=POSITIVE)
        explanation = f"RJ P(Supported) = {probs[0]:.3f} among the three labels"
        if truncated:
            explanation += f"; context truncated to fit {pt.MAX_LENGTH} tokens"
        return JudgeResult(
            verdict=config.verdict_policy.verdict(output),
            score=probs[0],
            explanation=explanation,
            criteria=criterion,
            raw_score=None,
            output=output,
            config=config,
            trace=CallTrace(prompt_tokens=tokens, completion_tokens=None, context_truncated=truncated,
                            latency_ms=(time.perf_counter() - started) * 1000.0),
        )

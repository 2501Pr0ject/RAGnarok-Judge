"""Tests of the scoring plugin of the retained judge and of the dev attempts converter.

    python -m pytest training/test_rj_judge.py -q

No data file is read. The rows of the converter are checked against the readers of the
analysis scripts. The tests of ``RJQwen3`` and of the configuration need ``ragnarok-ai`` (the
harness environment) and are skipped without it; they use a stand-in scorer, no weights. With
``RJ_BASES=1`` and ``RJ_ADAPTERS`` naming the retained checkpoint, the scorer is compared with
``judge.score`` on synthetic inputs, and a context beyond 4096 tokens is truncated.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

import dev_attempts
import dev_eval
import rj_judge

IDS = ["a", "b", "c"]
SCORES = {"a": 0.9876543210123, "b": 0.000123456789, "c": 0.5}


def write_converted(path: Path, config_hash: str = "h" * 64) -> None:
    header = {"header": 1, "config_hash": config_hash, "criterion": "faithfulness", "level": "sentence",
              "sample": None, "per_class": None, "seed": None, "stratify": None, "data_sha256": "d" * 64,
              "source_commit": None}
    rows = [dev_attempts.attempt_row(i, SCORES[i], 100 + k, i == "c", config_hash) for k, i in enumerate(IDS)]
    dev_attempts.write_attempts(path, header, rows)


def test_converted_rows_are_read_back_exactly_by_the_analysis(tmp_path: Path) -> None:
    path = tmp_path / "dev_attempts.jsonl"
    write_converted(path)
    scores, truncated, header = dev_eval.ra.read_attempts(path, set(IDS), "d" * 64)
    assert scores == SCORES and truncated == {"c"} and header["config_hash"] == "h" * 64
    with pytest.raises(SystemExit):
        dev_eval.ra.read_attempts(path, set(IDS), "e" * 64)  # another data file


def test_converted_file_is_written_once(tmp_path: Path) -> None:
    path = tmp_path / "dev_attempts.jsonl"
    write_converted(path)
    with pytest.raises(SystemExit):
        write_converted(path)


def test_adapters_of_another_checkpoint_are_refused(tmp_path: Path) -> None:
    (tmp_path / "adapters.safetensors").write_bytes(b"not the retained adapters")
    (tmp_path / "state.json").write_text(json.dumps({"config": {"base": "qwen3"}, "updates_done": 1}))
    with pytest.raises(ValueError, match="retained judge"):
        rj_judge.checked_adapters(tmp_path)
    sha, _ = rj_judge.checked_adapters(tmp_path, expected=None)
    assert sha == rj_judge.file_sha256(tmp_path / "adapters.safetensors")
    with pytest.raises(ValueError, match="trained on qwen3"):
        rj_judge.checked_adapters(tmp_path, base="pleias", expected=None)


# ----------------------------------------------------------------------------- harness (ragnarok-ai)


class StandInScorer:
    adapters_sha256 = rj_judge.ADAPTERS_SHA256

    def __init__(self, probs: list[float], truncated: bool) -> None:
        self.result = (probs, 1234, truncated)

    def probabilities(self, context: str, sentence: str) -> tuple[list[float], int, bool]:
        return self.result


def stand_in_judge(probs: list[float], truncated: bool = False) -> rj_judge.RJQwen3:
    judge = object.__new__(rj_judge.RJQwen3)
    judge.scorer = StandInScorer(probs, truncated)  # type: ignore[assignment]
    return judge


def test_judge_result_carries_the_score_the_probabilities_and_the_truncation() -> None:
    pytest.importorskip("ragnarok_ai")
    from ragnarok_ai.benchmarks.judges import attempt_to_dict
    from ragnarok_ai.judges.contract import JudgeAttempt

    probs = [0.25, 0.70, 0.05]
    judge = stand_in_judge(probs, truncated=True)
    result = asyncio.run(judge.judge("faithfulness", question="q", context="c", answer="s"))
    assert result.score == 0.25 and result.output.probs == tuple(probs) and result.verdict == "FAIL"
    assert result.trace.context_truncated is True and result.trace.prompt_tokens == 1234
    attempt = JudgeAttempt("x", "faithfulness", result.config, result=result, error=None, trace=result.trace)
    row = attempt_to_dict(attempt)  # as the harness writes it
    assert row["result"]["score"] == 0.25 and row["trace"]["context_truncated"] is True
    supported = asyncio.run(stand_in_judge([0.6, 0.3, 0.1]).judge("faithfulness", question="", context="c", answer="s"))
    assert supported.verdict == "PASS"
    with pytest.raises(ValueError):
        judge.config("relevance")


def test_converter_and_plugin_share_the_configuration_hash() -> None:
    pytest.importorskip("ragnarok_ai")
    config = stand_in_judge([1.0, 0.0, 0.0]).config("faithfulness")
    assert config.judge_class == f"{rj_judge.RJQwen3.__module__}.{rj_judge.RJQwen3.__qualname__}"
    assert config.config_hash == rj_judge.judge_config("faithfulness").config_hash
    other = rj_judge.judge_config("faithfulness", "0" * 64)
    assert other.config_hash != config.config_hash  # other adapters, another configuration


# ----------------------------------------------------------------------------- real weights


@pytest.mark.skipif(not (os.environ.get("RJ_BASES") and os.environ.get("RJ_ADAPTERS")),
                    reason="needs RJ_BASES=1 and RJ_ADAPTERS naming the retained checkpoint")
def test_scorer_matches_judge_score_and_truncates_long_contexts() -> None:
    import mlx.core as mx

    import judge
    import prompt_templates as pt

    scorer = rj_judge.Scorer(Path(os.environ["RJ_ADAPTERS"]))
    labels = mx.array(judge.label_ids("qwen3", scorer.tokenizer))
    for context, sentence in [("The bridge opened in 1932.", "The bridge opened in 1932."),
                              ("The bridge opened in 1932.", "The bridge opened in 1950.")]:
        probs, tokens, truncated = scorer.probabilities(context, sentence)
        ids = pt.build_input("qwen3", scorer.tokenizer, context, sentence)[0]
        assert probs[0] == judge.score(scorer.model, mx.array([ids]), labels)
        assert abs(sum(probs) - 1.0) < 1e-6 and tokens == len(ids) and not truncated
    probs, tokens, truncated = scorer.probabilities("word " * 6000, "A short sentence.")
    assert truncated and tokens <= pt.MAX_LENGTH

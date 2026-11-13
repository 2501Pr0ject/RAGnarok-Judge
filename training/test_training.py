"""Tests of the sampler, the three-logit loss, the schedule and the resumption of a run.

    python -m pytest training/test_training.py -q

They need no data file and no pretrained weights: the model is a small stand-in with the same
interface as the bases (``model.model`` returns the hidden states, the output head is tied to
the embeddings). With ``RJ_BASES=1`` one more test loads the two pretrained bases from the
local cache and compares the head applied to the last position with the full forward pass.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import sampler

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

import judge  # noqa: E402
import train  # noqa: E402

LABELS = {f"s{k:02d}": "SUPPORTED" for k in range(40)}
LABELS |= {f"u{k:02d}": "UNSUPPORTED" for k in range(9)}
LABELS |= {f"c{k:02d}": "CONTRADICTED" for k in range(3)}
VOCAB, WIDTH = 50, 16
LABEL_TOKENS = [7, 21, 33]


# ----------------------------------------------------------------------------- sampler


def test_epoch_is_reproducible_and_complete() -> None:
    first = sampler.epoch_ids(LABELS, 1, seed=13, supported=20)
    assert first == sampler.epoch_ids(dict(reversed(LABELS.items())), 1, seed=13, supported=20)
    assert len(first) == len(set(first)) == 32 == sampler.epoch_size(LABELS, 20)
    assert Counter(LABELS[i] for i in first) == {"SUPPORTED": 20, "UNSUPPORTED": 9, "CONTRADICTED": 3}


def test_supported_draw_is_renewed_at_every_epoch_and_depends_on_the_seed() -> None:
    def supported(epoch: int, seed: int) -> set[str]:
        return {i for i in sampler.epoch_ids(LABELS, epoch, seed=seed, supported=20) if LABELS[i] == "SUPPORTED"}

    assert supported(1, 13) != supported(2, 13)
    assert supported(1, 13) != supported(1, 14)
    assert sampler.epoch_ids(LABELS, 2, seed=13, supported=20) != sampler.epoch_ids(LABELS, 1, seed=13, supported=20)


def test_stream_from_any_position_matches_the_whole_run() -> None:
    whole = list(sampler.examples_from(LABELS, 0, epochs=2, seed=13, supported=20))
    assert [p for p, _ in whole] == list(range(64))
    assert [i for _, i in whole] == sampler.epoch_ids(LABELS, 1, 13, 20) + sampler.epoch_ids(LABELS, 2, 13, 20)
    for start in (0, 1, 31, 32, 33, 63, 64):
        assert list(sampler.examples_from(LABELS, start, epochs=2, seed=13, supported=20)) == whole[start:]


def test_frozen_sizes_of_the_plan() -> None:
    labels = {f"s{k}": "SUPPORTED" for k in range(49874)}
    labels |= {f"u{k}": "UNSUPPORTED" for k in range(3699)} | {f"c{k}": "CONTRADICTED" for k in range(917)}
    config = train.Config(base="pleias", learning_rate=1e-4)
    size = sampler.epoch_size(labels)
    assert size == 13848
    assert config.total_updates(size) == 3462
    assert config.checkpoints(size) == [866, 1731, 2597, 3462]
    assert config.warmup(size) == 173
    ids = sampler.epoch_ids(labels, 1)
    assert Counter(labels[i] for i in ids) == {"SUPPORTED": 9232, "UNSUPPORTED": 3699, "CONTRADICTED": 917}


def test_learning_rate_schedule() -> None:
    rate = lambda update: train.learning_rate(update, 3e-4, 3462, 173)  # noqa: E731
    assert rate(1) == pytest.approx(3e-4 / 173)
    assert rate(173) == pytest.approx(3e-4)
    assert rate(173 + (3462 - 173) // 2) == pytest.approx(1.5e-4, rel=1e-3)
    assert rate(3462) == pytest.approx(0.0, abs=1e-12)
    assert all(rate(u) >= rate(u + 1) for u in range(173, 3462))


# ----------------------------------------------------------------------------- loss


class Body(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, WIDTH)
        self.mix = nn.Linear(WIDTH, WIDTH)

    def __call__(self, tokens: mx.array) -> mx.array:
        x = self.embed_tokens(tokens)
        steps = mx.arange(1, tokens.shape[1] + 1)[None, :, None]
        return mx.tanh(self.mix(mx.cumsum(x, axis=1) / steps))


class StandIn(nn.Module):
    """Same interface as the bases: hidden states from ``model``, head tied to the embeddings."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        mx.random.seed(seed)
        self.model = Body()
        mx.eval(self.parameters())

    @property
    def layers(self) -> list:
        return []


def reference(model: StandIn, tokens: list[int]) -> np.ndarray:
    """Probabilities of the three labels, computed in numpy from the logits of the whole vocabulary."""
    embed = np.array(model.model.embed_tokens.weight, dtype=np.float64)
    weight = np.array(model.model.mix.weight, dtype=np.float64)
    bias = np.array(model.model.mix.bias, dtype=np.float64)
    x = np.cumsum(embed[tokens], axis=0) / np.arange(1, len(tokens) + 1)[:, None]
    hidden = np.tanh(x @ weight.T + bias)
    logits = (hidden @ embed.T)[-1]  # every position, every token; the last position is read
    three = logits[LABEL_TOKENS].astype(np.float32).astype(np.float64)
    return np.exp(three - three.max()) / np.exp(three - three.max()).sum()


@pytest.mark.parametrize("tokens", [[3, 7, 11, 2], [5], [9, 9, 4, 44, 1, 30, 21, 6, 8]])
def test_three_logit_loss_and_score_match_the_reference(tokens: list[int]) -> None:
    model = StandIn()
    labels = mx.array(LABEL_TOKENS)
    expected = reference(model, tokens)
    assert judge.score(model, mx.array([tokens]), labels) == pytest.approx(expected[0], abs=1e-5)
    for target in range(3):
        value = judge.loss(model, mx.array([tokens]), mx.array([target]), labels).item()
        assert value == pytest.approx(-np.log(expected[target]), abs=1e-5)
    assert judge.label_logits(model, mx.array([tokens]), labels).dtype == mx.float32


def test_loss_ignores_every_other_token_of_the_vocabulary() -> None:
    model = StandIn()
    tokens, labels = mx.array([[3, 7, 11, 2]]), mx.array(LABEL_TOKENS)
    before = judge.loss(model, tokens, mx.array([2]), labels).item()
    rows = np.array(model.model.embed_tokens.weight)
    other = [k for k in range(VOCAB) if k not in LABEL_TOKENS and k not in (3, 7, 11, 2)]
    rows[other] += 5.0
    model.model.embed_tokens.weight = mx.array(rows)
    assert judge.loss(model, tokens, mx.array([2]), labels).item() == pytest.approx(before, abs=1e-6)


@pytest.mark.skipif(not os.environ.get("RJ_BASES"), reason="set RJ_BASES=1 to load the two pretrained bases")
@pytest.mark.parametrize("base", ["pleias", "qwen3"])
def test_head_on_last_position_equals_full_forward_on_the_bases(base: str) -> None:
    model, tokenizer = judge.load_base(base)
    labels = judge.label_ids(base, tokenizer)
    record = {"id": "x", "context": "The bridge opened in 1932. It is 503 metres long.", "sentence": "It opened in 1932."}
    tokens = mx.array([judge.prompt_ids(base, tokenizer, record)])
    full = model(tokens)[0, -1, :][mx.array(labels)].astype(mx.float32)
    last = judge.label_logits(model, tokens, mx.array(labels))[0]
    assert np.array(mx.abs(full - last).max()).item() <= 0.125  # one step of bfloat16 at this magnitude
    assert int(mx.argmax(full).item()) == int(mx.argmax(last).item())


# ----------------------------------------------------------------------------- run and resumption


def example(example_id: str) -> tuple[list[int], int]:
    rng = np.random.default_rng(sum(map(ord, example_id)))
    return rng.integers(0, VOCAB, size=int(rng.integers(3, 12))).tolist(), judge.TARGET[LABELS[example_id]]


CONFIG = train.Config(base="stand-in", learning_rate=1e-2, accumulation=2, supported_per_epoch=20,
                      resume_every=3, grad_checkpoint=False)  # 32 examples and 16 updates per epoch, 32 in all


def launch(out: Path, stop_after: int | None = None, score_dev=None) -> tuple[StandIn, list[str], dict]:
    model, seen = StandIn(), []

    def recorded(example_id: str) -> tuple[list[int], int]:
        seen.append(example_id)
        return example(example_id)

    result = train.run(model, LABELS, recorded, LABEL_TOKENS, CONFIG, out, stop_after=stop_after,
                       score_dev=score_dev, swap_margin_mb=None)
    return model, seen, result


def test_resumed_run_sees_the_same_examples_and_ends_on_the_same_weights(tmp_path: Path) -> None:
    whole_model, whole_seen, whole = launch(tmp_path / "whole")
    assert whole["status"] == "completed" and whole["updates_done"] == 32 and whole["resumed_from"] == []
    assert whole_seen == [i for _, i in sampler.examples_from(LABELS, 0, 2, CONFIG.seed, 20)]

    _, first_seen, first = launch(tmp_path / "cut", stop_after=11)
    assert first["updates_done"] == 11
    resumed_model, second_seen, second = launch(tmp_path / "cut")
    assert second["status"] == "completed" and second["resumed_from"] == [11]
    assert first_seen + second_seen == whole_seen
    for (name, a), (_, b) in zip(
        sorted(train.tree_flatten(whole_model.parameters())), sorted(train.tree_flatten(resumed_model.parameters())),
        strict=True,
    ):
        assert np.allclose(np.array(a), np.array(b), atol=1e-6), name
    rows = [json.loads(line) for line in (tmp_path / "cut" / "train_log.jsonl").read_text().splitlines()]
    assert [row["update"] for row in rows] == list(range(1, 33))
    assert all(np.isfinite(row["loss"]) and row["grad_norm"] >= 0 for row in rows)


def test_checkpoints_are_saved_at_half_epochs_and_scored_once(tmp_path: Path) -> None:
    calls: list[str] = []

    def score_dev(folder: Path) -> None:
        calls.append(folder.name)
        if len(calls) == 2:
            raise KeyboardInterrupt  # interruption during the dev pass of the second checkpoint
        (folder / "dev.json").write_text("{}")

    with pytest.raises(KeyboardInterrupt):
        launch(tmp_path / "run", score_dev=score_dev)
    _, seen, result = launch(tmp_path / "run", score_dev=score_dev)
    assert calls == ["checkpoint-0008", "checkpoint-0016", "checkpoint-0016", "checkpoint-0024", "checkpoint-0032"]
    assert result["resumed_from"] == [16] and len(seen) == 32
    assert sorted(p.name for p in (tmp_path / "run").glob("checkpoint-*")) == [
        "checkpoint-0008", "checkpoint-0016", "checkpoint-0024", "checkpoint-0032"]


def test_a_run_folder_refuses_another_configuration(tmp_path: Path) -> None:
    launch(tmp_path / "run", stop_after=3)
    other = train.Config(**{**train.asdict(CONFIG), "learning_rate": 3e-2})
    with pytest.raises(SystemExit):
        train.run(StandIn(), LABELS, example, LABEL_TOKENS, other, tmp_path / "run", swap_margin_mb=None)


def test_a_pilot_writes_no_checkpoint(tmp_path: Path) -> None:
    _, seen, result = launch(tmp_path / "pilot", stop_after=20)
    assert result["status"] == "stopped after 20 updates" and len(seen) == 40
    assert not list((tmp_path / "pilot").glob("checkpoint-*")) and (tmp_path / "pilot" / "resume").is_dir()

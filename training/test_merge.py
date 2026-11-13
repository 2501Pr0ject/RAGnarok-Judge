"""Tests of the merging of the adapters and of the equivalence check.

    python -m pytest training/test_merge.py -q

No data file and no pretrained weights: a small stand-in model carrying one LoRA module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.tuner.lora import LoRALinear  # noqa: E402

import judge  # noqa: E402
import merge  # noqa: E402
from test_training import LABEL_TOKENS, StandIn  # noqa: E402

PROMPTS = [[3, 7, 11, 2], [5], [9, 9, 4, 44, 1, 30, 21, 6, 8]]


def adapted(dtype: mx.Dtype = mx.float32) -> StandIn:
    model = StandIn()
    model.update(merge.tree_unflatten([(name, value.astype(dtype)) for name, value in tree_flatten(model.parameters())]))
    model.freeze()
    lora = LoRALinear.from_base(model.model.mix, r=4, scale=2.0)
    mx.random.seed(3)
    lora.lora_b = mx.random.normal(lora.lora_b.shape) * 0.3
    model.model.mix = lora
    mx.eval(model.parameters())
    return model


def scores(model: StandIn) -> list[float]:
    return [judge.score(model, mx.array([tokens]), mx.array(LABEL_TOKENS)) for tokens in PROMPTS]


def test_merged_model_gives_the_scores_of_the_separate_adapters() -> None:
    model = adapted()
    plain = scores(StandIn())
    before = scores(model)
    assert max(abs(a - b) for a, b in zip(before, plain, strict=True)) > 1e-3  # the adapters do change the scores
    assert merge.merge(model) == 1
    assert scores(model) == pytest.approx(before, abs=1e-5)
    assert not any("lora" in name for name, _ in tree_flatten(model.parameters()))
    assert merge.merge(model) == 0


def test_merged_weight_keeps_the_precision_of_the_base_and_its_bias() -> None:
    model = adapted(mx.bfloat16)
    module = model.model.mix
    merged = merge.merged_linear(module)
    assert merged.weight.dtype == mx.bfloat16 and merged.weight.shape == module.linear.weight.shape
    assert np.array_equal(np.array(merged.bias.astype(mx.float32)), np.array(module.linear.bias.astype(mx.float32)))
    exact = np.array(module.linear.weight.astype(mx.float32)) + 2.0 * (np.array(module.lora_a) @ np.array(module.lora_b)).T
    rounded = np.array(mx.array(exact).astype(mx.bfloat16).astype(mx.float32))
    assert np.array_equal(np.array(merged.weight.astype(mx.float32)), rounded)  # one rounding, from float32


def test_adapters_saved_by_a_run_are_loaded_back(tmp_path: Path) -> None:
    model = adapted()
    model.model.mix.unfreeze(keys=["lora_a", "lora_b"])
    mx.save_safetensors(str(tmp_path / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    expected = scores(model)
    other = adapted()
    other.model.mix.unfreeze(keys=["lora_a", "lora_b"])
    other.model.mix.lora_b = mx.zeros(other.model.mix.lora_b.shape)
    assert judge.load_adapters(other, tmp_path) == 2
    assert scores(other) == pytest.approx(expected, abs=1e-6)
    mx.save_safetensors(str(tmp_path / "adapters.safetensors"), {"model.mix.lora_a": model.model.mix.lora_a})
    with pytest.raises(SystemExit):
        judge.load_adapters(other, tmp_path)


def test_equivalence_passes_on_the_merge_in_float32_only() -> None:
    a = {"x": 0.50, "y": 0.90}
    b = {"x": 0.52, "y": 0.90}
    result = merge.equivalence(a, b, {"x": 0.5204, "y": 0.90}, {"x": 0.45, "y": 0.91})
    assert result["passed"] and result["examples"] == 2
    assert result["merge_float32"]["max"] == pytest.approx(0.0004)
    assert result["merged_bfloat16_vs_adapters"]["max"] == pytest.approx(0.05)
    assert result["float32_vs_bfloat16_adapters"]["mean"] == pytest.approx(0.01)
    assert not merge.equivalence(a, b, {"x": 0.53, "y": 0.90}, a)["passed"]
    with pytest.raises(SystemExit):
        merge.equivalence(a, b, {"x": 0.5}, a)

"""Tests of the cost measurement of the trained judge (``cost/measure_cost.py --adapter``).

    python -m pytest training/test_cost_adapter.py -q

With ``RJ_BASES=1`` one test loads Pleias-RAG-350M from the local cache with adapters drawn at
random and checks that the cost script scores a sentence exactly as ``training/judge.py`` does.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cost"))
import measure_cost  # noqa: E402

RECORD = {"id": "x", "context": "The bridge opened in 1932. It is 503 metres long.", "sentence": "It opened in 1932."}


def test_adapter_option_is_refused_without_an_mlx_base(tmp_path: Path) -> None:
    for spec in ("fake", "hhem", "hf:some/repo"):
        args = measure_cost.parse_args(["--judge", spec, "--data", "d.jsonl", "--out", "o", "--adapter", str(tmp_path)])
        with pytest.raises(SystemExit):
            measure_cost.build_judge(args)
    assert measure_cost.parse_args(["--judge", "fake", "--data", "d.jsonl", "--out", "o"]).adapter is None


@pytest.mark.skipif(not os.environ.get("RJ_BASES"), reason="set RJ_BASES=1 to load a pretrained base")
def test_cost_script_scores_the_adapted_base_as_training_does(tmp_path: Path) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    import judge
    import prompt_templates as pt

    pinned = pt.BASES["pleias"]
    model, tokenizer = judge.load_base("pleias")
    judge.add_lora(model)
    mx.random.seed(5)
    weights = {name: mx.random.normal(value.shape) * 0.02 if name.endswith("lora_b") else value
               for name, value in tree_flatten(model.trainable_parameters())}
    mx.save_safetensors(str(tmp_path / "adapters.safetensors"), weights)
    (tmp_path / "state.json").write_text(json.dumps({"config": {"base": "pleias", "seed": 13}, "updates_done": 7}))
    judge.load_adapters(model, tmp_path)
    model.eval()
    labels = mx.array(judge.label_ids("pleias", tokenizer))
    tokens = judge.prompt_ids("pleias", tokenizer, RECORD)
    plain = judge.score(judge.load_base("pleias")[0], mx.array([tokens]), labels)
    expected = judge.score(model, mx.array([tokens]), labels)
    assert abs(expected - plain) > 1e-6  # the adapters do change the score
    del model

    measured = measure_cost.MlxBase(pinned["repo"], pinned["revision"], "auto", 4096, tmp_path)
    score, length, truncated = measured.call(RECORD)
    assert (score, length, truncated) == (expected, len(tokens), False)
    described = measured.describe()
    assert described["adapters"] == "separate, not merged" and described["adapter_update"] == 7
    assert described["labels"] == 3 and described["template"] == "declared (pleias)"
    assert described["dtype"] == "mlx.core.bfloat16"
    with pytest.raises(SystemExit):
        measure_cost.MlxBase(pinned["repo"], "0" * 40, "auto", 4096, tmp_path)

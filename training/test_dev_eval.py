"""Tests of the dev criterion and of the tie-break rule.

The reproduction test recomputes the selection criterion of the two baselines from their dev
scores and compares it with the reference values of TARGETS.md (sections 12 and 13) and of
``results/ragtruth/analysis-minicheck-vs-hhem.json``. It needs the frozen dev file and the dev
runs of the two baselines, given by three environment variables; without them it is skipped:

    RJ_DEV=ragtruth-out/dev.jsonl RJ_RUN_HHEM=bench-1.13/full/hhem RJ_RUN_MINICHECK=bench-1.13/full/minicheck \
        python -m pytest training/test_dev_eval.py -q

Only the dev set is read.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import dev_eval

DEV = os.environ.get("RJ_DEV")
RUNS = {"hhem": os.environ.get("RJ_RUN_HHEM"), "minicheck": os.environ.get("RJ_RUN_MINICHECK")}
REFERENCE = Path(__file__).resolve().parents[1] / "results" / "ragtruth" / "analysis-minicheck-vs-hhem.json"
# TARGETS.md, sections 12 and 13: criterion [interval], UNSUPPORTED, CONTRADICTED
TARGETS = {
    "hhem": (0.6406, 0.6155, 0.6670, 0.744, 0.537),
    "minicheck": (0.6185, 0.5915, 0.6486, 0.694, 0.543),
}


def test_tie_break_rule() -> None:
    assert dev_eval.tie_break(0.700, 0.705, 0.012) == "pleias"  # Qwen3 ahead by less than one standard error
    assert dev_eval.tie_break(0.705, 0.700, 0.012) == "pleias"
    assert dev_eval.tie_break(0.700, 0.720, 0.012) == "qwen3"  # ahead by more
    assert dev_eval.tie_break(0.720, 0.700, 0.012) == "pleias"
    assert dev_eval.tie_break(0.700, 0.712, 0.012) == "qwen3"  # exactly one standard error is not "less than"
    assert dev_eval.tie_break(0.700, 0.700, 0.0) == "pleias"  # exact tie


def synthetic() -> tuple[dict, dict, dict]:
    import numpy as np

    rng = np.random.default_rng(0)
    dev, good, noise = {}, {}, {}
    for k in range(600):
        label = "SUPPORTED" if k % 10 < 8 else ("UNSUPPORTED" if k % 10 == 8 else "CONTRADICTED")
        dev[f"d{k}"] = {"id": f"d{k}", "label": label, "source_id": f"g{k // 6}"}
        good[f"d{k}"] = float(rng.normal(1.0 if label == "SUPPORTED" else -1.0))
        noise[f"d{k}"] = float(rng.normal())
    return dev, good, noise


def test_criterion_on_synthetic_scores() -> None:
    dev, good, noise = synthetic()
    perfect = {i: 1.0 if r["label"] == "SUPPORTED" else 0.0 for i, r in dev.items()}
    assert dev_eval.point(dev, perfect) == {"criterion": 1.0, "pauc_U": 1.0, "pauc_C": 1.0}
    assert dev_eval.point(dev, good)["criterion"] > 0.7 > dev_eval.point(dev, noise)["criterion"]
    result = dev_eval.criterion(dev, {"a": good, "b": noise, "c": good}, resamples=200)
    assert result["judges"]["a"]["criterion"]["value"] == dev_eval.point(dev, good)["criterion"]
    assert result["differences"]["a-c"]["value"] == 0.0 and result["differences"]["a-c"]["se"] == 0.0
    assert result["differences"]["a-b"]["se"] > 0


def test_selection_keeps_the_less_costly_base_within_one_standard_error() -> None:
    dev, good, noise = synthetic()
    same = dev_eval.select_base(dev, good, dict(good), resamples=200)["selection"]
    assert same["difference"] == 0.0 and same["retained"] == "pleias"
    assert dev_eval.select_base(dev, noise, good, resamples=200)["selection"]["retained"] == "qwen3"
    assert dev_eval.select_base(dev, good, noise, resamples=200)["selection"]["retained"] == "pleias"


@pytest.mark.skipif(not (DEV and all(RUNS.values())), reason="set RJ_DEV, RJ_RUN_HHEM and RJ_RUN_MINICHECK")
def test_reference_values_of_the_baselines_are_reproduced() -> None:
    dev = dev_eval.ra.read_split(Path(DEV))
    sha = dev_eval.ra.sha256(Path(DEV))
    scores = {name: dev_eval.read_scores(Path(run) / "dev_attempts.jsonl", dev, sha) for name, run in RUNS.items()}
    result = dev_eval.criterion(dev, scores)  # 10000 resamples by source, seed 7
    assert result["sources"] == 163

    for name, (value, low, high, unsupported, contradicted) in TARGETS.items():
        block = result["judges"][name]
        assert round(block["criterion"]["value"], 4) == value
        assert (round(block["criterion"]["low"], 4), round(block["criterion"]["high"], 4)) == (low, high)
        assert round(block["pauc_U"]["value"], 3) == unsupported
        assert round(block["pauc_C"]["value"], 3) == contradicted
        assert dev_eval.point(dev, scores[name]) == {k: block[k]["value"] for k in ("criterion", "pauc_U", "pauc_C")}
    difference = result["differences"]["hhem-minicheck"]
    assert round(difference["se"], 3) == 0.012

    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))["selection_criterion_dev"]
    for name, key in (("minicheck", "A"), ("hhem", "B")):  # the analysis was run with MiniCheck as A, HHEM as B
        for term in ("criterion", "pauc_U", "pauc_C"):
            for field in ("value", "low", "high", "se"):
                assert result["judges"][name][term][field] == pytest.approx(reference["judges"][key][term][field], abs=1e-12)
    assert difference["value"] == pytest.approx(-reference["differences"]["A-B"]["criterion"]["value"], abs=1e-12)
    assert difference["se"] == pytest.approx(reference["differences"]["A-B"]["criterion"]["se"], abs=1e-12)

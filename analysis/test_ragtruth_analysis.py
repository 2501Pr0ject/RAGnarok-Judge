"""Controls of ``ragtruth_analysis.py`` on runs built apart, never written to ``results/``.

1. The reference judge against itself: Delta_C = Delta_U = 0 in every resample, so P1 (null
   Delta_C <= 0) is not met and P2 (null Delta_U <= -0.05) is met.
2. An oracle candidate, score 1 for SUPPORTED and 0 otherwise on test and dev: P1 and P2 are
   met, and Holm compares the smaller p-value with 0.0125 first, then the other with 0.025.

The controls need the frozen test and dev files and a run of the reference judge on both,
given by three environment variables; without them they are skipped:

    RJ_TEST=ragtruth-out/test.jsonl RJ_DEV=ragtruth-out/dev.jsonl RJ_RUN_B=bench-1.13/full/hhem python3 -m pytest analysis/test_ragtruth_analysis.py -q

The runs are built in a temporary directory. ``RJ_RESAMPLES`` (default 2000) sets the number
of resamples: the conclusions do not depend on it beyond a few hundred, and the full 10000
take about a minute per control.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "ragtruth_analysis.py"
TEST = os.environ.get("RJ_TEST")
DEV = os.environ.get("RJ_DEV")
RUN_B = os.environ.get("RJ_RUN_B")
RESAMPLES = int(os.environ.get("RJ_RESAMPLES", "2000"))

needs_data = pytest.mark.skipif(
    not (TEST and DEV and RUN_B), reason="set RJ_TEST, RJ_DEV and RJ_RUN_B to the frozen files and a reference run"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_oracle(split: Path, out: Path, config_hash: str) -> None:
    """A run whose score is 1 for SUPPORTED sentences and 0 for the others."""
    header = {"header": 1, "config_hash": config_hash, "criterion": "faithfulness", "data_sha256": sha256(split)}
    with out.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header) + "\n")
        for record in read_jsonl(split):
            score = 1.0 if record["label"] == "SUPPORTED" else 0.0
            attempt = {"sample_id": record["id"], "ok": True, "result": {"score": score}, "trace": {}}
            handle.write(json.dumps(attempt) + "\n")


def run_analysis(run_a: Path, run_b: Path, out: Path) -> dict:
    command = [
        sys.executable,
        str(SCRIPT),
        "--test",
        TEST,
        "--dev",
        DEV,
        "--run-a",
        str(run_a),
        "--run-b",
        str(run_b),
        "--name-a",
        "A",
        "--name-b",
        "B",
        "--resamples",
        str(RESAMPLES),
        "--out",
        str(out),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(out.read_text(encoding="utf-8"))


@needs_data
def test_reference_against_itself(tmp_path: Path) -> None:
    copy = tmp_path / "run-b-copy"
    shutil.copytree(RUN_B, copy)
    result = run_analysis(copy, Path(RUN_B), tmp_path / "self.json")
    primary = result["primary_equal_specificity"]
    diff = primary["differences"]["A-B"]
    for key in ("C", "U", "NS"):
        assert diff[key]["value"] == 0.0
        assert diff[key]["low"] == 0.0 and diff[key]["high"] == 0.0 and diff[key]["se"] == 0.0
    holm = primary["holm"]
    assert holm["p_values"]["P1"] == pytest.approx(1.0)
    assert holm["p_values"]["P2"] == pytest.approx(1 / (RESAMPLES + 1))
    assert holm["met"] == {"P1": False, "P2": True}
    assert holm["compared_first"] == "P2"
    assert holm["claim"].startswith("no demonstrated gain on contradictions")
    for key in ("C", "U", "NS"):
        a, b = primary["judges"]["A"][key], primary["judges"]["B"][key]
        assert a == b


@needs_data
def test_oracle_candidate(tmp_path: Path) -> None:
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    write_oracle(Path(TEST), oracle / "attempts.jsonl", "oracle")
    write_oracle(Path(DEV), oracle / "dev_attempts.jsonl", "oracle")
    result = run_analysis(oracle, Path(RUN_B), tmp_path / "oracle.json")
    primary = result["primary_equal_specificity"]
    judge_a = primary["judges"]["A"]
    for key in ("C", "U", "NS", "spec", "precision", "f1"):
        assert judge_a[key]["value"] == pytest.approx(1.0)
    diff = primary["differences"]["A-B"]
    assert diff["C"]["value"] == pytest.approx(1.0 - primary["judges"]["B"]["C"]["value"])
    assert diff["U"]["value"] == pytest.approx(1.0 - primary["judges"]["B"]["U"]["value"])
    assert diff["C"]["low"] > 0 and diff["U"]["low"] > -0.05
    holm = primary["holm"]
    assert holm["p_values"]["P1"] == pytest.approx(1 / (RESAMPLES + 1))
    assert holm["p_values"]["P2"] == pytest.approx(1 / (RESAMPLES + 1))
    first = holm["compared_first"]
    second = "P2" if first == "P1" else "P1"
    assert holm["levels"] == {first: 0.0125, second: 0.025}
    assert holm["p_values"][first] <= holm["p_values"][second]
    assert holm["met"] == {"P1": True, "P2": True}
    assert holm["claim"].startswith("at 96% specificity")
    thresholds = result["thresholds"]["A"]
    assert thresholds["equal_specificity_threshold_test"] == 1.0 and thresholds["dev_threshold"] == 1.0
    assert result["without_threshold_test"]["judges"]["A"]["auc"]["value"] == pytest.approx(1.0)
    assert result["without_threshold_test"]["judges"]["A"]["criterion"]["value"] == pytest.approx(1.0)


def test_holm_order_unit() -> None:
    sys.path.insert(0, str(HERE))
    from ragtruth_analysis import holm

    both = holm(0.001, 0.02)
    assert both["compared_first"] == "P1" and both["met"] == {"P1": True, "P2": True}
    only_second = holm(0.5, 0.001)
    assert only_second["compared_first"] == "P2" and only_second["met"] == {"P1": False, "P2": True}
    blocked = holm(0.02, 0.03)  # the smaller p-value fails at 0.0125, so the other is not tested
    assert blocked["met"] == {"P1": False, "P2": False} and blocked["claim"] == "no claim"
    first_only = holm(0.001, 0.03)  # 0.001 passes at 0.0125, then 0.03 fails at 0.025
    assert first_only["met"] == {"P1": True, "P2": False}

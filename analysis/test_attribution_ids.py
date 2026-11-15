"""Tests of analysis/attribution_ids.py.

    python -m pytest analysis/test_attribution_ids.py -q

The unit tests use synthetic trace lines. With ``RJ_TRACE`` (the converter's trace.jsonl) and
``RJ_TRAIN_RAW`` (train.jsonl, before the exclusion rules), the patterns are applied to the
training split only: they must give the 115 candidates of training set rule 1, the 74 left
after rule 5, and these 74 must hold the 58 ids of ``train_excluded_ids_rule1.txt`` plus the
16 sentences validated and kept (TARGETS.md, section 2). No other split is decoded.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attribution_ids as ai  # noqa: E402

SCRIPT = Path(__file__).resolve().parent / "attribution_ids.py"
CONVERSION = Path(__file__).resolve().parents[1] / "conversion"


def line(sample: str, split: str, label: str, notes: list[str | None]) -> str:
    spans = [{"start": 0, "end": 1, "text": "", "label_type": "Evident Conflict", "implicit_true": False,
              "due_to_null": False, "meta": note} for note in notes]
    return json.dumps({"id": sample, "split": split, "label": label, "spans": spans}) + "\n"


def test_patterns_ignore_case_and_need_a_passage_reference() -> None:
    assert ai.matches("The figure is in Passage 2, not in passage 1.")
    assert ai.matches("PASSAGE 3: states the opposite")
    assert ai.matches("passage 1 doesn't say this")
    assert ai.matches("This is mentioned in passage three")
    assert not ai.matches("The passage does not mention the date.")
    assert not ai.matches("")
    assert not ai.matches(None)


def test_candidates_of_one_split_and_label_only(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        line("t1", "train", "CONTRADICTED", ["the date is in passage 2"])
        + line("t2", "train", "CONTRADICTED", ["the date is wrong", None])
        + line("t3", "train", "CONTRADICTED", [None, "Passage 1: says 1990"])
        + line("t4", "train", "UNSUPPORTED", ["in passage 2"])
        + line("x1", "test", "CONTRADICTED", ["in passage 2"]),
        encoding="utf-8",
    )
    pool, found = ai.candidates(trace, "train")
    assert pool == {"t1", "t2", "t3"} and found == {"t1", "t3"}
    assert ai.candidates(trace, "test") == ({"x1"}, {"x1"})


def test_the_expected_count_is_enforced_and_only_ids_are_written(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(line("a", "test", "CONTRADICTED", ["in passage 2"]) + line("b", "test", "CONTRADICTED", ["no"]),
                     encoding="utf-8")
    out = tmp_path / "ids.txt"
    base = [sys.executable, str(SCRIPT), "--trace", str(trace), "--split", "test", "--out", str(out)]
    failed = subprocess.run([*base, "--expect", "2"], capture_output=True, text=True)
    assert failed.returncode != 0 and not out.exists()
    subprocess.run([*base, "--expect", "1"], check=True, capture_output=True)
    assert out.read_text(encoding="utf-8") == "a\n"
    again = subprocess.run([*base, "--expect", "1"], capture_output=True, text=True)
    assert again.returncode != 0  # never overwritten


@pytest.mark.skipif(not (os.environ.get("RJ_TRACE") and os.environ.get("RJ_TRAIN_RAW")),
                    reason="needs RJ_TRACE and RJ_TRAIN_RAW")
def test_training_candidates_of_rule_1(tmp_path: Path) -> None:
    trace, train = Path(os.environ["RJ_TRACE"]), Path(os.environ["RJ_TRAIN_RAW"])
    rule5 = ai.read_ids(CONVERSION / "train_excluded_ids_rule5.txt")
    rule1 = ai.read_ids(CONVERSION / "train_excluded_ids_rule1.txt")
    pool, found = ai.candidates(trace, "train")
    assert len(found) == 115 and len(found & rule5) == 41
    left = found - rule5
    assert len(left) == 74 and rule1 <= left and len(left - rule1) == 16
    out = tmp_path / "train-ids.txt"
    subprocess.run([sys.executable, str(SCRIPT), "--trace", str(trace), "--split", "train", "--check-split", str(train),
                    "--drop-ids", str(CONVERSION / "train_excluded_ids_rule5.txt"), "--expect", "74", "--out", str(out)],
                   check=True, capture_output=True)
    assert ai.read_ids(out) == left

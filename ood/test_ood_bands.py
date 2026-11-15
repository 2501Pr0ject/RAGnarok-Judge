"""Tests of ood/ood_bands.py on synthetic files.

    python -m pytest ood/test_ood_bands.py -q
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "ood_bands.py"


def write_case(tmp: Path) -> tuple[Path, Path]:
    # WiCE, short band: 2 supported (0.9, 0.4), 2 not supported (0.6, 0.1); ExpertQA, long band: one class only
    rows = [
        ("a", "WiCE", "SUPPORTED", "words_0000_0250", 0.9),
        ("b", "WiCE", "SUPPORTED", "words_0000_0250", 0.4),
        ("c", "WiCE", "UNSUPPORTED", "words_0000_0250", 0.6),
        ("d", "WiCE", "UNSUPPORTED", "words_0000_0250", 0.1),
        ("e", "ExpertQA", "SUPPORTED", "words_2501_plus", 0.7),
    ]
    data = tmp / "data.jsonl"
    data.write_text("".join(json.dumps({"id": i, "dataset": d, "label": lab, "length_band": b, "source_id": i,
                                        "doc_words": 1, "context": "", "sentence": ""}) + "\n"
                            for i, d, lab, b, _ in rows), encoding="utf-8")
    run = tmp / "run"
    run.mkdir()
    (run / "attempts.jsonl").write_text(
        json.dumps({"config_hash": "x"}) + "\n"
        + "".join(json.dumps({"sample_id": i, "ok": True, "result": {"score": s}, "trace": {}}) + "\n"
                  for i, _, _, _, s in rows), encoding="utf-8")
    return data, run


def test_cells_metrics_and_flags(tmp_path: Path) -> None:
    data, run = write_case(tmp_path)
    out = tmp_path / "bands.json"
    subprocess.run([sys.executable, str(SCRIPT), "--data", str(data), "--run", "J", str(run), "0.5", "--out", str(out)],
                   check=True, capture_output=True)
    cells = json.loads(out.read_text())["judges"]["J"]["cells"]
    short = cells["WiCE"]["words_0000_0250"]
    assert (short["n"], short["n_supported"], short["n_not_supported"], short["minority_below_10"]) == (4, 2, 2, True)
    assert short["auc"] == 0.75  # 3 of the 4 supported / not supported pairs are ordered
    at = short["thresholds"]["0.5"]  # flagged: 0.4 and 0.1
    assert (at["recall_not_supported"], at["specificity"], at["balanced_accuracy"]) == (0.5, 0.5, 0.5)
    assert "auc" not in cells["ExpertQA"]["words_2501_plus"] and cells["ExpertQA"]["words_2501_plus"]["n"] == 1
    assert cells["REVEAL"]["words_0251_1000"]["n"] == 0
    again = subprocess.run([sys.executable, str(SCRIPT), "--data", str(data), "--run", "J", str(run), "0.5",
                            "--out", str(out)], capture_output=True, text=True)
    assert again.returncode != 0  # never overwritten

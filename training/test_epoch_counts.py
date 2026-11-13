"""Tests of the counts of meta sentences and abstentions per epoch.

    python -m pytest training/test_epoch_counts.py -q

The sentences are written for the test and appear in neither the training nor the dev file;
no data file is read.
"""

from __future__ import annotations

import epoch_counts
import sampler

SENTENCES = {
    "announcement": "Okay, here is a concise recap of the provided story in 12 words or fewer.",
    "courtesy": "Hope these steps help you out!",
    "abstention": "Unfortunately, I am unable to provide an answer from the provided information.",
    None: "The bridge opened in 1932.",
}


def records() -> dict[str, dict[str, str]]:
    rows = {}
    for k in range(40):
        kind = ("announcement", "courtesy", "abstention", None)[k % 4]
        rows[f"s{k:02d}"] = {"sentence": SENTENCES[kind], "label": "SUPPORTED"}
    for k in range(9):
        rows[f"u{k:02d}"] = {"sentence": SENTENCES["abstention" if k == 0 else None], "label": "UNSUPPORTED"}
    for k in range(3):
        rows[f"c{k:02d}"] = {"sentence": SENTENCES[None], "label": "CONTRADICTED"}
    return rows


def test_categories_follow_the_frozen_lists() -> None:
    found = epoch_counts.categories(records())
    assert found["s00"] == "announcement" and found["s01"] == "courtesy" and found["s02"] == "abstention"
    assert "s03" not in found and found["u00"] == "abstention" and "c00" not in found
    assert epoch_counts.CATEGORIES == ("announcement", "courtesy", "abstention")


def test_counts_of_the_training_file_and_of_each_epoch() -> None:
    rows = records()
    result = epoch_counts.epochs(rows, epochs=2, seed=13, supported=20)
    whole = result["training_file"]
    assert whole["examples"] == 52 and whole["meta"] == 20 and whole["abstention"] == 11
    assert whole["by_category"]["abstention"] == {"total": 11, "SUPPORTED": 10, "UNSUPPORTED": 1, "CONTRADICTED": 0}
    labels = {i: row["label"] for i, row in rows.items()}
    for epoch in (1, 2):
        block = result["epochs"][str(epoch)]
        ids = sampler.epoch_ids(labels, epoch, seed=13, supported=20)
        assert block["examples"] == 32 and block["labels"] == {"SUPPORTED": 20, "UNSUPPORTED": 9, "CONTRADICTED": 3}
        for name in ("announcement", "courtesy", "abstention"):
            expected = sum(rows[i]["sentence"] == SENTENCES[name] for i in ids)
            assert block["by_category"][name]["total"] == expected
        assert block["meta"] == block["by_category"]["announcement"]["total"] + block["by_category"]["courtesy"]["total"]
        assert block["by_category"]["abstention"]["UNSUPPORTED"] == 1  # every UNSUPPORTED sentence is in every epoch

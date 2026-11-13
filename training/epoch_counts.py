"""Meta sentences and abstentions in the examples of each epoch (TRAINING.md, section 2).

The sampler draws on the label only. This script counts, after the fact, how many sentences
of each epoch are matched by the frozen pattern lists of ``analysis/meta_match.py`` (meta:
announcement and courtesy; abstention, apart), by label. The training file only.

    python training/epoch_counts.py --train TRAIN.jsonl [--epochs 2] [--out OUT.json]

Prints counts only, no sentence.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import meta_match  # noqa: E402

import sampler  # noqa: E402

CATEGORIES = (*meta_match.META_PATTERNS, *meta_match.ABSTENTION_PATTERNS)  # announcement, courtesy, abstention
LABELS = ("SUPPORTED", "UNSUPPORTED", "CONTRADICTED")


def categories(records: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """id -> category, for the sentences matched by one of the two lists."""
    found = {}
    for example_id, record in records.items():
        hit = meta_match.meta_category(record["sentence"])
        abstains = meta_match.is_abstention(record["sentence"])
        if hit and abstains:
            raise SystemExit(f"{example_id}: matched by the meta list and by the abstention list")
        if hit:
            found[example_id] = hit[0]
        elif abstains:
            found[example_id] = "abstention"
    return found


def count(ids: Iterable[str], labels: Mapping[str, str], category: Mapping[str, str]) -> dict[str, Any]:
    """Counts over the examples ``ids``: total, per label, and per category and label."""
    ids = list(ids)
    by = {name: dict.fromkeys(LABELS, 0) for name in CATEGORIES}
    per_label = dict.fromkeys(LABELS, 0)
    for example_id in ids:
        per_label[labels[example_id]] += 1
        if example_id in category:
            by[category[example_id]][labels[example_id]] += 1
    totals = {name: sum(by[name].values()) for name in CATEGORIES}
    return {
        "examples": len(ids),
        "labels": per_label,
        "meta": sum(totals[name] for name in meta_match.META_PATTERNS),
        "abstention": totals["abstention"],
        "by_category": {name: {"total": totals[name], **by[name]} for name in CATEGORIES},
    }


def epochs(
    records: Mapping[str, Mapping[str, Any]],
    epochs: int = 2,
    seed: int = sampler.SEED,
    supported: int = sampler.SUPPORTED_PER_EPOCH,
) -> dict[str, Any]:
    labels = {example_id: record["label"] for example_id, record in records.items()}
    category = categories(records)
    result: dict[str, Any] = {
        "patterns_version": meta_match.VERSION,
        "seed": seed,
        "supported_per_epoch": supported,
        "training_file": count(labels, labels, category),
        "epochs": {},
    }
    for epoch in range(1, epochs + 1):
        result["epochs"][str(epoch)] = count(sampler.epoch_ids(labels, epoch, seed, supported), labels, category)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train", type=Path, required=True, help="train-rule5-rule1.jsonl")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out is not None and args.out.exists():
        raise SystemExit(f"{args.out} exists; choose another --out")
    with args.train.open(encoding="utf-8") as handle:
        records = {record["id"]: record for record in map(json.loads, filter(str.strip, handle))}
    result = epochs(records, args.epochs)
    print(f"patterns {result['patterns_version']}, seed {result['seed']}")
    print("| | examples | meta | announcement | courtesy | abstention |")
    print("|---|---|---|---|---|---|")

    def cell(block: dict[str, Any]) -> str:
        return f"{block['total']} ({block['SUPPORTED']} S, {block['UNSUPPORTED']} U, {block['CONTRADICTED']} C)"

    for name, block in [("training file", result["training_file"]), *result["epochs"].items()]:
        by = block["by_category"]
        row_name = name if name == "training file" else f"epoch {name}"
        print(f"| {row_name} | {block['examples']} | {block['meta']} | {cell(by['announcement'])} | "
              f"{cell(by['courtesy'])} | {cell(by['abstention'])} |")
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

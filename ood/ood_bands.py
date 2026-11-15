"""Out-of-domain metrics by band of document length, per dataset (section 7 of TARGETS.md).

Section 7: "the AUC and the balanced accuracies are also reported by band of document length
[...] A metric by band is computed per dataset; cells with fewer than 10 sentences of the
minority class are flagged." This script computes them from existing runs; it scores nothing.

The bands are the ``length_band`` field written by ``ood/convert_llm_aggrefact.py``: words of
the document, up to 250, 251 to 1000, 1001 to 2500, above 2500. For every judge given, for
every dataset and every band, on all inputs: AUC ROC on raw scores (Mann-Whitney, ties counted
one half), and balanced accuracy, recall on not supported claims and specificity at the
judge's RAGTruth dev threshold and at 0.5; a claim is flagged when score < threshold. The
metric functions are those of ``ood/ood_metrics.py``. Point values only, as section 7 asks no
interval for this breakdown. A cell without one of the two classes has no metric.

Prints and writes aggregates only: no text, no label and no score of an instance.

    python ood/ood_bands.py --data ood-out/llm-aggrefact-5.jsonl --run RJ ood-runs/rj-qwen3 0.20575958490371704 --run HHEM ood-runs/hhem-l4096-mps 0.18617 --run MiniCheck ood-runs/minicheck-bounded-mps 0.04438 --out ood-runs/bands.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ood_metrics import DATASETS, MIN_MINORITY, Block, measures, read_data, read_run  # noqa: E402

BANDS = ("words_0000_0250", "words_0251_1000", "words_1001_2500", "words_2501_plus")
BAND_NAMES = {
    "words_0000_0250": "up to 250",
    "words_0251_1000": "251 to 1000",
    "words_1001_2500": "1001 to 2500",
    "words_2501_plus": "above 2500",
}
MIDPOINT = 0.5


def read_bands(path: Path) -> dict[str, str]:
    bands: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record["length_band"] not in BANDS:
                    raise SystemExit(f"unexpected band in {path}: {record['length_band']}")
                bands[record["id"]] = record["length_band"]
    return bands


def cells(records: dict[str, dict[str, Any]], bands: dict[str, str]) -> dict[str, dict[str, list[str]]]:
    """Sorted ids of every dataset and band."""
    out: dict[str, dict[str, list[str]]] = {d: {b: [] for b in BANDS} for d in DATASETS}
    for sample in sorted(records):
        out[records[sample]["dataset"]][bands[sample]].append(sample)
    return out


def metrics(ids: list[str], records: dict[str, dict[str, Any]], scores: dict[str, float],
            thresholds: tuple[float, ...]) -> dict[str, Any]:
    supported = np.array([records[s]["supported"] for s in ids], dtype=bool)
    out: dict[str, Any] = {"n": len(ids), "n_supported": int(supported.sum()), "n_not_supported": int((~supported).sum())}
    minority = min(out["n_supported"], out["n_not_supported"])
    out["minority_below_10"] = minority < MIN_MINORITY
    if minority == 0:
        return out
    block = Block(np.array([scores[s] for s in ids], dtype=np.float64), supported,
                  np.array([records[s]["group"] for s in ids]))
    point = measures(block, np.ones((1, len(ids))), thresholds)
    out["auc"] = float(point["auc"][0])
    out["thresholds"] = {
        repr(t): {m: float(point[f"{m}@{t!r}"][0]) for m in ("balanced_accuracy", "recall_not_supported", "specificity")}
        for t in thresholds
    }
    return out


def fmt(row: dict[str, Any], key: str, threshold: float | None = None) -> str:
    if "auc" not in row:
        return "-"
    value = row["auc"] if threshold is None else row["thresholds"][repr(threshold)][key]
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", nargs=3, action="append", required=True, metavar=("NAME", "RUN", "DEV_THRESHOLD"),
                        help="a judge, its run folder and its RAGTruth dev threshold; repeatable")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; nothing is overwritten")

    data_sha256 = hashlib.sha256(args.data.read_bytes()).hexdigest()
    records = read_data(args.data)
    grid = cells(records, read_bands(args.data))
    judges: dict[str, Any] = {}
    for name, run, dev in args.run:
        scores, truncated, unscored, header = read_run(Path(run))
        if header.get("data_sha256") not in (None, data_sha256):
            raise SystemExit(f"{run} was made on another data file")
        if set(scores) != set(records):
            raise SystemExit(f"{run} and the data do not match")
        thresholds = (float(dev), MIDPOINT)
        judges[name] = {
            "run": run, "config_hash": header.get("config_hash"), "dev_threshold": float(dev),
            "truncated_in_run": len(truncated),
            "cells": {d: {b: metrics(grid[d][b], records, scores, thresholds) for b in BANDS} for d in DATASETS},
        }

    names = [name for name, _, _ in args.run]
    print(f"data {args.data} sha256 {data_sha256}; all inputs; point values")
    print("AUC: probability that a supported claim scores above a not supported one, ties one half")
    print("a claim is flagged when score < threshold; a cell is flagged when its minority class has fewer than 10 claims")
    for dataset in DATASETS:
        print(f"\n{dataset}")
        heads = ["Band, words", "n", "Supported", "Not supported", "Flag", *(f"AUC {n}" for n in names),
                 *(f"BAcc {n} at dev threshold" for n in names), *(f"BAcc {n} at 0.5" for n in names)]
        print("| " + " | ".join(heads) + " |")
        print("|" + "---|" * len(heads))
        for band in BANDS:
            first = judges[names[0]]["cells"][dataset][band]
            row = [BAND_NAMES[band], str(first["n"]), str(first["n_supported"]), str(first["n_not_supported"]),
                   "minority below 10" if first["minority_below_10"] else ""]
            row += [fmt(judges[n]["cells"][dataset][band], "auc") for n in names]
            row += [fmt(judges[n]["cells"][dataset][band], "balanced_accuracy", judges[n]["dev_threshold"]) for n in names]
            row += [fmt(judges[n]["cells"][dataset][band], "balanced_accuracy", MIDPOINT) for n in names]
            print("| " + " | ".join(row) + " |")

    result = {"data": str(args.data), "data_sha256": data_sha256, "inputs": "all", "bands": BAND_NAMES,
              "min_minority": MIN_MINORITY, "judges": judges}
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=1) + "\n")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()

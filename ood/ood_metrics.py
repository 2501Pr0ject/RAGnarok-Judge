"""Out-of-domain metrics of one judge on the five datasets of LLM-AggreFact.

Reads the converted file and the ``attempts.jsonl`` of one ``ragnarok judge-bench`` run.
Prints and writes aggregates only: no text, no label and no score of an instance leaves
the script.

Per dataset, then the unweighted mean over the five datasets:

    AUC ROC on raw scores (Mann-Whitney, ties counted one half)
    balanced accuracy, recall on not supported claims and specificity at every threshold
    given; a claim is flagged when score < threshold

on all inputs, on the inputs that no listed run truncates, and, as a description only, on
the truncated inputs.

Intervals: percentile, 95%, 10000 resamples, seed 7. Documents (``source_id``) are drawn
with replacement within each dataset. The draws depend on the data file and on the set of
inputs only, never on the scores: they are the same for every judge.

With ``--published`` and ``--judge``, the balanced accuracies at 0.5 are compared with the
published figures and every gap above two points is flagged. This is a check of the
pipeline, not a target.

    python ood/ood_metrics.py --data ood-out/llm-aggrefact-5.jsonl --run ood-runs/hhem-l4096-mps --thresholds 0.18617,0.5 --published ood/published_bacc.json --judge hhem --out ood-runs/metrics-hhem-l4096-mps.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

DATASETS = ("WiCE", "REVEAL", "ClaimVerify", "FactCheck-GPT", "ExpertQA")
LABELS = {"SUPPORTED": True, "UNSUPPORTED": False}
RESAMPLES = 10000
SEED = 7
CHUNK = 250
MIN_MINORITY = 10
MAX_GAP_POINTS = 2.0
# third element of the seed of a resampling stream
ALL, UNTRUNCATED = 0, 1


def read_data(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["dataset"] not in DATASETS:
                raise SystemExit(f"unexpected dataset in {path}: {record['dataset']}")
            if record["id"] in records:
                raise SystemExit(f"duplicate id in {path}: {record['id']}")
            records[record["id"]] = {
                "dataset": record["dataset"],
                "group": record["source_id"],
                "supported": LABELS[record["label"]],
            }
    return records


def read_run(run: Path) -> tuple[dict[str, float], set[str], int, dict[str, Any]]:
    """Scores, truncated ids, number of samples without a score, and header of a run.

    The last successful attempt of a sample is kept.
    """
    scores: dict[str, float] = {}
    truncated: set[str] = set()
    seen: set[str] = set()
    header: dict[str, Any] = {}
    with (run / "attempts.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            if "sample_id" not in entry:
                header = entry
                continue
            sample = entry["sample_id"]
            seen.add(sample)
            if not entry.get("ok") or entry.get("result") is None:
                continue
            scores[sample] = float(entry["result"]["score"])
            if (entry.get("trace") or {}).get("context_truncated"):
                truncated.add(sample)
            else:
                truncated.discard(sample)
    return scores, truncated, len(seen - set(scores)), header


class Block:
    """The instances of one dataset, sorted by score, with their tie blocks and documents."""

    def __init__(self, scores: np.ndarray, supported: np.ndarray, groups: np.ndarray) -> None:
        order = np.argsort(scores, kind="stable")
        self.score = scores[order]
        self.pos = supported[order].astype(np.float64)
        self.neg = 1.0 - self.pos
        # documents are numbered in the order of their names, whatever the scores
        names, self.group = np.unique(groups[order], return_inverse=True)
        self.n_groups = len(names)
        change = np.flatnonzero(np.diff(self.score) != 0) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [len(self.score)]))
        block = np.repeat(np.arange(len(starts)), ends - starts)
        self.lo = starts[block]  # first instance of the tie block
        self.hi = ends[block]  # one past its last instance


def measures(block: Block, weights: np.ndarray, thresholds: tuple[float, ...]) -> dict[str, np.ndarray]:
    """Metrics for each row of ``weights``, the multiplicities of the instances in a resample."""
    wp = weights * block.pos
    wn = weights * block.neg
    total_pos, total_neg = wp.sum(axis=1), wn.sum(axis=1)
    cum = np.concatenate((np.zeros((weights.shape[0], 1)), np.cumsum(wn, axis=1)), axis=1)
    below = cum[:, block.lo]  # not supported claims scored strictly lower
    tied = cum[:, block.hi] - below  # not supported claims with the same score
    out: dict[str, np.ndarray] = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        out["auc"] = (wp * (below + 0.5 * tied)).sum(axis=1) / (total_pos * total_neg)
        for threshold in thresholds:
            flagged = block.score < threshold
            recall = (wn * flagged).sum(axis=1) / total_neg
            specificity = (wp * ~flagged).sum(axis=1) / total_pos
            out[f"balanced_accuracy@{threshold!r}"] = (recall + specificity) / 2
            out[f"recall_not_supported@{threshold!r}"] = recall
            out[f"specificity@{threshold!r}"] = specificity
    return out


def resample(block: Block, thresholds: tuple[float, ...], key: list[int]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(key)
    share = np.full(block.n_groups, 1.0 / block.n_groups)
    parts = []
    for start in range(0, RESAMPLES, CHUNK):
        counts = rng.multinomial(block.n_groups, share, size=min(CHUNK, RESAMPLES - start))
        parts.append(measures(block, counts[:, block.group].astype(np.float64), thresholds))
    return {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}


def summary(value: float, draws: np.ndarray) -> dict[str, Any]:
    valid = draws[~np.isnan(draws)]
    low, high = np.percentile(valid, [2.5, 97.5])
    return {"value": float(value), "low": float(low), "high": float(high), "excluded": int(len(draws) - len(valid))}


def nest(flat: dict[str, Any]) -> dict[str, Any]:
    """{"auc": x, "specificity@0.5": y} -> {"auc": x, "thresholds": {"0.5": {"specificity": y}}}."""
    out: dict[str, Any] = {"thresholds": {}}
    for name, value in flat.items():
        if "@" in name:
            metric, threshold = name.split("@")
            out["thresholds"].setdefault(threshold, {})[metric] = value
        else:
            out[name] = value
    return out


def analyse(
    ids: list[str],
    records: dict[str, dict[str, Any]],
    scores: dict[str, float],
    thresholds: tuple[float, ...],
    stream: int | None,
) -> dict[str, Any]:
    """Metrics per dataset and their unweighted mean. Without ``stream``, point values only."""
    result: dict[str, Any] = {}
    points: dict[str, list[float]] = {}
    draws: dict[str, list[np.ndarray]] = {}
    for index, dataset in enumerate(DATASETS):
        chosen = [sample for sample in ids if records[sample]["dataset"] == dataset]
        supported = np.array([records[sample]["supported"] for sample in chosen], dtype=bool)
        counts = {
            "n": len(chosen),
            "n_supported": int(supported.sum()),
            "n_not_supported": int((~supported).sum()),
            "documents": len({records[sample]["group"] for sample in chosen}),
        }
        counts["minority_below_10"] = min(counts["n_supported"], counts["n_not_supported"]) < MIN_MINORITY
        if min(counts["n_supported"], counts["n_not_supported"]) == 0:
            result[dataset] = counts
            continue
        block = Block(
            np.array([scores[sample] for sample in chosen], dtype=np.float64),
            supported,
            np.array([records[sample]["group"] for sample in chosen]),
        )
        point = {
            name: float(value[0]) for name, value in measures(block, np.ones((1, len(chosen))), thresholds).items()
        }
        if stream is None:
            result[dataset] = {**counts, **nest(point)}
            continue
        drawn = resample(block, thresholds, [SEED, index, stream])
        result[dataset] = {**counts, **nest({name: summary(point[name], drawn[name]) for name in point})}
        for name, value in point.items():
            points.setdefault(name, []).append(value)
            draws.setdefault(name, []).append(drawn[name])
    if stream is not None and all(len(values) == len(DATASETS) for values in points.values()) and points:
        # a resample where one dataset has no value is excluded: the mean of NaN is NaN
        result["mean"] = nest(
            {name: summary(float(np.mean(points[name])), np.mean(np.vstack(draws[name]), axis=0)) for name in points}
        )
    return result


def cell(entry: dict[str, Any] | float | None) -> str:
    if entry is None:
        return "-"
    if isinstance(entry, float):
        return f"{entry:.4f}"
    return f"{entry['value']:.4f} [{entry['low']:.4f}, {entry['high']:.4f}]"


def show(title: str, table: dict[str, Any], thresholds: tuple[float, ...]) -> None:
    print(f"\n{title}")
    heads = ["Dataset", "n", "Supported", "Not supported", "Documents", "AUC"]
    for threshold in thresholds:
        heads += [f"BAcc at {threshold!r}", f"Recall at {threshold!r}", f"Specificity at {threshold!r}"]
    print("| " + " | ".join(heads) + " |")
    print("|" + "---|" * len(heads))
    for name in (*DATASETS, "mean"):
        row = table.get(name)
        if row is None:
            continue
        flag = " (minority below 10)" if row.get("minority_below_10") and row.get("n") else ""
        cells = [
            ("Mean of the five" if name == "mean" else name) + flag,
            *(str(row.get(key, "")) for key in ("n", "n_supported", "n_not_supported", "documents")),
            cell(row.get("auc")),
        ]
        for threshold in thresholds:
            at = row.get("thresholds", {}).get(repr(threshold), {})
            cells += [cell(at.get(key)) for key in ("balanced_accuracy", "recall_not_supported", "specificity")]
        print("| " + " | ".join(cells) + " |")
    excluded = sum(
        entry.get("excluded", 0)
        for row in table.values()
        for entry in [row.get("auc"), *(v for at in row.get("thresholds", {}).values() for v in at.values())]
        if isinstance(entry, dict)
    )
    print(f"resamples excluded for a missing class, all cells together: {excluded}")


def published_check(table: dict[str, Any], published: dict[str, Any], judge: str) -> dict[str, Any]:
    figures = published["judges"][judge]
    rows = {}
    for dataset in DATASETS:
        ours = 100 * table[dataset]["thresholds"]["0.5"]["balanced_accuracy"]["value"]
        gap = ours - figures[dataset]
        rows[dataset] = {
            "ours": round(ours, 2),
            "published": figures[dataset],
            "gap_points": round(gap, 2),
            "flagged": bool(abs(gap) > MAX_GAP_POINTS),
        }
    ours_mean = float(np.mean([row["ours"] for row in rows.values()]))
    published_mean = float(np.mean([row["published"] for row in rows.values()]))
    rows["mean"] = {
        "ours": round(ours_mean, 2),
        "published": round(published_mean, 2),
        "gap_points": round(ours_mean - published_mean, 2),
        "flagged": bool(abs(ours_mean - published_mean) > MAX_GAP_POINTS),
    }
    return {"entry": figures["name"], "source": published["source"], "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--thresholds", required=True, help="comma-separated, for instance 0.18617,0.5")
    parser.add_argument(
        "--truncation-run",
        type=Path,
        action="append",
        help="run whose truncated inputs leave the untruncated subset; repeatable; default: --run",
    )
    parser.add_argument("--published", type=Path)
    parser.add_argument("--judge", help="key of the judge in the --published file")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    thresholds = tuple(float(value) for value in args.thresholds.split(","))

    data_sha256 = hashlib.sha256(args.data.read_bytes()).hexdigest()
    records = read_data(args.data)
    scores, truncated_here, unscored, header = read_run(args.run)
    if header.get("data_sha256") not in (None, data_sha256):
        raise SystemExit(f"the run was made on another data file: {header.get('data_sha256')} != {data_sha256}")
    unknown = set(scores) - set(records)
    missing = set(records) - set(scores)
    if unknown or missing:
        raise SystemExit(f"run and data do not match: {len(unknown)} unknown ids, {len(missing)} ids without score")
    ids = sorted(records)

    truncated: set[str] = set()
    for run in args.truncation_run or [args.run]:
        truncated |= truncated_here if run == args.run else read_run(run)[1]
    kept = [sample for sample in ids if sample not in truncated]
    cut = [sample for sample in ids if sample in truncated]

    print(f"data {args.data} sha256 {data_sha256}")
    print(
        f"run {args.run}: {len(scores)} scored of {len(records)}, {unscored} without score, {len(truncated_here)} truncated"
    )
    print(f"{RESAMPLES} resamples by document within each dataset, seed {SEED}, percentile intervals at 95%")
    print("AUC: probability that a supported claim scores above a not supported one, ties one half")
    print("a claim is flagged when score < threshold; recall is on not supported claims")

    result: dict[str, Any] = {
        "data": str(args.data),
        "data_sha256": data_sha256,
        "run": str(args.run),
        "config_hash": header.get("config_hash"),
        "scored": len(scores),
        "truncated_in_run": len(truncated_here),
        "truncation_runs": [str(run) for run in args.truncation_run or [args.run]],
        "resamples": RESAMPLES,
        "seed": SEED,
        "thresholds": list(thresholds),
        "all": analyse(ids, records, scores, thresholds, ALL),
        "untruncated": analyse(kept, records, scores, thresholds, UNTRUNCATED),
        "truncated_only": analyse(cut, records, scores, thresholds, None) if cut else {},
    }
    show(f"All inputs ({len(ids)})", result["all"], thresholds)
    show(f"Inputs not truncated ({len(kept)})", result["untruncated"], thresholds)
    if cut:
        show(f"Truncated inputs only ({len(cut)}), description without interval", result["truncated_only"], thresholds)

    if args.published and args.judge:
        if 0.5 not in thresholds:
            raise SystemExit("the comparison with the published figures needs the threshold 0.5")
        check = published_check(result["all"], json.loads(args.published.read_text(encoding="utf-8")), args.judge)
        result["published_check"] = check
        print(f"\nBalanced accuracy at 0.5 against the published figures of {check['entry']}")
        print(f"source: {check['source']}")
        print("| Dataset | Ours | Published | Gap, points | Above 2 points |")
        print("|---|---|---|---|---|")
        for name, row in check["rows"].items():
            label = "Mean of the five" if name == "mean" else name
            print(
                f"| {label} | {row['ours']:.1f} | {row['published']:.1f} | {row['gap_points']:+.1f} | {'YES' if row['flagged'] else 'no'} |"
            )

    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()

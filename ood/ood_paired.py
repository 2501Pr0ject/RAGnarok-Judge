"""Paired difference of mean AUC between two judges on the five datasets of LLM-AggreFact.

Reads the converted file and the ``attempts.jsonl`` of two ``ragnarok judge-bench`` runs,
A and B, and reports Delta = mean AUC(A) - mean AUC(B) with the resampling of
``ood_metrics.py``: 10000 resamples by document within each dataset, seed 7, the same draws
for both judges, so the difference is paired. Prints and writes aggregates only.

For Delta, on all inputs and on the inputs that no listed run truncates:

    point value, bootstrap standard error (standard deviation of the resampled differences),
    percentile interval at 95%, correlation of the two mean AUCs across resamples;
    one-sided p-value of the null "Delta <= -margin", as in section 9 of TARGETS.md:
    p = (1 + number of resamples with Delta* <= -margin) / (resamples + 1);
    the observed Delta from which that null is rejected at 0.025, by normal approximation;
    the power of the test at a few true values of Delta, by normal approximation, taking the
    measured standard error as that of the pair tested.

The per-dataset differences are reported the same way, without the test.

    python ood/ood_paired.py --data ood-out/llm-aggrefact-5.jsonl --run-a ood-runs/minicheck-bounded-mps --run-b ood-runs/hhem-l4096-mps --truncation-run ood-runs/hhem-l4096-mps --margin 0.02 --out ood-runs/paired-minicheck-hhem.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from ood_metrics import ALL, DATASETS, RESAMPLES, SEED, UNTRUNCATED, Block, measures, read_data, read_run, resample

ALPHA = 0.025
TRUE_DELTAS = (-0.02, -0.01, 0.0, 0.01, 0.02)


def normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def z_alpha(alpha: float) -> float:
    """Upper quantile of the standard normal, by bisection on ``normal_cdf``."""
    low, high = 0.0, 10.0
    for _ in range(200):
        mid = (low + high) / 2
        if normal_cdf(mid) < 1 - alpha:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def auc_draws(
    ids: list[str], records: dict[str, dict[str, Any]], scores: dict[str, float], stream: int
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """AUC per dataset: point value and the 10000 resampled values, draws keyed by dataset."""
    points: dict[str, float] = {}
    draws: dict[str, np.ndarray] = {}
    for index, dataset in enumerate(DATASETS):
        chosen = [sample for sample in ids if records[sample]["dataset"] == dataset]
        supported = np.array([records[sample]["supported"] for sample in chosen], dtype=bool)
        if min(supported.sum(), (~supported).sum()) == 0:
            raise SystemExit(f"{dataset}: a single class, no AUC")
        block = Block(
            np.array([scores[sample] for sample in chosen], dtype=np.float64),
            supported,
            np.array([records[sample]["group"] for sample in chosen]),
        )
        points[dataset] = float(measures(block, np.ones((1, len(chosen))), ())["auc"][0])
        draws[dataset] = resample(block, (), [SEED, index, stream])["auc"]
    return points, draws


def describe(point: float, draws: np.ndarray) -> dict[str, Any]:
    valid = draws[~np.isnan(draws)]
    low, high = np.percentile(valid, [2.5, 97.5])
    return {
        "value": point,
        "standard_error": float(valid.std(ddof=1)),
        "low": float(low),
        "high": float(high),
        "excluded": int(len(draws) - len(valid)),
    }


def compare(
    ids: list[str],
    records: dict[str, dict[str, Any]],
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    stream: int,
    margin: float,
) -> dict[str, Any]:
    points_a, draws_a = auc_draws(ids, records, scores_a, stream)
    points_b, draws_b = auc_draws(ids, records, scores_b, stream)
    out: dict[str, Any] = {"n": len(ids), "datasets": {}}
    for dataset in DATASETS:
        diff = draws_a[dataset] - draws_b[dataset]
        out["datasets"][dataset] = {
            "auc_a": points_a[dataset],
            "auc_b": points_b[dataset],
            "difference": describe(points_a[dataset] - points_b[dataset], diff),
            "correlation": float(np.corrcoef(draws_a[dataset], draws_b[dataset])[0, 1]),
        }
    mean_a = np.mean(np.vstack([draws_a[d] for d in DATASETS]), axis=0)
    mean_b = np.mean(np.vstack([draws_b[d] for d in DATASETS]), axis=0)
    diff = mean_a - mean_b
    valid = diff[~np.isnan(diff)]
    se = float(valid.std(ddof=1))
    z = z_alpha(ALPHA)
    point = float(np.mean(list(points_a.values())) - np.mean(list(points_b.values())))
    out["mean"] = {
        "auc_a": describe(float(np.mean(list(points_a.values()))), mean_a),
        "auc_b": describe(float(np.mean(list(points_b.values()))), mean_b),
        "difference": describe(point, diff),
        "correlation": float(np.corrcoef(mean_a, mean_b)[0, 1]),
        "standard_error_if_independent": float(math.hypot(mean_a.std(ddof=1), mean_b.std(ddof=1))),
        "test": {
            "null": f"Delta <= {-margin}",
            "alpha": ALPHA,
            "p_value": float((1 + int((valid <= -margin).sum())) / (RESAMPLES + 1)),
            "rejected": bool((1 + int((valid <= -margin).sum())) / (RESAMPLES + 1) < ALPHA),
            "observed_delta_to_reject": float(-margin + z * se),
            "power_by_true_delta": {f"{true:+.2f}": normal_cdf((true + margin) / se - z) for true in TRUE_DELTAS},
        },
    }
    return out


def show(title: str, table: dict[str, Any], name_a: str, name_b: str) -> None:
    print(f"\n{title}")
    print(f"| Dataset | AUC {name_a} | AUC {name_b} | Difference | SE | 95% interval | Correlation |")
    print("|---|---|---|---|---|---|---|")
    rows = [(d, table["datasets"][d]) for d in DATASETS] + [("Mean of the five", table["mean"])]
    for label, row in rows:
        diff = row["difference"]
        auc_a = row["auc_a"]["value"] if isinstance(row["auc_a"], dict) else row["auc_a"]
        auc_b = row["auc_b"]["value"] if isinstance(row["auc_b"], dict) else row["auc_b"]
        print(
            f"| {label} | {auc_a:.4f} | {auc_b:.4f} | {diff['value']:+.4f} | {diff['standard_error']:.4f} "
            f"| [{diff['low']:+.4f}, {diff['high']:+.4f}] | {row['correlation']:.2f} |"
        )
    test = table["mean"]["test"]
    mean = table["mean"]
    print(f"resamples excluded: {mean['difference']['excluded']}")
    print(f"standard error if the two judges were independent: {mean['standard_error_if_independent']:.4f}")
    print(f"null {test['null']}: p = {test['p_value']:.4f}, rejected at {ALPHA}: {'yes' if test['rejected'] else 'no'}")
    print(f"observed Delta from which the null is rejected at {ALPHA}: {test['observed_delta_to_reject']:+.4f}")
    print(
        "power by normal approximation, true Delta -> power: "
        + ", ".join(f"{true} -> {power:.2f}" for true, power in test["power_by_true_delta"].items())
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-a", type=Path, required=True, help="the judge whose AUC is compared, Delta = A - B")
    parser.add_argument("--run-b", type=Path, required=True, help="the reference judge")
    parser.add_argument(
        "--truncation-run",
        type=Path,
        action="append",
        help="run whose truncated inputs leave the untruncated subset; repeatable; default: both runs",
    )
    parser.add_argument("--margin", type=float, default=0.02, help="non-inferiority margin, default 0.02")
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose another --out")

    data_sha256 = hashlib.sha256(args.data.read_bytes()).hexdigest()
    records = read_data(args.data)
    runs = {}
    for run in (args.run_a, args.run_b):
        scores, truncated, unscored, header = read_run(run)
        if header.get("data_sha256") not in (None, data_sha256):
            raise SystemExit(f"{run} was made on another data file: {header.get('data_sha256')} != {data_sha256}")
        if set(scores) != set(records) or unscored:
            raise SystemExit(f"{run} and data do not match: {len(set(records) - set(scores))} ids without score")
        runs[run] = (scores, truncated, header.get("config_hash"))
    ids = sorted(records)
    truncated: set[str] = set()
    for run in args.truncation_run or [args.run_a, args.run_b]:
        truncated |= runs[run][1] if run in runs else read_run(run)[1]
    kept = [sample for sample in ids if sample not in truncated]

    print(f"data {args.data} sha256 {data_sha256}")
    print(f"A = {args.name_a}: {args.run_a}; B = {args.name_b}: {args.run_b}; Delta = mean AUC(A) - mean AUC(B)")
    print(f"{RESAMPLES} resamples by document within each dataset, seed 7, the same draws for both judges")
    result: dict[str, Any] = {
        "data": str(args.data),
        "data_sha256": data_sha256,
        "run_a": {"path": str(args.run_a), "name": args.name_a, "config_hash": runs[args.run_a][2]},
        "run_b": {"path": str(args.run_b), "name": args.name_b, "config_hash": runs[args.run_b][2]},
        "truncation_runs": [str(run) for run in args.truncation_run or [args.run_a, args.run_b]],
        "resamples": RESAMPLES,
        "seed": SEED,
        "margin": args.margin,
        "all": compare(ids, records, runs[args.run_a][0], runs[args.run_b][0], ALL, args.margin),
        "untruncated": compare(kept, records, runs[args.run_a][0], runs[args.run_b][0], UNTRUNCATED, args.margin),
    }
    show(f"All inputs ({len(ids)})", result["all"], args.name_a, args.name_b)
    show(f"Inputs not truncated ({len(kept)})", result["untruncated"], args.name_a, args.name_b)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()

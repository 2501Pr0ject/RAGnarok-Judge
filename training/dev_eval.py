"""Selection criterion on the dev set and tie-break between the bases.

The criterion is the one of TARGETS.md, section 12: the mean of two standardised partial AUCs
over a false alarm rate of 0 to 0.10, UNSUPPORTED against SUPPORTED and CONTRADICTED against
SUPPORTED. It is computed by the functions of ``analysis/ragtruth_analysis.py``, with its
resamples (by source, 10000, seed 7, the same draws for every judge). The dev set only.

    python training/dev_eval.py --dev DEV.jsonl --scores NAME=FILE [NAME=FILE ...] [--out OUT.json]

FILE is a JSON object id -> score, or the ``dev_attempts.jsonl`` of a harness run. With the
two names ``pleias`` and ``qwen3`` the tie-break rule is applied (amendment 1 of TARGETS.md).
Prints aggregates only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import ragtruth_analysis as ra  # noqa: E402

RESAMPLES = 10000
SEED = 7
LESS_COSTLY = "pleias"


def point(dev: dict[str, dict[str, Any]], scores: dict[str, float]) -> dict[str, float]:
    """Criterion of one judge and its two terms, no resampling."""
    split = ra.Split(dev, {"j": scores}, 0, SEED)
    return _terms(split, "j", split.one)


def _terms(split: ra.Split, judge: str, weights: Any) -> dict[str, float]:
    p_c, p_u = split.pauc(judge, weights, split.con), split.pauc(judge, weights, split.uns)
    return {"criterion": (p_c + p_u) / 2, "pauc_U": p_u, "pauc_C": p_c}


def criterion(
    dev: dict[str, dict[str, Any]], scores: dict[str, dict[str, float]], resamples: int = RESAMPLES, seed: int = SEED
) -> dict[str, Any]:
    """Criterion and terms of every judge with percentile intervals, and the paired differences
    of the criterion between judges, with their standard error."""
    for name, values in scores.items():
        if set(values) != set(dev):
            raise SystemExit(f"{name}: scores do not cover the dev set exactly")
    split = ra.Split(dev, scores, resamples, seed)
    names = list(scores)
    points = {j: _terms(split, j, split.one) for j in names}
    draws: dict[str, dict[str, list[float]]] = {j: {k: [] for k in points[j]} for j in names}
    for b in range(resamples):
        weights = split.w(b)
        for j in names:
            for k, value in _terms(split, j, weights).items():
                draws[j][k].append(value)
    result: dict[str, Any] = {
        "resamples": resamples,
        "seed": seed,
        "sources": split.G,
        "judges": {j: {k: ra.summary(points[j][k], draws[j][k]) for k in points[j]} for j in names},
        "differences": {},
    }
    for i, a in enumerate(names):
        for b_name in names[i + 1 :]:
            result["differences"][f"{a}-{b_name}"] = ra.paired(
                points[a]["criterion"] - points[b_name]["criterion"],
                draws[a]["criterion"],
                draws[b_name]["criterion"],
                {},
                resamples,
            )
    return result


def tie_break(criterion_pleias: float, criterion_qwen3: float, standard_error: float) -> str:
    """Base retained: the less costly one when the two criteria differ by less than one standard
    error of their difference, the one with the higher criterion otherwise."""
    ahead = criterion_qwen3 - criterion_pleias  # an exact tie goes to the less costly base as well
    return "qwen3" if ahead > 0 and ahead >= standard_error else LESS_COSTLY


def select_base(
    dev: dict[str, dict[str, Any]], pleias: dict[str, float], qwen3: dict[str, float], resamples: int = RESAMPLES
) -> dict[str, Any]:
    """The tie-break applied to the best dev checkpoint of each base."""
    result = criterion(dev, {"pleias": pleias, "qwen3": qwen3}, resamples)
    difference = result["differences"]["pleias-qwen3"]
    values = {j: result["judges"][j]["criterion"]["value"] for j in ("pleias", "qwen3")}
    result["selection"] = {
        "criterion": values,
        "difference": difference["value"],
        "standard_error": difference["se"],
        "within_one_standard_error": abs(difference["value"]) < difference["se"],
        "retained": tie_break(values["pleias"], values["qwen3"], difference["se"]),
    }
    return result


def read_scores(path: Path, dev: dict[str, dict[str, Any]], dev_sha: str) -> dict[str, float]:
    if path.suffix == ".jsonl":
        return ra.read_attempts(path, set(dev), dev_sha)[0]
    return {str(k): float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--scores", nargs="+", required=True, metavar="NAME=FILE")
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out is not None and args.out.exists():
        raise SystemExit(f"{args.out} exists; choose another --out")
    dev = ra.read_split(args.dev)
    sha = ra.sha256(args.dev)
    scores = {}
    for item in args.scores:
        name, _, file = item.partition("=")
        scores[name] = read_scores(Path(file), dev, sha)
    if set(scores) == {"pleias", "qwen3"}:
        result = select_base(dev, scores["pleias"], scores["qwen3"], args.resamples)
    else:
        result = criterion(dev, scores, args.resamples)
    result["dev_sha256"] = sha
    for name, block in result["judges"].items():
        c, u, k = block["criterion"], block["pauc_U"], block["pauc_C"]
        print(f"{name}: criterion {c['value']:.4f} [{c['low']:.4f}, {c['high']:.4f}], "
              f"UNSUPPORTED {u['value']:.4f}, CONTRADICTED {k['value']:.4f}")
    for name, block in result["differences"].items():
        print(f"{name}: difference {block['value']:+.4f}, standard error {block['se']:.4f}")
    if "selection" in result:
        print(f"retained: {result['selection']['retained']}")
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

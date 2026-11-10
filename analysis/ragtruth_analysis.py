"""Primary and secondary analyses of RAGnarok Judge on the RAGTruth test set (TARGETS.md).

Implements sections 4, 5, 6 and 9 of TARGETS.md for a candidate judge A against the reference
judge B (HHEM), with an optional second reference C (MiniCheck) reported alongside:

    section 4   operating points: for n SUPPORTED sentences, k = floor(0.04 n) and the threshold
                is the (k+1)-th smallest score on them; a sentence is flagged when score < t.
                Equal specificity on test (threshold on the 9777 SUPPORTED test sentences, re-derived
                in every resample) and the dev threshold (on the 5536 SUPPORTED dev sentences, a
                constant).
    section 5   P1: Delta_C = recall_C(A) - recall_C(B), null Delta_C <= 0.
                P2: Delta_U = recall_U(A) - recall_U(B), null Delta_U <= -0.05.
                One-sided bootstrap p-values, Holm over P1 and P2 at a family-wise level of 0.025.
    section 6   recall on the not supported sentences, AUC ROC and per-class AUCs, the selection
                criterion (mean of two standardised partial AUCs, false alarm rate 0 to 0.10) on
                test and on dev, precision and F1 of not supported at the natural prevalence, the
                differences A minus C, the metrics outside meta sentences and on abstentions, recall
                on CONTRADICTED outside a list of ids, the inputs truncated by each judge, and the
                breakdowns by task and generator. Transfer of the dev threshold to the test set,
                with the shift of specificity between dev and test.
    section 9   resampling by source (``source_id``), 10000 resamples, seed 7, the same draws for
                every judge; percentile intervals at 95%; p = (1 + #{Delta* <= d0}) / (B + 1); a
                resample with a zero denominator is excluded and counted.

Inputs: the frozen test and dev files of the converter, and for each judge the directory of a
``ragnarok judge-bench`` run holding ``attempts.jsonl`` (test) and ``dev_attempts.jsonl`` (dev).
Every sentence of both splits must have a score. The headers' ``data_sha256`` must match the
files. Prints and writes aggregates only: no text, no label and no score of a sentence leaves
the script. The subsets outside meta sentences and on abstentions use the pattern lists next to
this script (``meta_patterns.json``, ``abstention_patterns.json``) through ``meta_match.py``.

    python3 analysis/ragtruth_analysis.py --test ragtruth-out/test.jsonl --dev ragtruth-out/dev.jsonl --run-a bench-1.13/full/minicheck --run-b bench-1.13/full/hhem --name-a MiniCheck --name-b HHEM --out analysis-minicheck-vs-hhem.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

LABELS = ("SUPPORTED", "UNSUPPORTED", "CONTRADICTED")
TARGET_SPECIFICITY = 0.96
FPR_MAX = 0.10
MARGIN_U = 0.05
ALPHA_FAMILY = 0.025
MIN_MINORITY = 10
RATE_KEYS = ("C", "U", "NS", "S", "spec", "precision", "f1")


# ----------------------------------------------------------------------------- reading


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_split(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for record in read_jsonl(path):
        if record["label"] not in LABELS:
            raise SystemExit(f"{path}: unexpected label {record['label']!r}")
        if record["id"] in records:
            raise SystemExit(f"{path}: duplicate id {record['id']}")
        records[record["id"]] = record
    return records


def read_attempts(
    path: Path, expected_ids: set[str], data_sha: str
) -> tuple[dict[str, float], set[str], dict[str, Any]]:
    """Scores of the last successful attempt per sample, ids whose input was truncated, header."""
    rows = read_jsonl(path)
    header, attempts = rows[0], rows[1:]
    if header.get("data_sha256") not in (None, data_sha):
        raise SystemExit(f"{path}: made on another data file ({header.get('data_sha256')} != {data_sha})")
    scores: dict[str, float] = {}
    truncated: set[str] = set()
    for attempt in attempts:
        if not attempt.get("ok") or attempt.get("result") is None:
            continue
        scores[attempt["sample_id"]] = float(attempt["result"]["score"])
        if (attempt.get("trace") or {}).get("context_truncated"):
            truncated.add(attempt["sample_id"])
        else:
            truncated.discard(attempt["sample_id"])
    if set(scores) != expected_ids:
        missing = expected_ids - set(scores)
        unknown = set(scores) - expected_ids
        raise SystemExit(f"{path}: {len(missing)} sentences without score, {len(unknown)} unknown ids")
    return scores, truncated, header


# ----------------------------------------------------------------------------- resampling


class Split:
    """One split with the scores of every judge, and the resamples of its sources."""

    def __init__(
        self, records: dict[str, dict[str, Any]], scores: dict[str, dict[str, float]], resamples: int, seed: int
    ) -> None:
        self.ids = sorted(records)
        self.lab = np.array([records[i]["label"] for i in self.ids])
        self.s = {j: np.array([scores[j][i] for i in self.ids], dtype=float) for j in scores}
        groups = sorted({records[i]["source_id"] for i in self.ids})
        index = {g: k for k, g in enumerate(groups)}
        self.g = np.array([index[records[i]["source_id"]] for i in self.ids])
        self.G = len(groups)
        self.draws = np.random.default_rng(seed).integers(0, self.G, size=(resamples, self.G))
        self.sup = self.lab == "SUPPORTED"
        self.con = self.lab == "CONTRADICTED"
        self.uns = self.lab == "UNSUPPORTED"
        self.ns = ~self.sup
        self.one = np.ones(len(self.ids))
        self.sup_order = {j: np.argsort(self.s[j][self.sup], kind="stable") for j in self.s}
        self.order, self.starts = {}, {}
        for j in self.s:
            order = np.argsort(self.s[j], kind="stable")
            sorted_scores = self.s[j][order]
            self.order[j] = order
            self.starts[j] = np.concatenate(([0], np.flatnonzero(np.diff(sorted_scores) != 0) + 1))

    def w(self, b: int) -> np.ndarray:
        return np.bincount(self.draws[b], minlength=self.G)[self.g].astype(float)

    def thr(self, j: str, w: np.ndarray) -> float:
        """Largest score that flags at most 4% of the SUPPORTED weight (section 4)."""
        order = self.sup_order[j]
        sorted_scores = self.s[j][self.sup][order]
        weights = w[self.sup][order]
        before = np.cumsum(weights) - weights
        k = np.where((before <= (1 - TARGET_SPECIFICITY) * weights.sum() + 1e-9) & (weights > 0))[0].max()
        return float(sorted_scores[k])

    def rates(self, j: str, t: float, w: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
        """Recall per class, specificity, precision and F1 of not supported; flagged when score < t."""
        flagged = self.s[j] < t
        keep = self.one.astype(bool) if mask is None else mask
        out: dict[str, float] = {}
        for name, m in (("C", self.con), ("U", self.uns), ("NS", self.ns), ("S", self.sup)):
            m = m & keep
            d = float(np.sum(w * m))
            out[name] = float(np.sum(w * (m & flagged)) / d) if d else float("nan")
        out["spec"] = 1 - out["S"]
        hits = float(np.sum(w * (self.ns & keep & flagged)))
        called = float(np.sum(w * (keep & flagged)))
        out["precision"] = hits / called if called else float("nan")
        recall = out["NS"]
        out["f1"] = (
            2 * out["precision"] * recall / (out["precision"] + recall)
            if called and not np.isnan(recall) and (out["precision"] + recall) > 0
            else float("nan")
        )
        return out

    def roc(self, j: str, w: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order, starts = self.order[j], self.starts[j]
        gp = np.add.reduceat((w * pos)[order], starts)
        gn = np.add.reduceat((w * self.sup)[order], starts)
        x = np.concatenate(([0.0], np.cumsum(gn) / gn.sum()))
        y = np.concatenate(([0.0], np.cumsum(gp) / gp.sum()))
        return x, y

    def auc(self, j: str, w: np.ndarray, pos: np.ndarray) -> float:
        x, y = self.roc(j, w, pos)
        return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))

    def pauc(self, j: str, w: np.ndarray, pos: np.ndarray) -> float:
        """Standardised partial AUC over a false alarm rate of 0 to FPR_MAX (McClish)."""
        x, y = self.roc(j, w, pos)
        stop = int(np.searchsorted(x, FPR_MAX, side="right"))
        x0, x1, y0, y1 = x[stop - 1], x[stop], y[stop - 1], y[stop]
        yb = y0 + (y1 - y0) * (FPR_MAX - x0) / (x1 - x0)
        xs = np.concatenate((x[:stop], [FPR_MAX]))
        ys = np.concatenate((y[:stop], [yb]))
        raw = float(np.sum(np.diff(xs) * (ys[1:] + ys[:-1]) / 2))
        low, high = FPR_MAX**2 / 2, FPR_MAX
        return 0.5 * (1 + (raw - low) / (high - low))


def summary(value: float, draws: list[float] | np.ndarray) -> dict[str, Any]:
    a = np.asarray(draws, dtype=float)
    valid = a[~np.isnan(a)]
    if len(valid) == 0:
        return {"value": value, "low": None, "high": None, "se": None, "excluded": len(a)}
    low, high = np.percentile(valid, [2.5, 97.5])
    return {
        "value": value,
        "low": float(low),
        "high": float(high),
        "se": float(valid.std()),
        "excluded": int(len(a) - len(valid)),
    }


def paired(point: float, a: list[float], b: list[float], nulls: dict[str, float], resamples: int) -> dict[str, Any]:
    da, db = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = da - db
    out = summary(point, diff)
    valid = ~np.isnan(diff)
    constant = valid.sum() <= 2 or da[valid].std() == 0 or db[valid].std() == 0
    out["correlation"] = None if constant else float(np.corrcoef(da[valid], db[valid])[0, 1])
    out["p_values"] = {name: float((1 + np.sum(diff[valid] <= d0)) / (resamples + 1)) for name, d0 in nulls.items()}
    return out


def holm(p1: float, p2: float) -> dict[str, Any]:
    """Holm over P1 and P2 at a family-wise one-sided level of 0.025 (section 5)."""
    first, second = sorted((("P1", p1), ("P2", p2)), key=lambda item: item[1])
    met = {first[0]: first[1] < ALPHA_FAMILY / 2}
    met[second[0]] = met[first[0]] and second[1] < ALPHA_FAMILY
    claims = {
        (
            True,
            True,
        ): "at 96% specificity on the RAGTruth test, A detects more contradictions than B and is not worse on unsupported sentences",
        (True, False): "A detects more contradictions than B; a loss on unsupported sentences is not excluded",
        (False, True): "no demonstrated gain on contradictions; A is not worse than B on unsupported sentences",
        (False, False): "no claim",
    }
    return {
        "p_values": {"P1": p1, "P2": p2},
        "compared_first": first[0],
        "levels": {first[0]: ALPHA_FAMILY / 2, second[0]: ALPHA_FAMILY},
        "met": {"P1": bool(met["P1"]), "P2": bool(met["P2"])},
        "claim": claims[(bool(met["P1"]), bool(met["P2"]))],
    }


def flag(counts: dict[str, int]) -> bool:
    return (
        min(counts["UNSUPPORTED"] + counts["CONTRADICTED"], counts["SUPPORTED"]) < MIN_MINORITY
        or min(counts["UNSUPPORTED"], counts["CONTRADICTED"]) < MIN_MINORITY
    )


def label_counts(lab: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    return {label: int(np.sum(mask & (lab == label))) for label in LABELS}


# ----------------------------------------------------------------------------- analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument(
        "--run-a", type=Path, required=True, help="the candidate: directory with attempts.jsonl and dev_attempts.jsonl"
    )
    parser.add_argument("--run-b", type=Path, required=True, help="the reference compared with (HHEM)")
    parser.add_argument("--run-c", type=Path, help="a second reference reported alongside, no test (MiniCheck)")
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--name-c", default="C")
    parser.add_argument(
        "--exclude-ids", type=Path, help="ids left out for the recall on CONTRADICTED outside them, one per line"
    )
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose another --out")

    test = read_split(args.test)
    dev = read_split(args.dev)
    if set(test) & set(dev):
        raise SystemExit("test and dev share ids")
    sha_test, sha_dev = sha256(args.test), sha256(args.dev)
    judges = {"A": args.run_a, "B": args.run_b}
    names = {"A": args.name_a, "B": args.name_b}
    if args.run_c is not None:
        judges["C"] = args.run_c
        names["C"] = args.name_c
    sc_test, sc_dev, truncated, headers = {}, {}, {}, {}
    for j, run in judges.items():
        sc_test[j], truncated[j], headers[j] = read_attempts(run / "attempts.jsonl", set(test), sha_test)
        sc_dev[j], _, header_dev = read_attempts(run / "dev_attempts.jsonl", set(dev), sha_dev)
        if header_dev.get("config_hash") != headers[j].get("config_hash"):
            raise SystemExit(f"{run}: the dev and test runs have different configurations")

    B = args.resamples
    T = Split(test, sc_test, B, args.seed)
    D = Split(dev, sc_dev, B, args.seed)
    J = list(judges)

    # section 4: thresholds
    thr_dev = {j: D.thr(j, D.one) for j in J}
    thr_eq = {j: T.thr(j, T.one) for j in J}
    thresholds = {
        j: {
            "dev_threshold": thr_dev[j],
            "dev_specificity": D.rates(j, thr_dev[j], D.one)["spec"],
            "equal_specificity_threshold_test": thr_eq[j],
            "test_specificity_at_it": T.rates(j, thr_eq[j], T.one)["spec"],
            "supported_flagged_at_it": int(np.sum(T.s[j][T.sup] < thr_eq[j])),
            "tied_at_it": int(np.sum(T.s[j] == thr_eq[j])),
        }
        for j in J
    }

    # subsets of the test set
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from meta_match import is_abstention, meta_category

    sentences = [test[i]["sentence"] for i in T.ids]
    meta = np.array([meta_category(s) is not None for s in sentences])
    abstention = np.array([is_abstention(s) for s in sentences])
    if np.any(meta & abstention):
        raise SystemExit("a sentence is both meta and an abstention")
    subsets = {"outside_meta": ~meta, "abstentions": abstention}
    excluded_ids: set[str] = set()
    if args.exclude_ids is not None:
        excluded_ids = {
            line.strip() for line in args.exclude_ids.read_text(encoding="utf-8").splitlines() if line.strip()
        }
        unknown = excluded_ids - set(test)
        if unknown:
            raise SystemExit(f"{len(unknown)} excluded ids are not in the test set")
        subsets["outside_excluded_ids"] = np.array([i not in excluded_ids for i in T.ids])
    task = np.array([test[i]["task"] for i in T.ids])
    generator = np.array([test[i]["generator"] for i in T.ids])

    # section 9: the test resamples, thresholds re-derived (equal specificity) and constant (dev, 0.5)
    eq = {j: {k: [] for k in RATE_KEYS} for j in J}
    fx = {j: {k: [] for k in RATE_KEYS} for j in J}
    half = {j: {k: [] for k in RATE_KEYS} for j in J}
    au = {j: {k: [] for k in ("auc", "auc_C", "auc_U", "pauc", "pauc_C", "pauc_U", "criterion")} for j in J}
    sub_eq = {j: {name: {k: [] for k in RATE_KEYS} for name in subsets} for j in J}
    sub_fx = {j: {name: {k: [] for k in RATE_KEYS} for name in subsets} for j in J}
    sub_auc = {j: [] for j in J}
    for b in range(B):
        w = T.w(b)
        for j in J:
            t = T.thr(j, w)
            r_eq, r_fx, r_half = T.rates(j, t, w), T.rates(j, thr_dev[j], w), T.rates(j, 0.5, w)
            for k in RATE_KEYS:
                eq[j][k].append(r_eq[k])
                fx[j][k].append(r_fx[k])
                half[j][k].append(r_half[k])
            a_all, a_c, a_u = T.auc(j, w, T.ns), T.auc(j, w, T.con), T.auc(j, w, T.uns)
            p_all, p_c, p_u = T.pauc(j, w, T.ns), T.pauc(j, w, T.con), T.pauc(j, w, T.uns)
            for k, v in zip(au[j], (a_all, a_c, a_u, p_all, p_c, p_u, (p_c + p_u) / 2), strict=True):
                au[j][k].append(v)
            for name, mask in subsets.items():
                r1, r2 = T.rates(j, t, w, mask), T.rates(j, thr_dev[j], w, mask)
                for k in RATE_KEYS:
                    sub_eq[j][name][k].append(r1[k])
                    sub_fx[j][name][k].append(r2[k])
            # AUC outside meta sentences: positives and negatives restricted to the subset through the weights
            sub_auc[j].append(T.auc(j, w * subsets["outside_meta"], T.ns))

    def point_block(j: str, t: float, mask: np.ndarray | None = None) -> dict[str, float]:
        return T.rates(j, t, T.one, mask)

    result: dict[str, Any] = {
        "files": {"test": str(args.test), "test_sha256": sha_test, "dev": str(args.dev), "dev_sha256": sha_dev},
        "judges": {
            j: {
                "name": names[j],
                "run": str(judges[j]),
                "config_hash": headers[j].get("config_hash"),
                "truncated_test_inputs": len(truncated[j]),
            }
            for j in J
        },
        "resamples": B,
        "seed": args.seed,
        "counts": {
            "test": label_counts(T.lab, T.one.astype(bool)) | {"sources": T.G},
            "dev": label_counts(D.lab, D.one.astype(bool)) | {"sources": D.G},
        },
        "thresholds": thresholds,
    }

    # section 5: primary analysis
    primary: dict[str, Any] = {"judges": {}, "differences": {}}
    for j in J:
        pt = point_block(j, thr_eq[j])
        primary["judges"][j] = {k: summary(pt[k], eq[j][k]) for k in RATE_KEYS}
    nulls = {
        "C": {"P1: Delta <= 0": 0.0},
        "U": {"P2: Delta <= -0.05": -MARGIN_U, "Delta <= 0": 0.0},
        "NS": {"Delta <= 0": 0.0},
    }
    for other in [j for j in J if j != "A"]:
        pa, pb = point_block("A", thr_eq["A"]), point_block(other, thr_eq[other])
        primary["differences"][f"A-{other}"] = {
            k: paired(pa[k] - pb[k], eq["A"][k], eq[other][k], nulls.get(k, {"Delta <= 0": 0.0}), B)
            for k in ("C", "U", "NS", "precision", "f1")
        }
    p1 = primary["differences"]["A-B"]["C"]["p_values"]["P1: Delta <= 0"]
    p2 = primary["differences"]["A-B"]["U"]["p_values"]["P2: Delta <= -0.05"]
    primary["holm"] = holm(p1, p2)
    result["primary_equal_specificity"] = primary

    # section 6: transfer of the dev threshold, and 0.5
    transfer: dict[str, Any] = {"judges": {}, "differences": {}}
    for j in J:
        pt, pt5 = point_block(j, thr_dev[j]), point_block(j, 0.5)
        transfer["judges"][j] = {
            "dev_threshold": thr_dev[j],
            "at_dev_threshold": {k: summary(pt[k], fx[j][k]) for k in RATE_KEYS},
            "balanced_accuracy_at_dev_threshold": summary(
                (pt["NS"] + pt["spec"]) / 2, (np.array(fx[j]["NS"]) + np.array(fx[j]["spec"])) / 2
            ),
            "at_0.5": {k: summary(pt5[k], half[j][k]) for k in RATE_KEYS},
            "balanced_accuracy_at_0.5": summary(
                (pt5["NS"] + pt5["spec"]) / 2, (np.array(half[j]["NS"]) + np.array(half[j]["spec"])) / 2
            ),
        }
    for other in [j for j in J if j != "A"]:
        pa, pb = point_block("A", thr_dev["A"]), point_block(other, thr_dev[other])
        transfer["differences"][f"A-{other}"] = {
            k: paired(pa[k] - pb[k], fx["A"][k], fx[other][k], {"Delta <= 0": 0.0}, B) for k in RATE_KEYS
        }
    result["transfer_dev_threshold"] = transfer

    # section 6: without threshold, test
    without: dict[str, Any] = {"judges": {}, "differences": {}}
    points = {}
    for j in J:
        p_c, p_u = T.pauc(j, T.one, T.con), T.pauc(j, T.one, T.uns)
        points[j] = {
            "auc": T.auc(j, T.one, T.ns),
            "auc_C": T.auc(j, T.one, T.con),
            "auc_U": T.auc(j, T.one, T.uns),
            "pauc": T.pauc(j, T.one, T.ns),
            "pauc_C": p_c,
            "pauc_U": p_u,
            "criterion": (p_c + p_u) / 2,
        }
        without["judges"][j] = {k: summary(points[j][k], au[j][k]) for k in points[j]}
    for other in [j for j in J if j != "A"]:
        without["differences"][f"A-{other}"] = {
            k: paired(points["A"][k] - points[other][k], au["A"][k], au[other][k], {"Delta <= 0": 0.0}, B)
            for k in points["A"]
        }
    result["without_threshold_test"] = without

    # section 12 on dev: the selection criterion, with the dev resamples
    crit = {j: {k: [] for k in ("auc", "pauc_C", "pauc_U", "criterion")} for j in J}
    for b in range(B):
        w = D.w(b)
        for j in J:
            p_c, p_u = D.pauc(j, w, D.con), D.pauc(j, w, D.uns)
            for k, v in zip(crit[j], (D.auc(j, w, D.ns), p_c, p_u, (p_c + p_u) / 2), strict=True):
                crit[j][k].append(v)
    dev_block: dict[str, Any] = {"judges": {}, "differences": {}}
    dev_points = {}
    for j in J:
        p_c, p_u = D.pauc(j, D.one, D.con), D.pauc(j, D.one, D.uns)
        dev_points[j] = {"auc": D.auc(j, D.one, D.ns), "pauc_C": p_c, "pauc_U": p_u, "criterion": (p_c + p_u) / 2}
        dev_block["judges"][j] = {k: summary(dev_points[j][k], crit[j][k]) for k in dev_points[j]}
    for other in [j for j in J if j != "A"]:
        dev_block["differences"][f"A-{other}"] = {
            k: paired(dev_points["A"][k] - dev_points[other][k], crit["A"][k], crit[other][k], {"Delta <= 0": 0.0}, B)
            for k in dev_points["A"]
        }
    result["selection_criterion_dev"] = dev_block

    # section 6: shift of specificity between dev and test (dev and test resampled together)
    rng = np.random.default_rng(args.seed + 1)
    dev_draws = rng.integers(0, D.G, size=(B, D.G))
    both = {j: [] for j in J}
    for b in range(B):
        wd = np.bincount(dev_draws[b], minlength=D.G)[D.g].astype(float)
        wt = T.w(b)
        for j in J:
            both[j].append(T.rates(j, D.thr(j, wd), wt)["spec"])
    shift: dict[str, Any] = {"judges": {}, "differences": {}}
    for j in J:
        a = np.array(both[j])
        by_task = {}
        for split_name, S in (("dev", D), ("test", T)):
            tasks = np.array([(dev if split_name == "dev" else test)[i]["task"] for i in S.ids])
            by_task[split_name] = {
                tk: {
                    "specificity": float(1 - np.mean(S.s[j][S.sup & (tasks == tk)] < thr_dev[j])),
                    "supported": int(np.sum(S.sup & (tasks == tk))),
                }
                for tk in sorted(set(tasks))
            }
        shift["judges"][j] = {
            "test_specificity_at_dev_threshold": T.rates(j, thr_dev[j], T.one)["spec"],
            "interval_dev_and_test_resampled": summary(T.rates(j, thr_dev[j], T.one)["spec"], a),
            "share_of_resamples_at_or_below_0.96": float((1 + np.sum(a <= TARGET_SPECIFICITY)) / (B + 1)),
            "dev_supported_flagged_at_test_equal_specificity_threshold": float(np.mean(D.s[j][D.sup] < thr_eq[j])),
            "specificity_at_dev_threshold_by_task": by_task,
        }
    for other in [j for j in J if j != "A"]:
        pa, pb = T.rates("A", thr_dev["A"], T.one)["spec"], T.rates(other, thr_dev[other], T.one)["spec"]
        shift["differences"][f"A-{other}"] = paired(pa - pb, both["A"], both[other], {"Delta <= 0": 0.0}, B)
    result["specificity_shift_dev_to_test"] = shift

    # section 6: subsets (thresholds of the full set), breakdowns, truncation
    subsets_block: dict[str, Any] = {}
    for name, mask in subsets.items():
        counts = label_counts(T.lab, mask)
        entry: dict[str, Any] = {
            "sentences": int(mask.sum()),
            "counts": counts,
            "minority_below_10": flag(counts),
            "judges": {},
        }
        for j in J:
            pt_eq, pt_fx = point_block(j, thr_eq[j], mask), point_block(j, thr_dev[j], mask)
            entry["judges"][j] = {
                "at_equal_specificity_threshold": {k: summary(pt_eq[k], sub_eq[j][name][k]) for k in RATE_KEYS},
                "at_dev_threshold": {k: summary(pt_fx[k], sub_fx[j][name][k]) for k in RATE_KEYS},
            }
            if name == "outside_meta":
                entry["judges"][j]["auc"] = summary(T.auc(j, T.one * mask, T.ns), sub_auc[j])
        for other in [j for j in J if j != "A"]:
            pa, pb = point_block("A", thr_eq["A"], mask), point_block(other, thr_eq[other], mask)
            entry[f"A-{other}_at_equal_specificity_threshold"] = {
                k: paired(pa[k] - pb[k], sub_eq["A"][name][k], sub_eq[other][name][k], {"Delta <= 0": 0.0}, B)
                for k in ("C", "U", "NS")
            }
        subsets_block[name] = entry
    if args.exclude_ids is not None:
        subsets_block["outside_excluded_ids"]["excluded_ids_file"] = str(args.exclude_ids)
        subsets_block["outside_excluded_ids"]["excluded_in_test"] = len(excluded_ids)
    result["subsets"] = subsets_block

    breakdowns: dict[str, Any] = {}
    for axis, values in (("task", task), ("generator", generator)):
        breakdowns[axis] = {}
        for value in sorted(set(values)):
            mask = values == value
            counts = label_counts(T.lab, mask)
            cell = {"counts": counts, "minority_below_10": flag(counts), "judges": {}}
            for j in J:
                cell["judges"][j] = {
                    "at_equal_specificity_threshold": point_block(j, thr_eq[j], mask),
                    "at_dev_threshold": point_block(j, thr_dev[j], mask),
                }
            breakdowns[axis][value] = cell
    result["breakdowns_point_values"] = breakdowns

    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    show(result, names)
    print(f"\nwritten: {args.out}")


# ----------------------------------------------------------------------------- display


def cell(entry: dict[str, Any], digits: int = 4) -> str:
    value = entry["value"]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if entry.get("low") is None:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} [{entry['low']:.{digits}f}, {entry['high']:.{digits}f}]"


def show(result: dict[str, Any], names: dict[str, str]) -> None:
    J = list(result["judges"])
    head = "| Quantity | " + " | ".join(names[j] for j in J) + " |"
    sep = "|---|" + "---|" * len(J)
    thr = result["thresholds"]
    print(
        f"\nTest: {result['counts']['test']}; dev: {result['counts']['dev']}; {result['resamples']} resamples, seed {result['seed']}"
    )
    print("\nPrimary analysis, equal specificity on test (thresholds re-derived in every resample)")
    print(head + "\n" + sep)
    print(
        "| Threshold on the full test | "
        + " | ".join(f"{thr[j]['equal_specificity_threshold_test']:.5f}" for j in J)
        + " |"
    )
    print(
        "| Specificity | "
        + " | ".join(
            f"{thr[j]['test_specificity_at_it']:.4f} ({thr[j]['supported_flagged_at_it']} flagged, {thr[j]['tied_at_it']} tied)"
            for j in J
        )
        + " |"
    )
    prim = result["primary_equal_specificity"]
    for k, label in (
        ("C", "Recall, CONTRADICTED"),
        ("U", "Recall, UNSUPPORTED"),
        ("NS", "Recall, not supported"),
        ("precision", "Precision, not supported"),
        ("f1", "F1, not supported"),
    ):
        print(f"| {label} | " + " | ".join(cell(prim["judges"][j][k]) for j in J) + " |")
    for pair, diffs in prim["differences"].items():
        a, b = pair.split("-")
        print(f"\nDifferences {names[a]} minus {names[b]}:")
        for k in ("C", "U", "NS"):
            d = diffs[k]
            pv = ", ".join(f"p({n}) = {p:.4f}" for n, p in d["p_values"].items())
            corr = "-" if d["correlation"] is None else f"{d['correlation']:.3f}"
            print(
                f"  Delta_{k} {d['value']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}], se {d['se']:.4f}, correlation {corr}, {pv}"
            )
    h = prim["holm"]
    print(
        f"Holm: {h['compared_first']} compared first; P1 {'met' if h['met']['P1'] else 'not met'} (p = {h['p_values']['P1']:.4f}), P2 {'met' if h['met']['P2'] else 'not met'} (p = {h['p_values']['P2']:.4f}); claim: {h['claim']}"
    )

    tr = result["transfer_dev_threshold"]
    print("\nTransfer of the dev threshold to the test set (thresholds constant)")
    print(head + "\n" + sep)
    print(
        "| Dev threshold | "
        + " | ".join(f"{thr[j]['dev_threshold']:.5f} (dev specificity {thr[j]['dev_specificity']:.4f})" for j in J)
        + " |"
    )
    for k, label in (
        ("spec", "Specificity on test"),
        ("C", "Recall, CONTRADICTED"),
        ("U", "Recall, UNSUPPORTED"),
        ("NS", "Recall, not supported"),
    ):
        print(f"| {label} | " + " | ".join(cell(tr["judges"][j]["at_dev_threshold"][k]) for j in J) + " |")
    print(
        "| Balanced accuracy at the dev threshold | "
        + " | ".join(cell(tr["judges"][j]["balanced_accuracy_at_dev_threshold"]) for j in J)
        + " |"
    )
    print(
        "| Balanced accuracy at 0.5 | "
        + " | ".join(cell(tr["judges"][j]["balanced_accuracy_at_0.5"]) for j in J)
        + " |"
    )

    wt = result["without_threshold_test"]
    print("\nWithout threshold")
    print(head + "\n" + sep)
    for k, label in (
        ("auc", "AUC ROC"),
        ("auc_C", "AUC, CONTRADICTED against SUPPORTED"),
        ("auc_U", "AUC, UNSUPPORTED against SUPPORTED"),
        ("pauc_U", "Partial AUC, UNSUPPORTED against SUPPORTED, test"),
        ("pauc_C", "Partial AUC, CONTRADICTED against SUPPORTED, test"),
        ("criterion", "Selection criterion (mean of the two), test"),
    ):
        print(f"| {label} | " + " | ".join(cell(wt["judges"][j][k]) for j in J) + " |")
    dv = result["selection_criterion_dev"]
    print("| Selection criterion, dev | " + " | ".join(cell(dv["judges"][j]["criterion"]) for j in J) + " |")
    print(
        "| Partial AUC, classes pooled, test (not the criterion) | "
        + " | ".join(cell(wt["judges"][j]["pauc"]) for j in J)
        + " |"
    )

    sh = result["specificity_shift_dev_to_test"]
    print("\nShift of specificity between dev and test")
    for j in J:
        e = sh["judges"][j]
        iv = e["interval_dev_and_test_resampled"]
        tasks = "; ".join(
            f"{tk}: dev {e['specificity_at_dev_threshold_by_task']['dev'][tk]['specificity']:.3f} -> test {e['specificity_at_dev_threshold_by_task']['test'][tk]['specificity']:.3f}"
            for tk in e["specificity_at_dev_threshold_by_task"]["test"]
        )
        print(
            f"  {names[j]}: test specificity {e['test_specificity_at_dev_threshold']:.4f}, dev and test resampled [{iv['low']:.4f}, {iv['high']:.4f}], share at or below 0.96: {e['share_of_resamples_at_or_below_0.96']:.4f}; {tasks}"
        )
    for pair, d in sh["differences"].items():
        a, b = pair.split("-")
        print(f"  {names[a]} minus {names[b]}: {d['value']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}]")

    print("\nSubsets of the test set (thresholds of the full set)")
    for name, entry in result["subsets"].items():
        extra = ", flagged: minority below 10" if entry["minority_below_10"] else ""
        print(f"  {name}: {entry['sentences']} sentences {entry['counts']}{extra}")
        for j in J:
            r = entry["judges"][j]["at_equal_specificity_threshold"]
            print(
                f"    {names[j]}: recall C {cell(r['C'])}, U {cell(r['U'])}, NS {cell(r['NS'])}, specificity {cell(r['spec'])}"
            )
    print(
        "\nInputs truncated by each judge on the test set: "
        + ", ".join(f"{names[j]} {result['judges'][j]['truncated_test_inputs']}" for j in J)
    )
    flagged = [
        f"{axis} {value}"
        for axis, cells in result["breakdowns_point_values"].items()
        for value, c in cells.items()
        if c["minority_below_10"]
    ]
    print("Breakdown cells flagged for a minority class below 10: " + (", ".join(flagged) if flagged else "none"))


if __name__ == "__main__":
    main()

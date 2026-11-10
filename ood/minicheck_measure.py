"""Measure a bounded MiniCheck judge on a few documents: peak memory, duration, scores.

The scores are compared with those of a run of the stock judge on the same device (the
stopped run ``ood-runs/minicheck-mps``), so the stock judge is not run again. Three groups
of calls, taken in one dataset and measured in this order:

    controls   the first calls of the file whose batches the rule does not split: the forward
               passes are those of the stock judge, the scores must be exactly equal
    longest    every claim of the document with the most words
    heaviest   the call whose stock batch is the largest (rows * padded_length ** 2)

For ``longest`` and ``heaviest`` the batches are split: the rows are the same but they go
through the model in other groups, and float32 results can move in the last digits. They
must stay within ``--tolerance``.

Memory: the physical footprint of the process (the figure of Activity Monitor and of
``/usr/bin/time -l``, GPU memory included), its lifetime maximum read after each group, and
the memory held by the Metal driver just before each release of the cache.

Nothing but identifiers, counts, durations, memory and score differences is printed or
written: no text, no label, no score.

    PYTHONPATH=ood python ood/minicheck_measure.py --data ood-out/llm-aggrefact-5.jsonl --reference ood-runs/minicheck-mps --judge minicheck_bounded:MiniCheckBoundedMps --dataset WiCE --out ood-runs/measure-minicheck-bounded-wice.json

With ``--compare-run`` no model is loaded: the scores of a finished run are compared with
those of the reference run on the identifiers they share.

    python ood/minicheck_measure.py --reference ood-runs/minicheck-mps --compare-run ood-runs/minicheck-bounded-mps --thresholds 0.04438,0.5
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import datetime as dt
import importlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

CRITERION = "faithfulness"
GROUPS = ("controls", "longest", "heaviest")
_LIBPROC: Any = None


def footprint() -> tuple[int | None, int | None]:
    """(current, lifetime maximum) physical memory footprint of this process, in bytes; macOS only.

    ``rusage_info_v4`` starts with a 16-byte uuid followed by 8-byte fields:
    ``ri_phys_footprint`` is the 8th and ``ri_lifetime_max_phys_footprint`` the 29th.
    """
    global _LIBPROC
    if sys.platform != "darwin":
        return None, None
    if _LIBPROC is None:
        _LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib")
    buf = (ctypes.c_uint64 * 64)()
    if _LIBPROC.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf)) != 0:
        return None, None
    return int(buf[2 + 7]), int(buf[2 + 28])


def gb(value: int | None) -> str:
    return "n/a" if value is None else f"{value / 1e9:.2f} GB"


def read_reference(run: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Header and last successful attempt per id of a ``judge-bench`` run."""
    header: dict[str, Any] = {}
    rows: dict[str, dict[str, Any]] = {}
    with (run / "attempts.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "header" in row and "sample_id" not in row:
                header = row
            elif row.get("ok") and row.get("result") is not None:
                rows[str(row["sample_id"])] = {
                    "score": float(row["result"]["score"]),
                    "tokens": row["trace"]["prompt_tokens"],
                    "truncated": bool(row["trace"]["context_truncated"]),
                    "latency_s": row["trace"]["latency_ms"] / 1000.0,
                }
    return header, rows


def compare_runs(reference: Path, other: Path, thresholds: list[float], tolerance: float) -> int:
    """Scores of two runs on the identifiers they share. No model is loaded."""
    ref_header, ref = read_reference(reference)
    new_header, new = read_reference(other)
    common = [sample for sample in ref if sample in new]
    if not common:
        raise SystemExit("the two runs share no identifier")
    diffs = [abs(ref[sample]["score"] - new[sample]["score"]) for sample in common]
    summary = {
        "reference": str(reference),
        "reference_config_hash": ref_header.get("config_hash"),
        "run": str(other),
        "run_config_hash": new_header.get("config_hash"),
        "common": len(common),
        "exactly_equal": sum(d == 0.0 for d in diffs),
        "max_abs_difference": max(diffs),
        "above_tolerance": sum(d > tolerance for d in diffs),
        "tolerance": tolerance,
        "tokens_equal": sum(ref[s]["tokens"] == new[s]["tokens"] for s in common),
        "truncation_equal": sum(ref[s]["truncated"] == new[s]["truncated"] for s in common),
        "decisions_changed": {
            str(t): sum((ref[s]["score"] < t) != (new[s]["score"] < t) for s in common) for t in thresholds
        },
    }
    print(json.dumps(summary, indent=1))
    same = summary["above_tolerance"] == 0 and not any(summary["decisions_changed"].values())
    return 0 if same else 1


def official_batches(judge: Any, context: str, claim: str) -> list[tuple[int, int]]:
    """(rows, padded length) of each batch the stock judge would send for this call."""
    from ragnarok_ai.judges.minicheck import FLAN_PREFIX, OFFICIAL_BATCH_SIZE

    stock = judge.backend.stock
    eos = stock.tokenizer.eos_token
    texts = [FLAN_PREFIX + eos.join([chunk, claim]) for chunk in judge.chunks(context)]
    lengths = (
        [len(ids) for ids in stock.tokenizer(texts, padding=False, truncation=False)["input_ids"]] if texts else []
    )
    limit = stock.checkpoint.max_model_len
    batches = []
    for start in range(0, len(lengths), OFFICIAL_BATCH_SIZE):
        part = lengths[start : start + OFFICIAL_BATCH_SIZE]
        batches.append((len(part), min(max(part), limit)))
    return batches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reference", type=Path, required=True, help="run of the stock judge on the same device")
    parser.add_argument("--compare-run", type=Path, help="compare this finished run with --reference and stop")
    parser.add_argument("--thresholds", default="0.5", help="thresholds whose decisions are compared (--compare-run)")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--judge", default="minicheck_bounded:MiniCheckBoundedMps", help="<module>:<Class>")
    parser.add_argument("--dataset", default="WiCE")
    parser.add_argument("--controls", type=int, default=20, help="number of control calls, 0 for none")
    parser.add_argument("--skip-heaviest", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-5, help="largest score difference on split batches")
    parser.add_argument("--footprint-limit-gb", type=float, default=8.0, help="half of the memory of a 16 GB machine")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    if args.compare_run is not None:
        return compare_runs(
            args.reference, args.compare_run, [float(t) for t in args.thresholds.split(",")], args.tolerance
        )
    if args.data is None or args.out is None:
        parser.error("--data and --out are required to measure")
    if args.out.exists():
        raise SystemExit(f"{args.out} already exists; nothing is overwritten")

    ref_header, reference = read_reference(args.reference)
    records = []
    with args.data.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["dataset"] == args.dataset:
                records.append(
                    {key: record[key] for key in ("id", "source_id", "context", "sentence")}
                    | {"words": len(record["context"].split())}
                )
    if not records:
        raise SystemExit(f"no record of {args.dataset} in {args.data}")

    module_name, _, class_name = args.judge.rpartition(":")
    started_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    load_started = time.perf_counter()
    judge = getattr(importlib.import_module(module_name), class_name)()
    load_s = time.perf_counter() - load_started
    backend = judge.backend
    torch = backend.stock._torch
    on_mps = judge.device.split(":", 1)[0] == "mps"
    rows_rule = importlib.import_module(module_name).forward_rows

    driver_readings: list[int] = []
    original_release = backend.release

    def release_with_reading() -> None:
        if on_mps:
            driver_readings.append(int(torch.mps.driver_allocated_memory()))
        original_release()

    backend.release = release_with_reading
    footprint_after_load = footprint()[0]

    # Selection, on lengths only
    for record in records:
        record["batches"] = official_batches(judge, record["context"], record["sentence"])
        record["cells"] = max((rows * length * length for rows, length in record["batches"]), default=0)
        record["split"] = any(rows > rows_rule(length) for rows, length in record["batches"])
    longest_source = max(records, key=lambda r: (r["words"], r["id"]))["source_id"]
    selection: dict[str, list[dict[str, Any]]] = {name: [] for name in GROUPS}
    selection["longest"] = [r for r in records if r["source_id"] == longest_source]
    if not args.skip_heaviest:
        heaviest = max(records, key=lambda r: (r["cells"], r["id"]))
        if heaviest["source_id"] != longest_source:
            selection["heaviest"] = [heaviest]
    taken = {r["id"] for group in selection.values() for r in group}
    for record in records:
        if len(selection["controls"]) >= args.controls:
            break
        if record["id"] in reference and record["id"] not in taken and not record["split"]:
            selection["controls"].append(record)

    calls: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {}
    for name in GROUPS:
        group_started = time.perf_counter()
        for record in selection[name]:
            driver_readings.clear()
            passes_before = backend.forward_passes
            call_started = time.perf_counter()
            result = asyncio.run(
                judge.judge(CRITERION, question="", context=record["context"], answer=record["sentence"])
            )
            latency = time.perf_counter() - call_started
            current, lifetime_max = footprint()
            ref = reference.get(record["id"])
            calls.append(
                {
                    "group": name,
                    "id": record["id"],
                    "words": record["words"],
                    "chunks": sum(rows for rows, _ in record["batches"]),
                    "stock_batches": record["batches"],
                    "stock_attention_gib": record["cells"] * 16 * 4 / 2**30,
                    "split": record["split"],
                    "forward_passes": backend.forward_passes - passes_before,
                    "latency_s": latency,
                    "stock_latency_s": None if ref is None else ref["latency_s"],
                    "score_difference": None if ref is None else abs(result.score - ref["score"]),
                    "tokens_equal": None if ref is None else result.trace.prompt_tokens == ref["tokens"],
                    "truncation_equal": None
                    if ref is None
                    else bool(result.trace.context_truncated) == ref["truncated"],
                    "driver_before_release_max": max(driver_readings, default=None),
                    "footprint_after_call": current,
                    "footprint_lifetime_max": lifetime_max,
                }
            )
        mine = [c for c in calls if c["group"] == name]
        compared = [c for c in mine if c["score_difference"] is not None]
        groups[name] = {
            "calls": len(mine),
            "compared_with_reference": len(compared),
            "chunks": sum(c["chunks"] for c in mine),
            "forward_passes": sum(c["forward_passes"] for c in mine),
            "duration_s": time.perf_counter() - group_started,
            "stock_duration_s": sum(c["stock_latency_s"] for c in compared) if compared else None,
            "exactly_equal": sum(c["score_difference"] == 0.0 for c in compared),
            "max_score_difference": max((c["score_difference"] for c in compared), default=None),
            "tokens_and_truncation_equal": all(c["tokens_equal"] and c["truncation_equal"] for c in compared),
            "largest_stock_attention_gib": max((c["stock_attention_gib"] for c in mine), default=None),
            "driver_before_release_max": max(
                (c["driver_before_release_max"] for c in mine if c["driver_before_release_max"] is not None),
                default=None,
            ),
            "footprint_lifetime_max_after_group": footprint()[1],
        }

    peak = footprint()[1]
    checks = []
    controls = groups["controls"]
    if controls["compared_with_reference"]:
        checks.append(
            {
                "name": "controls: scores exactly equal to the stock run",
                "value": f"{controls['exactly_equal']} of {controls['compared_with_reference']}",
                "met": controls["exactly_equal"] == controls["compared_with_reference"],
            }
        )
    for name in ("longest", "heaviest"):
        group = groups[name]
        if group["compared_with_reference"]:
            checks.append(
                {
                    "name": f"{name}: largest score difference with the stock run at most {args.tolerance:g}",
                    "value": f"{group['max_score_difference']:.3g} ({group['exactly_equal']} of {group['compared_with_reference']} exactly equal)",
                    "met": group["max_score_difference"] <= args.tolerance,
                }
            )
    compared_groups = [g for g in groups.values() if g["compared_with_reference"]]
    checks.append(
        {
            "name": "tokens sent and truncation flag equal to the stock run",
            "value": str(all(g["tokens_and_truncation_equal"] for g in compared_groups)),
            "met": bool(compared_groups) and all(g["tokens_and_truncation_equal"] for g in compared_groups),
        }
    )
    if peak is not None:
        checks.append(
            {
                "name": f"peak memory footprint below {args.footprint_limit_gb:g} GB",
                "value": gb(peak),
                "met": peak < args.footprint_limit_gb * 1e9,
            }
        )
    summary = {
        "judge": args.judge,
        "config_hash": judge.config(CRITERION).config_hash,
        "device": judge.device,
        "reference": str(args.reference),
        "reference_config_hash": ref_header.get("config_hash"),
        "dataset": args.dataset,
        "started_at": started_at,
        "load_s": load_s,
        "torch": torch.__version__,
        "memory": {
            "footprint_after_load": footprint_after_load,
            "footprint_lifetime_max": peak,
            "max_resident": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "groups": groups,
        "checks": checks,
        "calls": calls,
    }
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, indent=1) + "\n")

    print(
        f"judge {args.judge} on {judge.device}, loaded in {load_s:.1f} s, footprint after loading {gb(footprint_after_load)}"
    )
    for name in GROUPS:
        group = groups[name]
        if not group["calls"]:
            continue
        stock = "n/a" if group["stock_duration_s"] is None else f"{group['stock_duration_s']:.1f} s"
        diff = "n/a" if group["max_score_difference"] is None else f"{group['max_score_difference']:.3g}"
        print(
            f"{name:9s} calls {group['calls']:3d}  chunks {group['chunks']:4d}  forward passes {group['forward_passes']:4d}  "
            f"duration {group['duration_s']:.1f} s (stock run: {stock})  largest score difference {diff}  "
            f"driver before release {gb(group['driver_before_release_max'])}  "
            f"peak footprint so far {gb(group['footprint_lifetime_max_after_group'])}"
        )
    for check in checks:
        print(f"{'met   ' if check['met'] else 'MISSED'}  {check['name']}: {check['value']}")
    print(f"written: {args.out}")
    return 0 if all(check["met"] for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

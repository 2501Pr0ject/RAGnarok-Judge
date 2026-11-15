"""Ids of the CONTRADICTED sentences whose annotator note matches the attribution patterns.

TARGETS.md, section 2 (training set rule 1: "the 115 candidate sentences were found by
patterns on the notes") and section 11 (14 of the 139 CONTRADICTED test sentences carry an
annotator note that matches the attribution patterns; the recall on CONTRADICTED outside
them is a secondary measure of section 6, ``--exclude-ids`` of ``ragtruth_analysis.py``).

    python3 analysis/attribution_ids.py --trace ragtruth-out/trace.jsonl --split test --check-split ragtruth-out/test.jsonl --expect 14 --out ATTRIBUTION_IDS.txt

The notes are the ``meta`` field of the annotated spans attached to each sentence, as the
converter writes them in ``trace.jsonl``. A sentence is a candidate when the note of any of
its attached spans matches one of ``PATTERNS``, case ignored. Only the lines of the split
asked are decoded. The script prints counts and writes ids only, one per line, sorted; it
stops in error, writing nothing, when the number found differs from ``--expect``.

``PATTERNS`` are the patterns of 2026-11-09, recorded with each of the 115 training candidates
when they were found (the patterns that matched each note). Applied with ``re.IGNORECASE`` to
the CONTRADICTED training sentences, they give the same 115 sentences, no more and no fewer.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PATTERNS = (
    r"from passage \d",
    r"in passage \d",
    r"mentioned in passage",
    r"not in passage",
    r"passage \d+\s*:",
    r"passage \d+ (does not|doesn't|did not|didn't)",
)
COMPILED = tuple(re.compile(pattern, re.IGNORECASE) for pattern in PATTERNS)
LABEL = "CONTRADICTED"


def matches(note: str | None) -> bool:
    return bool(note) and any(pattern.search(note) for pattern in COMPILED)


def candidates(trace: Path, split: str, label: str = LABEL) -> tuple[set[str], set[str]]:
    """(ids of the sentences of ``split`` with ``label``, the candidates among them)."""
    marker = f'"split": "{split}"'
    pool: set[str] = set()
    found: set[str] = set()
    with trace.open(encoding="utf-8") as handle:
        for line in handle:
            if marker not in line:  # lines of other splits are not decoded
                continue
            record = json.loads(line)
            if record["split"] != split or record["label"] != label:
                continue
            pool.add(record["id"])
            if any(matches(span.get("meta")) for span in record["spans"]):
                found.add(record["id"])
    return pool, found


def read_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--trace", type=Path, required=True, help="trace.jsonl of the converter")
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--check-split", type=Path, help="the split file: its CONTRADICTED ids must be the pool")
    parser.add_argument("--drop-ids", type=Path, help="ids left out before counting, one per line")
    parser.add_argument("--expect", type=int, required=True, help="number of ids expected; any other stops")
    parser.add_argument("--out", type=Path, help="ids file to write, must not exist")
    args = parser.parse_args()
    if args.out is not None and args.out.exists():
        raise SystemExit(f"{args.out} exists; nothing is overwritten")

    pool, found = candidates(args.trace, args.split)
    if args.check_split is not None:
        with args.check_split.open(encoding="utf-8") as handle:
            listed = {r["id"] for r in map(json.loads, filter(str.strip, handle)) if r["label"] == LABEL}
        if listed != pool:
            raise SystemExit(f"{args.check_split}: {len(listed)} {LABEL} ids, {len(pool)} in the trace for {args.split}")
    if args.drop_ids is not None:
        dropped = read_ids(args.drop_ids)
        pool, found = pool - dropped, found - dropped
    print(f"{args.split}: {len(pool)} {LABEL} sentences, {len(found)} with a note matching the attribution patterns")
    if len(found) != args.expect:
        raise SystemExit(f"{len(found)} ids found, {args.expect} expected: nothing written")
    if args.out is not None:
        with args.out.open("x", encoding="utf-8") as handle:
            handle.write("".join(f"{sample}\n" for sample in sorted(found)))
        print(f"{args.out}: {len(found)} ids")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

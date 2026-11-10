"""Apply an exclusion rule of TARGETS.md (section 2, training set rules) to the training file.

Two rules, never relabelling:

- rule 5 (default): the CONTRADICTED training sentences whose label is inherited from a span
  covering several sentences (converter rule ``conflict_span+multi_sentence_span``);
- rule 1 (``--ids``): the sentences of a validated list of ids, one per line; every id must be
  in the input file, and the file is expected to hold CONTRADICTED sentences only.

The input file is never modified. Every kept line is copied byte for byte, in the same
order. The outputs may not exist beforehand:

    --out        the training file without the excluded sentences
    --ids-out    the excluded ids, one per line, in the order of the input file, no text
                 (rule 5; with --ids the list given is the record)
    <out>.manifest.json   hashes, counts and the rule, next to --out

Only the training file is concerned: dev and test are frozen.

    python3 conversion/train_exclusions.py --train ragtruth-out/train.jsonl --out ragtruth-out/train-rule5.jsonl --ids-out conversion/train_excluded_ids_rule5.txt
    python3 conversion/train_exclusions.py --train ragtruth-out/train-rule5.jsonl --ids conversion/train_excluded_ids_rule1.txt --out ragtruth-out/train-rule5-rule1.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from collections import Counter
from pathlib import Path

EXCLUDED_LABEL = "CONTRADICTED"
EXCLUDED_RULE = "conflict_span+multi_sentence_span"
LABELS = ("SUPPORTED", "UNSUPPORTED", "CONTRADICTED")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ids-out", type=Path, help="required without --ids")
    parser.add_argument("--ids", type=Path, help="rule 1: the validated ids to exclude, one per line")
    args = parser.parse_args()
    if (args.ids is None) == (args.ids_out is None):
        raise SystemExit("give --ids-out (rule 5) or --ids (rule 1), not both")

    listed: set[str] = set()
    if args.ids is not None:
        listed = {line.strip() for line in args.ids.read_text(encoding="utf-8").splitlines() if line.strip()}
        if not listed:
            raise SystemExit(f"{args.ids} lists no id")

    manifest_path = args.out.with_name(args.out.name + ".manifest.json")
    targets = (args.out, manifest_path) if args.ids_out is None else (args.out, args.ids_out, manifest_path)
    if args.train.resolve() in {path.resolve() for path in targets}:
        raise SystemExit("an output path is the input file")
    for path in targets:
        if path.exists():
            raise SystemExit(f"{path} already exists; nothing is overwritten")

    before: Counter[str] = Counter()
    after: Counter[str] = Counter()
    sources_before: set[str] = set()
    sources_after: set[str] = set()
    seen: set[str] = set()
    excluded: list[str] = []
    kept: list[bytes] = []
    with args.train.open("rb") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                raise SystemExit(f"line {number} of {args.train} is empty")
            record = json.loads(line)
            if record["label"] not in LABELS:
                raise SystemExit(f"line {number}: unexpected label {record['label']!r}")
            if record["id"] in seen:
                raise SystemExit(f"line {number}: duplicate id {record['id']}")
            seen.add(record["id"])
            before[record["label"]] += 1
            sources_before.add(record["source_id"])
            if args.ids is not None:
                out = record["id"] in listed
                if out and record["label"] != EXCLUDED_LABEL:
                    raise SystemExit(
                        f"line {number}: listed id {record['id']} is {record['label']}, not {EXCLUDED_LABEL}"
                    )
            else:
                out = record["label"] == EXCLUDED_LABEL and record["rule"] == EXCLUDED_RULE
            if out:
                excluded.append(record["id"])
                continue
            after[record["label"]] += 1
            sources_after.add(record["source_id"])
            kept.append(line)

    if args.ids is not None and set(excluded) != listed:
        missing = sorted(listed - set(excluded))
        raise SystemExit(f"{len(missing)} listed ids are not in {args.train}: {', '.join(missing[:10])}")

    with args.out.open("xb") as handle:
        handle.writelines(kept)
    ids_path = args.ids if args.ids is not None else args.ids_out
    if args.ids_out is not None:
        with args.ids_out.open("x", encoding="utf-8") as handle:
            handle.write("".join(f"{sample}\n" for sample in excluded))

    manifest = {
        "rule": "TARGETS.md, training set rule 1, validated list" if args.ids else "TARGETS.md, training set rule 5",
        "excluded_when": {"id_in": str(args.ids)} if args.ids else {"label": EXCLUDED_LABEL, "rule": EXCLUDED_RULE},
        "source": str(args.train),
        "source_sha256": sha256(args.train),
        "output": str(args.out),
        "output_sha256": sha256(args.out),
        "excluded_ids": str(ids_path),
        "excluded_ids_sha256": sha256(ids_path),
        "written_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "sentences": {"before": sum(before.values()), "after": sum(after.values()), "excluded": len(excluded)},
        "labels_before": {label: before[label] for label in LABELS},
        "labels_after": {label: after[label] for label in LABELS},
        "sources": {"before": len(sources_before), "after": len(sources_after)},
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()

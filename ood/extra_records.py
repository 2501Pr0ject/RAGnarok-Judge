"""Records of one JSONL file whose id is absent from another: lines copied as they are.

Used to score, in a small complementary run, the records that an earlier version of a
converted file did not hold.

    python3 ood/extra_records.py --new ood-out/llm-aggrefact-5-v2.jsonl --old ood-out/llm-aggrefact-5.jsonl --out ood-out/llm-aggrefact-5-v2-extra.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} already exists")
    with args.old.open(encoding="utf-8") as handle:
        old_ids = {json.loads(line)["id"] for line in handle if line.strip()}
    kept = 0
    with args.new.open(encoding="utf-8") as source, args.out.open("w", encoding="utf-8") as target:
        for line in source:
            if line.strip() and json.loads(line)["id"] not in old_ids:
                target.write(line)
                kept += 1
    print(f"{kept} records of {args.new} are not in {args.old}; written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

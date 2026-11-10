"""Convert the test split of LLM-AggreFact to the JSONL format of ``ragnarok judge-bench``.

Only five datasets are kept: WiCE, REVEAL, ClaimVerify, FactCheck-GPT and ExpertQA. The dev
split of the benchmark is never read. The output stays local: the benchmark is licensed
CC BY-ND 4.0 and no converted file is ever published.

One record per claim:

    id            agf-<dataset>-<row of the test file>
    source_id     <dataset>-<hash of the document>, the group for the bootstrap
    context       the document
    sentence      the claim
    label         SUPPORTED (benchmark label 1) or UNSUPPORTED (label 0)
    dataset       the dataset name
    doc_words     number of whitespace-separated words of the document
    length_band   words_0000_0250, words_0251_1000, words_1001_2500 or words_2501_plus

Instances whose document overlaps the training contexts are removed and counted. The list
comes from ``check_overlap.py``, which has to be run first on the same test file.

Run with pyarrow available, for instance:

    uv run --no-project --python 3.12 --with pyarrow --with huggingface_hub python ood/convert_llm_aggrefact.py --overlap ood-out/overlap.json --out ood-out/llm-aggrefact-5.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

REPO = "lytang/LLM-AggreFact"
REVISION = "981dfd0bd8e58e7238a9ab92b2e6ea44bce918e4"
TEST_FILE = "data/test-00000-of-00001.parquet"
TEST_SHA256 = "ba6976c3da990a9d7b11670efaf4e3c93c1bd0530061ec742afd6b2d5bce698c"
KEEP = {
    "wice": "WiCE",
    "reveal": "REVEAL",
    "claimverify": "ClaimVerify",
    "factcheck-gpt": "FactCheck-GPT",
    "expertqa": "ExpertQA",
}
BANDS = ((250, "words_0000_0250"), (1000, "words_0251_1000"), (2500, "words_1001_2500"))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def band(words: int) -> str:
    for limit, name in BANDS:
        if words <= limit:
            return name
    return "words_2501_plus"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--parquet", type=Path, help="local test file; default: the pinned revision from the Hub cache")
    parser.add_argument("--overlap", type=Path, required=True, help="report written by check_overlap.py")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import pyarrow.parquet as pq

    source = args.parquet
    if source is None:
        from huggingface_hub import hf_hub_download

        source = Path(hf_hub_download(REPO, TEST_FILE, repo_type="dataset", revision=REVISION))
    digest = sha256_of(source)
    if digest != TEST_SHA256:
        raise SystemExit(f"unexpected test file: sha256 {digest}, expected {TEST_SHA256}")
    if args.out.exists():
        raise SystemExit(f"{args.out} already exists; remove it first to convert again")

    overlap = json.loads(args.overlap.read_text(encoding="utf-8"))
    if overlap["source_sha256"] != digest:
        raise SystemExit(f"{args.overlap} was computed on another test file")
    threshold = overlap["rule"]["threshold"]
    overlapping = {key for key, value in overlap["containment"].items() if value >= threshold}

    table = pq.read_table(source, columns=["dataset", "doc", "claim", "label"]).to_pydict()
    counts: dict[str, dict[str, int]] = {}
    documents: dict[str, set[str]] = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row, (name, doc, claim, label) in enumerate(
            zip(table["dataset"], table["doc"], table["claim"], table["label"], strict=True)
        ):
            dataset = KEEP.get(name.lower())
            if dataset is None:
                continue
            stats = counts.setdefault(dataset, {"read": 0, "written": 0, "overlap_removed": 0, "empty_removed": 0})
            stats["read"] += 1
            if label not in (0, 1):
                raise SystemExit(f"row {row}: unexpected label {label!r}")
            if not claim or not claim.strip() or not doc or not doc.strip():
                stats["empty_removed"] += 1
                continue
            doc_hash = hashlib.sha1(doc.encode("utf-8")).hexdigest()[:16]
            if f"{dataset.lower()}-{doc_hash}" in overlapping:
                stats["overlap_removed"] += 1
                continue
            words = len(doc.split())
            record = {
                "id": f"agf-{dataset.lower()}-{row:05d}",
                "source_id": f"{dataset.lower()}-{doc_hash}",
                "context": doc,
                "sentence": claim,
                "label": "SUPPORTED" if label == 1 else "UNSUPPORTED",
                "dataset": dataset,
                "doc_words": words,
                "length_band": band(words),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["written"] += 1
            documents.setdefault(dataset, set()).add(doc_hash)

    manifest = {
        "source": f"{REPO}@{REVISION}:{TEST_FILE}",
        "source_sha256": digest,
        "output": str(args.out),
        "output_sha256": sha256_of(args.out),
        "converted_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "overlap_report": str(args.overlap),
        "overlap_rule": overlap["rule"],
        "overlap_train_sha256": overlap["train_sha256"],
        "datasets": {name: {**stats, "documents": len(documents.get(name, ()))} for name, stats in counts.items()},
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    for name, stats in manifest["datasets"].items():
        print(
            f"{name}: {stats['written']} records, {stats['documents']} documents, "
            f"{stats['overlap_removed']} removed for overlap, {stats['empty_removed']} removed as empty"
        )
    print(f"written: {args.out} (sha256 {manifest['output_sha256'][:16]}), manifest next to it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

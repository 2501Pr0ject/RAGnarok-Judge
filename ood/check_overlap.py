"""Overlap between the documents of LLM-AggreFact and the training contexts of the judge.

Label-free: only the ``dataset`` and ``doc`` columns of the benchmark's test split are read,
and the ``context`` field of a JSONL training file. The dev split is never read.

Measure. Texts are lower-cased and reduced to their words. The containment of a document is
the share of its 13-word shingles that occur in at least one training context. A shared
shingle is 13 consecutive identical words, which does not happen by chance.

A document shorter than 13 words has no shingle. It overlaps only if it is identical, word
for word, to a whole training context; finding its one or two words somewhere inside a
context means nothing. Such documents are counted as too short to assess.

Rule. A document whose containment reaches the threshold (0.5 by default) is listed as
overlapping; the converter removes its instances and counts them. Documents sharing at least
one shingle are counted and reported as well.

Output: a JSON file with the rule, the hashes of the inputs, the counts per dataset and the
containment of every document that shares at least one shingle. No text is written.

    uv run --no-project --python 3.12 --with pyarrow --with huggingface_hub python ood/check_overlap.py --train ragtruth-out/train.jsonl --out ood-out/overlap.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
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
SHINGLE_WORDS = 13
WORD = re.compile(r"\w+")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def words_of(text: str) -> list[str]:
    return WORD.findall(text.lower())


def shingles(words: list[str]) -> list[str]:
    return [" ".join(words[i : i + SHINGLE_WORDS]) for i in range(len(words) - SHINGLE_WORDS + 1)]


def document_key(dataset: str, doc: str) -> str:
    """The same key as the converter's ``source_id``."""
    return f"{dataset.lower()}-{hashlib.sha1(doc.encode('utf-8')).hexdigest()[:16]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--train", type=Path, required=True, help="JSONL training file with a 'context' field")
    parser.add_argument("--parquet", type=Path, help="local test file; default: the pinned revision from the Hub cache")
    parser.add_argument("--threshold", type=float, default=0.5, help="containment from which a document overlaps")
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
        raise SystemExit(f"{args.out} already exists; remove it first to check again")

    contexts: set[str] = set()
    with args.train.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                contexts.add(json.loads(line)["context"])
    train_shingles: set[str] = set()
    whole_contexts: set[str] = set()
    for context in contexts:
        words = words_of(context)
        train_shingles.update(shingles(words))
        whole_contexts.add(" ".join(words))

    table = pq.read_table(source, columns=["dataset", "doc"]).to_pydict()
    containment: dict[str, float] = {}
    too_short: set[str] = set()
    instances: dict[str, int] = {}
    per_dataset: dict[str, set[str]] = {}
    for name, doc in zip(table["dataset"], table["doc"], strict=True):
        dataset = KEEP.get(name.lower())
        if dataset is None:
            continue
        key = document_key(dataset, doc)
        instances[key] = instances.get(key, 0) + 1
        per_dataset.setdefault(dataset, set()).add(key)
        if key in containment:
            continue
        words = words_of(doc)
        if not words:
            containment[key] = 0.0
        elif len(words) < SHINGLE_WORDS:
            # Rule for documents of fewer than 13 words: no shingle, so no containment to measure.
            # Such a document overlaps only when it equals a whole training context, never because
            # its few words occur somewhere inside one.
            too_short.add(key)
            containment[key] = 1.0 if " ".join(words) in whole_contexts else 0.0
        else:
            own = shingles(words)
            containment[key] = sum(1 for shingle in own if shingle in train_shingles) / len(own)

    datasets = {}
    for dataset, keys in per_dataset.items():
        shared = [key for key in keys if containment[key] > 0]
        over = [key for key in keys if containment[key] >= args.threshold]
        datasets[dataset] = {
            "documents": len(keys),
            "instances": sum(instances[key] for key in keys),
            "documents_too_short_to_assess": sum(1 for key in keys if key in too_short),
            "documents_sharing_a_shingle": len(shared),
            "documents_overlapping": len(over),
            "instances_overlapping": sum(instances[key] for key in over),
            "max_containment": round(max((containment[key] for key in keys), default=0.0), 4),
        }
    report = {
        "rule": {"shingle_words": SHINGLE_WORDS, "threshold": args.threshold},
        "source": f"{REPO}@{REVISION}:{TEST_FILE}",
        "source_sha256": digest,
        "train": str(args.train),
        "train_sha256": sha256_of(args.train),
        "train_contexts": len(contexts),
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "datasets": datasets,
        "containment": {key: round(value, 4) for key, value in sorted(containment.items()) if value > 0},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for dataset, stats in datasets.items():
        print(
            f"{dataset}: {stats['documents']} documents, {stats['documents_too_short_to_assess']} too short to assess, "
            f"{stats['documents_sharing_a_shingle']} sharing a shingle, "
            f"{stats['documents_overlapping']} overlapping ({stats['instances_overlapping']} instances), "
            f"largest containment {stats['max_containment']}"
        )
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

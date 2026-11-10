#!/usr/bin/env python3
"""Convert RAGTruth span-level annotations into sentence-level grounding labels.

Target task (RAGnarok Judge v0): given a retrieved context, an optional
question and one sentence of a generated answer, predict whether the sentence
is SUPPORTED, CONTRADICTED or UNSUPPORTED by the context.

Conversion rules
----------------
Scope
    Only the QA and Summary tasks are converted. Data-to-text (structured
    JSON contexts) is out of scope for v0.
Response-level exclusion
    Responses whose ``quality`` is not ``good`` (``incorrect_refusal``,
    ``truncated``) are excluded entirely.
Segmentation
    Responses are split on line breaks, then into sentences with pysbd.
    Every sentence keeps its character offsets in the original response.
    A trailing citation marker ("(Passage 3)") is merged into the sentence
    it follows.
Span projection
    A span is attached to a sentence when they share at least
    ``--min-overlap`` alphanumeric characters, or all of the span's
    alphanumeric characters when the span is shorter (e.g. "55", "°F").
    A span covering several sentences (a fabricated passage) labels each of
    them.
Labels (priority CONTRADICTED > UNSUPPORTED > SUPPORTED)
    * any Evident/Subtle Conflict span            -> CONTRADICTED
    * any Evident/Subtle Baseless Info span       -> UNSUPPORTED
      (``implicit_true`` spans are kept: true in the world is not grounded)
    * no span                                     -> SUPPORTED
AMBIGUOUS (excluded, documented in the trace)
    * a span crossing into the sentence while covering less than
      ``--cross-min-coverage`` of it, with no other span inside the sentence
    * a sentence with no alphanumeric content
    * a standalone citation marker ("Passage 1:")
    * a fragment with fewer than ``--min-words`` words and no digit
      ("Sure!", "Ingredients:")

Outputs (in --out-dir)
    train.jsonl / dev.jsonl / test.jsonl   labelled sentences
    trace.jsonl                            every sentence, with rule or exclusion reason
    report.json / report.md                audit statistics
    audit_sheet.csv / audit_key.jsonl      blind manual audit sample

Usage
    python ragtruth_to_ragnarok.py convert --data-dir RAGTruth/dataset --out-dir out
    python ragtruth_to_ragnarok.py score-audit --out-dir out
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    import pysbd
except ImportError:  # pragma: no cover
    sys.exit("pysbd is required: pip install pysbd")

SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
UNSUPPORTED = "UNSUPPORTED"
LABELS = (SUPPORTED, CONTRADICTED, UNSUPPORTED)

CONFLICT_TYPES = {"Evident Conflict", "Subtle Conflict"}
BASELESS_TYPES = {"Evident Baseless Info", "Subtle Baseless Info"}
IN_SCOPE_TASKS = {"QA", "Summary"}

ALNUM = re.compile(r"[^\W_]", re.UNICODE)
# Citation markers emitted by QA generators, e.g. "(Passage 3)" or "Passage 1 and 2:".
CITATION_MARKER = re.compile(
    r"^[\(\[]?\s*(?:source|passage|reference)s?\s*\d+(?:\s*(?:,|&|and|-)\s*\d+)*\s*[\)\]]?\s*[:.]?$",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class Span:
    start: int
    end: int
    text: str
    label_type: str
    implicit_true: bool
    due_to_null: bool
    meta: str


@dataclass
class SentenceRecord:
    id: str
    source_id: str
    response_id: str
    task: str
    source_dataset: str
    generator: str
    split: str
    sent_index: int
    char_start: int
    char_end: int
    sentence: str
    prev_sentence: str | None
    spans: list[dict[str, Any]] = field(default_factory=list)
    label: str | None = None
    rule: str | None = None
    exclusion_reason: str | None = None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_context(source: dict[str, Any]) -> tuple[str | None, str]:
    """Return (question, context) for a RAGTruth source entry."""
    info = source["source_info"]
    if source["task_type"] == "QA":
        return info["question"].strip(), info["passages"].strip()
    if source["task_type"] == "Summary":
        return None, str(info).strip()
    raise ValueError(f"Unsupported task type: {source['task_type']}")


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #


class SentenceSplitter:
    def __init__(self, language: str = "en") -> None:
        self._seg = pysbd.Segmenter(language=language, clean=False)

    def split(self, text: str) -> list[tuple[int, int]]:
        """Return (start, end) offsets of non-empty sentences in ``text``."""
        offsets: list[tuple[int, int]] = []
        for line_match in re.finditer(r"[^\n]+", text):
            line = line_match.group()
            base = line_match.start()
            cursor = 0
            for segment in self._seg.segment(line):
                stripped = segment.strip()
                if not stripped:
                    continue
                pos = line.find(stripped, cursor)
                if pos < 0:  # defensive: segmenter altered the text
                    raise RuntimeError(f"Cannot locate segment {stripped!r}")
                start, end = base + pos, base + pos + len(stripped)
                # A trailing citation marker belongs to the sentence it cites.
                if CITATION_MARKER.match(stripped) and offsets and offsets[-1][1] >= base:
                    offsets[-1] = (offsets[-1][0], end)
                else:
                    offsets.append((start, end))
                cursor = pos + len(stripped)
        return offsets


def alnum_overlap(text: str, a: int, b: int, s: int, e: int) -> int:
    lo, hi = max(a, s), min(b, e)
    if lo >= hi:
        return 0
    return len(ALNUM.findall(text[lo:hi]))


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #


def is_attached(response: str, a: int, b: int, sp: Span, min_overlap: int) -> bool:
    """A span is attached when it shares enough alphanumeric characters with the sentence.

    Spans shorter than ``min_overlap`` (ages, years, units such as "55" or "°F")
    are attached as soon as all their alphanumeric characters fall in the sentence.
    """
    span_alnum = len(ALNUM.findall(response[sp.start:sp.end]))
    if span_alnum == 0:
        return False
    return alnum_overlap(response, a, b, sp.start, sp.end) >= min(min_overlap, span_alnum)


def crosses_boundary(response: str, a: int, b: int, sp: Span) -> bool:
    # Whitespace or punctuation outside the sentence does not count.
    left = alnum_overlap(response, sp.start, min(sp.end, a), sp.start, sp.end)
    right = alnum_overlap(response, max(sp.start, b), sp.end, sp.start, sp.end)
    return bool(left or right)


def label_sentence(
    rec: SentenceRecord,
    response: str,
    spans: list[Span],
    min_overlap: int,
    cross_min_coverage: float,
    min_words: int,
) -> None:
    a, b = rec.char_start, rec.char_end
    sentence_alnum = [i for i in range(a, b) if ALNUM.match(response[i])]
    if not sentence_alnum:
        rec.exclusion_reason = "no_alphanumeric_content"
        return
    if CITATION_MARKER.match(rec.sentence):
        rec.exclusion_reason = "citation_marker_only"
        return
    if len(re.findall(r"\w+", rec.sentence)) < min_words and not re.search(r"\d", rec.sentence):
        rec.exclusion_reason = "short_fragment"
        return

    attached = [sp for sp in spans if is_attached(response, a, b, sp, min_overlap)]
    crossing = [sp for sp in attached if crosses_boundary(response, a, b, sp)]
    rec.spans = [asdict(sp) for sp in attached]

    # A span that covers several whole sentences (a fabricated passage) labels
    # each of them. Only a span that crosses into the sentence while covering
    # little of it is an unreliable projection.
    if crossing:
        covered = {
            i for i in sentence_alnum for sp in crossing if sp.start <= i < sp.end
        }
        coverage = len(covered) / len(sentence_alnum)
        if coverage < cross_min_coverage:
            contained = [sp for sp in attached if sp not in crossing]
            if not contained:
                rec.exclusion_reason = "partial_span_crossing_sentence_boundary"
                return
            attached, crossing = contained, []
            rec.spans = [asdict(sp) for sp in attached]

    if not attached:
        rec.label, rec.rule = SUPPORTED, "no_span"
    elif any(sp.label_type in CONFLICT_TYPES for sp in attached):
        rec.label, rec.rule = CONTRADICTED, "conflict_span"
    elif all(sp.label_type in BASELESS_TYPES for sp in attached):
        rec.label = UNSUPPORTED
        rec.rule = (
            "baseless_span_implicit_true"
            if all(sp.implicit_true for sp in attached)
            else "baseless_span"
        )
    else:
        rec.exclusion_reason = "unknown_label_type"
        return

    if crossing:
        rec.rule = f"{rec.rule}+multi_sentence_span"


def assign_splits(
    responses: Iterable[dict[str, Any]], dev_ratio: float, seed: int
) -> dict[str, str]:
    """Map source_id -> train/dev/test. Official test is kept; dev is grouped by source."""
    official: dict[str, set[str]] = defaultdict(set)
    for r in responses:
        official[r["source_id"]].add(r["split"])
    mixed = [sid for sid, s in official.items() if len(s) > 1]
    if mixed:
        raise AssertionError(f"{len(mixed)} sources appear in both official splits")

    train_sources = sorted(sid for sid, s in official.items() if s == {"train"})
    rng = random.Random(seed)
    rng.shuffle(train_sources)
    n_dev = round(len(train_sources) * dev_ratio)
    dev = set(train_sources[:n_dev])

    mapping = {}
    for sid, s in official.items():
        mapping[sid] = "test" if s == {"test"} else ("dev" if sid in dev else "train")
    return mapping


def convert(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    responses = read_jsonl(data_dir / "response.jsonl")
    sources = {s["source_id"]: s for s in read_jsonl(data_dir / "source_info.jsonl")}

    stats: dict[str, Any] = {"params": {k: v for k, v in vars(args).items() if k != "func"}}
    resp_counter: Counter[str] = Counter()
    span_counter: Counter[str] = Counter()

    in_scope = []
    for r in responses:
        task = sources[r["source_id"]]["task_type"]
        resp_counter[f"total:{task}"] += 1
        if task not in IN_SCOPE_TASKS:
            resp_counter["excluded:out_of_scope_task"] += 1
            continue
        in_scope.append(r)

    split_of = assign_splits(in_scope, args.dev_ratio, args.seed)
    splitter = SentenceSplitter("en")

    records: list[SentenceRecord] = []
    contexts: dict[str, tuple[str | None, str]] = {}
    responses_with_spans_all_supported = 0

    for r in in_scope:
        src = sources[r["source_id"]]
        task = src["task_type"]
        contexts.setdefault(r["source_id"], build_context(src))
        text = r["response"]

        spans = [
            Span(
                l["start"], l["end"], l["text"], l["label_type"],
                bool(l["implicit_true"]), bool(l["due_to_null"]), l["meta"],
            )
            for l in r["labels"]
        ]
        for sp in spans:
            span_counter[f"{task}:{sp.label_type}"] += 1
            if sp.implicit_true:
                span_counter[f"{task}:implicit_true"] += 1
            if sp.due_to_null:
                span_counter[f"{task}:due_to_null"] += 1
            if text[sp.start:sp.end] != sp.text:
                span_counter[f"{task}:misaligned"] += 1

        response_excluded = r["quality"] != "good"
        resp_counter["kept" if not response_excluded else f"excluded:quality:{r['quality']}"] += 1

        offsets = splitter.split(text)
        covered_spans: set[int] = set()
        resp_records = []
        prev: str | None = None
        for i, (a, b) in enumerate(offsets):
            sentence = text[a:b]
            rec = SentenceRecord(
                id=f"rt-{r['id']}-{i}",
                source_id=r["source_id"],
                response_id=r["id"],
                task=task,
                source_dataset=src["source"],
                generator=r["model"],
                split=split_of[r["source_id"]],
                sent_index=i,
                char_start=a,
                char_end=b,
                sentence=sentence,
                prev_sentence=prev,
            )
            prev = sentence
            if response_excluded:
                rec.exclusion_reason = f"response_quality:{r['quality']}"
            else:
                label_sentence(
                    rec, text, spans, args.min_overlap, args.cross_min_coverage, args.min_words
                )
            for k, sp in enumerate(spans):
                if is_attached(text, a, b, sp, args.min_overlap):
                    covered_spans.add(k)
            resp_records.append(rec)

        if not response_excluded:
            for k, sp in enumerate(spans):
                if k not in covered_spans:
                    span_counter[f"{task}:not_attached_to_any_sentence"] += 1
            labelled = [x for x in resp_records if x.label]
            if spans and labelled and len(labelled) == len(resp_records) and all(
                x.label == SUPPORTED for x in labelled
            ):
                responses_with_spans_all_supported += 1
        records.extend(resp_records)

    # ------------------------------------------------------------------ #
    # Leakage check
    # ------------------------------------------------------------------ #
    split_sources: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        split_sources[rec.split].add(rec.source_id)
    for x, y in (("train", "dev"), ("train", "test"), ("dev", "test")):
        inter = split_sources[x] & split_sources[y]
        if inter:
            raise AssertionError(f"Source leakage between {x} and {y}: {len(inter)} sources")

    # ------------------------------------------------------------------ #
    # Write datasets and trace
    # ------------------------------------------------------------------ #
    writers = {s: (out_dir / f"{s}.jsonl").open("w", encoding="utf-8") for s in ("train", "dev", "test")}
    with (out_dir / "trace.jsonl").open("w", encoding="utf-8") as trace:
        for rec in records:
            trace.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            if rec.label is None:
                continue
            question, context = contexts[rec.source_id]
            example = {
                "id": rec.id,
                "source_id": rec.source_id,
                "response_id": rec.response_id,
                "task": rec.task,
                "source_dataset": rec.source_dataset,
                "generator": rec.generator,
                "question": question,
                "context": context,
                "prev_sentence": rec.prev_sentence,
                "sentence": rec.sentence,
                "label": rec.label,
                "rule": rec.rule,
            }
            writers[rec.split].write(json.dumps(example, ensure_ascii=False) + "\n")
    for w in writers.values():
        w.close()

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #
    labelled = [r for r in records if r.label]
    excluded = [r for r in records if r.exclusion_reason]

    def dist(rows: Iterable[SentenceRecord], key) -> dict[str, dict[str, int]]:
        out: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            out[key(row)][row.label] += 1
        return {k: dict(v) for k, v in sorted(out.items())}

    def pct(values: list[int]) -> dict[str, float]:
        if not values:
            return {}
        q = statistics.quantiles(values, n=100, method="inclusive")
        return {"p50": q[49], "p90": q[89], "p99": q[98], "max": max(values)}

    ctx_chars = defaultdict(list)
    for sid, (question, context) in contexts.items():
        ctx_chars[sources[sid]["task_type"]].append(len(context) + len(question or ""))

    stats.update(
        {
            "responses": dict(resp_counter),
            "spans": dict(sorted(span_counter.items())),
            "sentences": {
                "total_in_scope": len(records),
                "labelled": len(labelled),
                "excluded": len(excluded),
                "exclusion_reasons": dict(Counter(r.exclusion_reason for r in excluded)),
                "excluded_rate_in_good_responses": round(
                    sum(1 for r in excluded if not r.exclusion_reason.startswith("response_quality"))
                    / max(1, sum(1 for r in records if not (r.exclusion_reason or "").startswith("response_quality"))),
                    4,
                ),
                "rules": dict(Counter(r.rule for r in labelled)),
                "multi_span_sentences": sum(1 for r in labelled if len(r.spans) > 1),
                "mixed_conflict_and_baseless": sum(
                    1
                    for r in labelled
                    if {s["label_type"] for s in r.spans} & CONFLICT_TYPES
                    and {s["label_type"] for s in r.spans} & BASELESS_TYPES
                ),
            },
            "labels_by_split": dist(labelled, lambda r: r.split),
            "labels_by_task": dist(labelled, lambda r: r.task),
            "labels_by_source_dataset": dist(labelled, lambda r: r.source_dataset),
            "labels_by_generator": dist(labelled, lambda r: r.generator),
            "sources_by_split": {k: len(v) for k, v in sorted(split_sources.items())},
            "responses_with_spans_but_all_supported": responses_with_spans_all_supported,
            "context_chars_by_task": {k: pct(v) for k, v in ctx_chars.items()},
            "context_over_6000_chars_rate": {
                k: round(sum(1 for x in v if x > 6000) / len(v), 4) for k, v in ctx_chars.items()
            },
            "sentence_chars": pct([len(r.sentence) for r in labelled]),
            "leakage_check": "passed",
        }
    )
    (out_dir / "report.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(stats), encoding="utf-8")

    write_audit_sample(records, contexts, out_dir, args.audit_size, args.seed)
    print(f"Wrote {len(labelled)} labelled sentences to {out_dir}")


# --------------------------------------------------------------------------- #
# Manual audit
# --------------------------------------------------------------------------- #


def write_audit_sample(
    records: list[SentenceRecord],
    contexts: dict[str, tuple[str | None, str]],
    out_dir: Path,
    size: int,
    seed: int,
) -> None:
    """Blind sample from the train split, stratified by label (equal share per class)."""
    rng = random.Random(seed + 1)
    pool: dict[str, list[SentenceRecord]] = defaultdict(list)
    for r in records:
        if r.label and r.split == "train":
            pool[r.label].append(r)
    per_class = size // len(LABELS)
    sample: list[SentenceRecord] = []
    for label in LABELS:
        sample.extend(rng.sample(pool[label], min(per_class, len(pool[label]))))
    rng.shuffle(sample)

    with (out_dir / "audit_sheet.csv").open("w", encoding="utf-8", newline="") as fh, (
        out_dir / "audit_key.jsonl"
    ).open("w", encoding="utf-8") as key:
        w = csv.writer(fh)
        w.writerow(["audit_id", "task", "question", "context", "prev_sentence", "sentence", "your_label", "notes"])
        for i, r in enumerate(sample):
            question, context = contexts[r.source_id]
            w.writerow([i, r.task, question or "", context, r.prev_sentence or "", r.sentence, "", ""])
            key.write(json.dumps({"audit_id": i, "id": r.id, "label": r.label, "rule": r.rule}) + "\n")


def score_audit(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    key = {k["audit_id"]: k for k in read_jsonl(out_dir / "audit_key.jsonl")}
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    prevalence = report["labels_by_split"]["train"]
    total = sum(prevalence.values())

    alias = {"S": SUPPORTED, "C": CONTRADICTED, "U": UNSUPPORTED, "A": "AMBIGUOUS"}
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    with (out_dir / "audit_sheet.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            human = row["your_label"].strip().upper()
            if not human:
                continue
            human = alias.get(human, human)
            confusion[key[int(row["audit_id"])]["label"]][human] += 1

    if not confusion:
        sys.exit("No annotations found in the your_label column.")

    print("Per-class agreement (share of converted labels the annotator confirms):")
    weighted = 0.0
    for label in LABELS:
        row = confusion.get(label, Counter())
        n = sum(row.values())
        agree = row[label] / n if n else float("nan")
        weighted += agree * prevalence.get(label, 0) / total if n else 0.0
        print(f"  {label:<13} n={n:<3} agreement={agree:.2%}  human labels={dict(row)}")
    print(f"Prevalence-weighted agreement (train distribution): {weighted:.2%}")
    ambiguous = sum(row["AMBIGUOUS"] for row in confusion.values())
    annotated = sum(sum(row.values()) for row in confusion.values())
    print(f"Marked AMBIGUOUS by annotator: {ambiguous}/{annotated}")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def render_report(s: dict[str, Any]) -> str:
    def table(d: dict[str, dict[str, int]]) -> str:
        lines = ["| | " + " | ".join(LABELS) + " | Total |", "|---" * (len(LABELS) + 2) + "|"]
        for k, row in d.items():
            tot = sum(row.values())
            cells = [f"{row.get(l, 0)} ({row.get(l, 0) / tot:.1%})" for l in LABELS]
            lines.append(f"| {k} | " + " | ".join(cells) + f" | {tot} |")
        return "\n".join(lines)

    sent = s["sentences"]
    return f"""# RAGTruth → RAGnarok Judge conversion report

Parameters: `{json.dumps(s['params'])}`

## Responses
```
{json.dumps(s['responses'], indent=2)}
```

## Spans (in-scope tasks)
```
{json.dumps(s['spans'], indent=2)}
```

## Sentences
- In scope: {sent['total_in_scope']}
- Labelled: {sent['labelled']}
- Excluded: {sent['excluded']} — reasons: `{json.dumps(sent['exclusion_reasons'])}`
- Exclusion rate among sentences of good-quality responses: {sent['excluded_rate_in_good_responses']:.2%}
- Sentences with several spans: {sent['multi_span_sentences']}
- Sentences mixing conflict and baseless spans (resolved as CONTRADICTED): {sent['mixed_conflict_and_baseless']}
- Responses with spans but every sentence labelled SUPPORTED: {s['responses_with_spans_but_all_supported']}
- Rules: `{json.dumps(sent['rules'])}`

## Label distribution by split
{table(s['labels_by_split'])}

Sources per split: `{json.dumps(s['sources_by_split'])}` — leakage check: {s['leakage_check']}

## By task
{table(s['labels_by_task'])}

## By source dataset (licence audit)
{table(s['labels_by_source_dataset'])}

## By generator model
{table(s['labels_by_generator'])}

## Lengths
- Context + question characters by task: `{json.dumps(s['context_chars_by_task'])}`
- Share of contexts above 6,000 characters (~1,500 tokens): `{json.dumps(s['context_over_6000_chars_rate'])}`
- Sentence characters: `{json.dumps(s['sentence_chars'])}`
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("convert", help="convert RAGTruth into sentence-level labels")
    c.add_argument("--data-dir", required=True, help="directory with response.jsonl and source_info.jsonl")
    c.add_argument("--out-dir", required=True)
    c.add_argument("--dev-ratio", type=float, default=0.1, help="share of train sources moved to dev")
    c.add_argument("--seed", type=int, default=13)
    c.add_argument("--min-overlap", type=int, default=3, help="min shared alphanumeric chars to attach a span")
    c.add_argument(
        "--cross-min-coverage",
        type=float,
        default=0.25,
        help="a span crossing a sentence boundary labels the sentence only if it covers at least this "
        "share of its alphanumeric characters (0 = always label, >1 = never)",
    )
    c.add_argument(
        "--min-words",
        type=int,
        default=3,
        help="sentences with fewer words and no digit (\"Sure!\", \"Ingredients:\") are excluded",
    )
    c.add_argument("--audit-size", type=int, default=99, help="blind audit sample size (split equally by label)")
    c.set_defaults(func=convert)

    a = sub.add_parser("score-audit", help="score a completed audit_sheet.csv (labels S/C/U/A)")
    a.add_argument("--out-dir", required=True)
    a.set_defaults(func=score_audit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

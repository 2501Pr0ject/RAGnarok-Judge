"""Dev attempts of the retained judge, from the scores of its run (TRAINING.md, section 6).

The selected checkpoint is not scored on dev again: its dev scores are those of the scoring
pass made during its run (``dev_scores.json``). This script writes them as the
``dev_attempts.jsonl`` of a ``judge-bench`` run, next to the ``attempts.jsonl`` of the test run
made with ``rj_judge.RJQwen3``, so that ``analysis/ragtruth_analysis.py`` reads both: same
header, same configuration hash (it refuses a dev file whose hash differs from the test run's).

    python training/dev_attempts.py --dev DEV.jsonl --checkpoint RUN_DIR/checkpoint-3462 --out OUT_DIR/dev_attempts.jsonl

The scores are copied unchanged. Before writing, the criterion of section 12 of TARGETS.md is
recomputed from them and compared with the ``dev.json`` of the checkpoint. Each input is
tokenised, without any model, to record its length and whether it would be truncated, as the
harness does. The configuration hash needs ``ragnarok-ai`` (the harness environment); the
rows themselves are plain JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

EXPLANATION = "dev score of the run (dev_scores.json of the checkpoint), copied unchanged, not scored again"


def attempt_row(sample_id: str, score: float, prompt_tokens: int, truncated: bool, config_hash: str,
                criterion: str = "faithfulness") -> dict[str, Any]:
    """One line of an attempts file, in the format of ``judge-bench``. No verdict: only the score
    of the run is known, not its three probabilities."""
    return {
        "sample_id": sample_id,
        "criterion": criterion,
        "ok": True,
        "result": {"verdict": None, "score": score, "raw_score": None, "explanation": EXPLANATION, "output": None},
        "error": None,
        "trace": {"prompt_tokens": prompt_tokens, "completion_tokens": None, "context_truncated": truncated,
                  "latency_ms": None, "raw_output": None, "type": "CallTrace"},
        "config_hash": config_hash,
    }


def write_attempts(path: Path, header: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists; it is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for line in [header, *rows]:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dev", type=Path, required=True, help="dev.jsonl, the frozen dev file")
    parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoint folder of the retained judge")
    parser.add_argument("--out", type=Path, required=True, help="dev_attempts.jsonl to write, must not exist")
    args = parser.parse_args()

    from ragnarok_ai.benchmarks.judges import attempts_header

    import dev_eval
    import prompt_templates as pt
    import rj_judge

    adapters_sha, _ = rj_judge.checked_adapters(args.checkpoint)
    config = rj_judge.judge_config("faithfulness", adapters_sha)
    dev = dev_eval.ra.read_split(args.dev)
    scores = json.loads((args.checkpoint / "dev_scores.json").read_text(encoding="utf-8"))
    if set(scores) != set(dev):
        raise SystemExit("dev_scores.json does not cover the dev set exactly")
    if not all(isinstance(s, float) and math.isfinite(s) and 0.0 <= s <= 1.0 for s in scores.values()):
        raise SystemExit("dev_scores.json holds a score outside [0, 1]")
    expected = json.loads((args.checkpoint / "dev.json").read_text(encoding="utf-8"))
    point = dev_eval.point(dev, scores)
    for key in ("criterion", "pauc_U", "pauc_C"):
        if point[key] != expected[key]:
            raise SystemExit(f"{key}: {point[key]} recomputed, {expected[key]} in dev.json")

    tokenizer = pt.load_tokenizer(rj_judge.BASE)
    rows = []
    for sample_id in sorted(dev):
        record = dev[sample_id]
        ids, truncated = pt.build_input(rj_judge.BASE, tokenizer, record["context"], record["sentence"], pt.MAX_LENGTH)
        rows.append(attempt_row(sample_id, scores[sample_id], len(ids), truncated, config.config_hash))
    level = "sentence"
    header = attempts_header(config.config_hash, "faithfulness", level, sample=None, seed=None, stratify=None,
                             data_sha256=file_sha256(args.dev), source_commit=None)
    write_attempts(args.out, header, rows)
    print(f"{args.out}: {len(rows)} dev scores of {args.checkpoint.name}, criterion {point['criterion']:.4f} "
          f"(as in dev.json), {sum(r['trace']['context_truncated'] for r in rows)} truncated inputs, "
          f"config_hash {config.config_hash}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

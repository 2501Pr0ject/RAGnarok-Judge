#!/usr/bin/env python3
# ruff: noqa: E501  (the docstring keeps its example commands on one line each)
"""Cost of a sentence-level judge on this machine: latency per call and peak memory.

Regime: one process, one judge, one sentence per call, and the device cache released after
each call (``torch.mps.empty_cache()`` for PyTorch, ``mx.clear_cache()`` for MLX). In this
regime the memory figures reflect what a call needs, not the order and variety of the inputs.

Judges (``--judge``):

    hhem, minicheck      the ragnarok-ai judges, built the way ``ragnarok judge-bench`` builds them
    hf:<repo-or-path>    a causal language model loaded with transformers (candidate base)
    mlx:<repo-or-path>   the same with mlx-lm, in bfloat16 (converted after loading if needed)
    fake                 no model at all; checks the plumbing

A candidate base is measured with its pretrained weights, before any training: the cost of a
forward pass does not depend on the values of the weights. The score is read at the last
position only, over two label tokens, which is how the trained judge is expected to work. The
score itself means nothing here and is not written to the run files.

Outputs in ``--out``: ``calls.jsonl`` (one row per call: identifier, length, timings, bytes;
never any text from the data), ``summary.json`` and ``summary.md``.

Targets checked by the script: median latency per call below 1 s, and peak memory footprint
below 3 GB over the inputs. With ``--add-max-length-input`` the run ends with one input of
``--max-length`` tokens; its latency and the peak footprint including it are reported without
any threshold.

Run one measurement at a time, from a cold start: no measurement and no other heavy load in
the 15 minutes before. Latency on a fanless machine depends on its thermal state, so the
summary records when the run started and ended and how long ago the previous run of the same
folder ended. It also gives the latency by fifth of the run on short inputs only: the inputs
are not in random order, so fifths are compared at a comparable length.

``--rebuild-report <run folder>`` rewrites the two summaries of an existing run from its
``calls.jsonl``, without running anything.

Examples, each on one line, from the folder that holds ``ragtruth-out`` and ``bench-1.13``:

    python-mlx cost/measure_cost.py --judge mlx:PleIAs/Pleias-RAG-350M --max-length 4096 --add-max-length-input --data ragtruth-out/test.jsonl --ids-from bench-1.13/quota/hhem-mps/attempts.jsonl --out cost/runs/pleias-rag-350m-mlx-739

    python cost/measure_cost.py --judge hhem --device mps --data ragtruth-out/test.jsonl --ids-from bench-1.13/quota/hhem-mps/attempts.jsonl --compare-scores bench-1.13/quota/hhem-mps/attempts.jsonl --out cost/runs/hhem-mps-739
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import datetime as dt
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

GB = 1_000_000_000
MIB = 1 << 20
TOKEN_BINS = ((0, 512), (513, 1024), (1025, 2048), (2049, 4096), (4097, 10**9))
SHORT_INPUT_TOKENS = 512

# Provisional prompt templates. The template of each base is declared with the base; these
# only give realistic lengths. Cost depends on the number of tokens, not on the wording.
SYSTEM_PROMPT = (
    "You are a faithfulness judge. Decide whether the sentence is supported by the context. Answer yes or no."
)
USER_PROMPT = "Context:\n{context}\n\nSentence:\n{sentence}"
PLAIN_PROMPT = (
    "Context:\n{context}\n\nSentence:\n{sentence}\n\n"
    "Is the sentence supported by the context? Answer yes or no.\nAnswer:"
)
PLEIAS_PROMPT = (
    "<|query_start|>Is the following sentence supported by the source? {sentence}<|query_end|>\n"
    "<|source_start|><|source_id|>1 {context}<|source_end|>\n<|source_analysis_start|>\n"
)


# =============================================================================
# Process memory
# =============================================================================


def footprint() -> tuple[int | None, int | None]:
    """(current, lifetime maximum) physical memory footprint of this process, in bytes.

    macOS only. ``rusage_info_v4`` starts with a 16-byte uuid followed by 8-byte fields:
    ``ri_phys_footprint`` is the 8th and ``ri_lifetime_max_phys_footprint`` the 29th, the
    figure ``/usr/bin/time -l`` prints as "peak memory footprint". It includes GPU memory.
    """
    if sys.platform != "darwin":
        return None, None
    buf = (ctypes.c_uint64 * 64)()
    if _libproc().proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf)) != 0:
        return None, None
    return int(buf[2 + 7]), int(buf[2 + 28])


_LIBPROC: Any = None


def _libproc() -> Any:
    global _LIBPROC
    if _LIBPROC is None:
        _LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib")
    return _LIBPROC


def max_resident() -> int:
    """Maximum resident set size in bytes (GPU memory not included)."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value) if sys.platform == "darwin" else int(value) * 1024


# =============================================================================
# Data
# =============================================================================


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def attempts_scores(path: Path) -> tuple[list[str], dict[str, float]]:
    """Identifiers in file order and scores of a judge-bench ``attempts.jsonl``."""
    rows = [row for row in read_jsonl(path) if "sample_id" in row]
    scores = {row["sample_id"]: float(row["result"]["score"]) for row in rows if row.get("result")}
    return [row["sample_id"] for row in rows], scores


def select_records(data: Path, ids_from: Path | None, limit: int | None) -> list[dict[str, Any]]:
    records = read_jsonl(data)
    if ids_from is not None:
        by_id = {record["id"]: record for record in records}
        order, _ = attempts_scores(ids_from)
        missing = [i for i in order if i not in by_id]
        if missing:
            msg = f"{len(missing)} identifiers of {ids_from} are not in {data}, first: {missing[0]}"
            raise SystemExit(msg)
        records = [by_id[i] for i in order]
    return records[:limit] if limit else records


# =============================================================================
# Prompt building for candidate bases
# =============================================================================


class PromptBuilder:
    """Renders context and sentence with a template and applies the maximum input length.

    Beyond the maximum length the sentence is kept whole and the context is truncated at
    its end. No windowing.
    """

    def __init__(self, tokenizer: Any, template: str, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        if template == "auto":
            if "<|source_start|>" in tokenizer.get_vocab():
                template = "pleias"
            elif getattr(tokenizer, "chat_template", None):
                template = "chat"
            else:
                template = "plain"
        if template not in ("chat", "pleias", "plain"):
            msg = f"unknown template {template!r}"
            raise SystemExit(msg)
        self.template = template

    def render(self, context: str, sentence: str) -> str:
        if self.template == "pleias":
            return PLEIAS_PROMPT.format(context=context, sentence=sentence)
        if self.template == "plain":
            return PLAIN_PROMPT.format(context=context, sentence=sentence)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(context=context, sentence=sentence)},
        ]
        return str(
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        )

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def build(self, context: str, sentence: str) -> tuple[list[int], bool]:
        """Token ids of the input and whether the context was truncated."""
        ids = self.encode(self.render(context, sentence))
        if len(ids) <= self.max_length:
            return ids, False
        context_ids = self.encode(context)
        keep = len(context_ids) - (len(ids) - self.max_length)
        while keep > 0:
            shorter = self.tokenizer.decode(
                context_ids[:keep], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            ids = self.encode(self.render(shorter, sentence))
            if len(ids) <= self.max_length:
                return ids, True
            keep -= max(1, len(ids) - self.max_length)
        msg = f"template and sentence alone exceed the maximum length of {self.max_length} tokens"
        raise ValueError(msg)

    def label_ids(self) -> list[int]:
        return [self.encode("yes")[0], self.encode("no")[0]]

    def overhead(self) -> int:
        return len(self.encode(self.render("", "")))


def max_length_record(records: list[dict[str, Any]], builder: PromptBuilder) -> dict[str, Any]:
    """A record whose input fills the maximum length: contexts of the data, longest first, joined.

    The contexts are cycled through as many times as needed; ``PromptBuilder.build`` then
    truncates the joined context so that the input fits the maximum length.
    """
    contexts = sorted({record["context"] for record in records}, key=len, reverse=True)
    sentence = max(records, key=lambda record: len(record["context"]))["sentence"]
    joined = ""
    for step in range(10_000):
        context = contexts[step % len(contexts)]
        joined = f"{joined}\n\n{context}" if joined else context
        if len(builder.encode(builder.render(joined, sentence))) > builder.max_length:
            break
    return {"id": "synthetic-max-length", "context": joined, "sentence": sentence, "question": ""}


def long_record(records: list[dict[str, Any]], rows: list[dict[str, Any]], target_tokens: int) -> dict[str, Any]:
    """For judges without a prompt builder: a record of about *target_tokens* tokens.

    The length is set from the tokens-per-character ratio observed on the calls already
    made, so it is approximate; the realised token count is reported with the result.
    """
    done = [(record, row) for record, row in zip(records, rows, strict=False) if row["error"] is None]
    characters = sum(len(record["context"]) + len(record["sentence"]) for record, _ in done)
    ratio = sum(row["tokens"] for _, row in done) / max(1, characters)
    contexts = sorted({record["context"] for record in records}, key=len, reverse=True)
    sentence = max(records, key=lambda record: len(record["context"]))["sentence"]
    wanted = int(target_tokens / ratio) - len(sentence)
    joined = ""
    for step in range(10_000):
        if len(joined) >= wanted:
            break
        context = contexts[step % len(contexts)]
        joined = f"{joined}\n\n{context}" if joined else context
    return {"id": "synthetic-long", "context": joined[:wanted], "sentence": sentence, "question": ""}


# =============================================================================
# Judges
# =============================================================================


class FakeJudge:
    """No model. Lengths are whitespace words."""

    def describe(self) -> dict[str, Any]:
        return {"stack": "none", "model": "fake"}

    def call(self, record: dict[str, Any]) -> tuple[float, int, bool]:
        return 0.5, len(f"{record['context']} {record['sentence']}".split()), False

    def release(self) -> None:
        return None

    def memory(self) -> dict[str, int]:
        return {}


class _TorchDevice:
    def __init__(self, device: str) -> None:
        import torch

        self.torch = torch
        self.device = device
        self.kind = device.split(":", 1)[0]
        if self.kind == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("device 'mps' requested but Metal is not available")

    def release(self) -> None:
        if self.kind == "mps":
            self.torch.mps.empty_cache()
        elif self.kind == "cuda":
            self.torch.cuda.empty_cache()

    def memory(self) -> dict[str, int]:
        if self.kind == "mps":
            mps = self.torch.mps
            return {"allocated": int(mps.current_allocated_memory()), "driver": int(mps.driver_allocated_memory())}
        return {}

    def versions(self) -> dict[str, str]:
        import transformers

        return {"torch": self.torch.__version__, "transformers": transformers.__version__}


class HarnessJudge(_TorchDevice):
    """HHEM or MiniCheck from ragnarok-ai, built as ``ragnarok judge-bench`` builds them."""

    def __init__(self, kind: str, device: str) -> None:
        super().__init__(device)
        import ragnarok_ai

        if kind == "hhem":
            from ragnarok_ai.judges.hhem import HHEMJudge

            self.judge: Any = HHEMJudge(unit="answer", device=device)
        else:
            from ragnarok_ai.judges.minicheck import DEFAULT_CHECKPOINT, MiniCheckJudge

            self.judge = MiniCheckJudge(checkpoint=DEFAULT_CHECKPOINT, unit="answer", device=device)
        self.name = kind
        self.version = ragnarok_ai.__version__
        self.loop = asyncio.new_event_loop()

    def describe(self) -> dict[str, Any]:
        return {
            "stack": "torch",
            "model": self.name,
            "device": self.device,
            "dtype": "float32",
            "ragnarok_ai": self.version,
            **self.versions(),
        }

    def call(self, record: dict[str, Any]) -> tuple[float, int, bool]:
        result = self.loop.run_until_complete(
            self.judge.judge(
                "faithfulness",
                question=record.get("question") or "",
                context=record["context"],
                answer=record["sentence"],
            )
        )
        return float(result.score), int(result.trace.prompt_tokens), bool(result.trace.context_truncated)


class TorchBase(_TorchDevice):
    """A causal language model with transformers: one forward pass, score read at the last position."""

    def __init__(
        self, repo: str, revision: str | None, device: str, dtype: str, template: str, max_length: int
    ) -> None:
        super().__init__(device)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch = self.torch
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(repo, revision=revision, dtype=getattr(torch, dtype))
        self.parameters = sum(parameter.numel() for parameter in model.parameters())
        model = model.to(device).eval()
        self.body = model.base_model
        self.head = model.get_output_embeddings()
        self.builder = PromptBuilder(tokenizer, template, max_length)
        self.labels = self.builder.label_ids()
        self.repo, self.revision, self.dtype = repo, revision, dtype
        self.model_type = model.config.model_type
        self.max_positions = getattr(model.config, "max_position_embeddings", None)

    def describe(self) -> dict[str, Any]:
        return {
            "stack": "torch",
            "model": self.repo,
            "revision": self.revision,
            "model_type": self.model_type,
            "device": self.device,
            "dtype": self.dtype,
            "parameters": self.parameters,
            "max_position_embeddings": self.max_positions,
            "max_length": self.builder.max_length,
            "template": self.builder.template,
            "template_tokens": self.builder.overhead(),
            **self.versions(),
        }

    def call(self, record: dict[str, Any]) -> tuple[float, int, bool]:
        torch = self.torch
        ids, truncated = self.builder.build(record["context"], record["sentence"])
        inputs = torch.tensor([ids], device=self.device)
        with torch.inference_mode():
            hidden = self.body(input_ids=inputs, use_cache=False).last_hidden_state[:, -1, :]
            logits = self.head(hidden)[0, self.labels].float()
            score = float(torch.softmax(logits, dim=-1)[0].item())
        return score, len(ids), truncated


class MlxBase:
    """The same forward pass with mlx-lm, in bfloat16.

    Weights published in another precision are converted to bfloat16 after loading, and the
    conversion is recorded in the description of the judge.
    """

    def __init__(self, repo: str, revision: str | None, template: str, max_length: int) -> None:
        import mlx.core as mx
        import mlx_lm
        from mlx.utils import tree_flatten
        from transformers import AutoTokenizer

        self.mx = mx
        kwargs = {"revision": revision} if revision else {}
        self.model, _ = mlx_lm.load(repo, **kwargs)
        self.model.eval()
        tokenizer = AutoTokenizer.from_pretrained(repo, **kwargs)
        leaves = [value for _, value in tree_flatten(self.model.parameters())]
        self.parameters = int(sum(value.size for value in leaves))
        self.published = sorted({str(value.dtype) for value in leaves})
        self.converted = any(mx.issubdtype(value.dtype, mx.floating) and value.dtype != mx.bfloat16 for value in leaves)
        if self.converted:
            self.model.set_dtype(mx.bfloat16)
            mx.eval(self.model.parameters())
        self.dtypes = sorted({str(value.dtype) for _, value in tree_flatten(self.model.parameters())})
        self.builder = PromptBuilder(tokenizer, template, max_length)
        self.labels = mx.array(self.builder.label_ids())
        self.repo, self.revision = repo, revision
        self.versions = {"mlx": getattr(mx, "__version__", "?"), "mlx_lm": getattr(mlx_lm, "__version__", "?")}
        body = self.model.model
        self.project = self.model.lm_head if hasattr(self.model, "lm_head") else body.embed_tokens.as_linear
        self.body = body

    def describe(self) -> dict[str, Any]:
        return {
            "stack": "mlx",
            "model": self.repo,
            "revision": self.revision,
            "device": "gpu",
            "dtype": ", ".join(self.dtypes),
            "published_dtype": ", ".join(self.published),
            "converted_to_bfloat16": self.converted,
            "parameters": self.parameters,
            "max_length": self.builder.max_length,
            "template": self.builder.template,
            "template_tokens": self.builder.overhead(),
            **self.versions,
        }

    def call(self, record: dict[str, Any]) -> tuple[float, int, bool]:
        mx = self.mx
        ids, truncated = self.builder.build(record["context"], record["sentence"])
        hidden = self.body(mx.array([ids]))[:, -1, :]
        logits = self.project(hidden)[0][self.labels].astype(mx.float32)
        score = float(mx.softmax(logits, axis=-1)[0].item())
        return score, len(ids), truncated

    def release(self) -> None:
        self.mx.clear_cache()

    def memory(self) -> dict[str, int]:
        mx = self.mx
        return {
            "allocated": int(mx.get_active_memory()),
            "cache": int(mx.get_cache_memory()),
            "mlx_peak": int(mx.get_peak_memory()),
        }


def build_judge(args: argparse.Namespace) -> Any:
    kind, _, rest = args.judge.partition(":")
    if kind == "fake":
        return FakeJudge()
    if kind in ("hhem", "minicheck") and not rest:
        return HarnessJudge(kind, args.device)
    if kind == "hf" and rest:
        return TorchBase(rest, args.revision, args.device, args.dtype, args.template, args.max_length)
    if kind == "mlx" and rest:
        return MlxBase(rest, args.revision, args.template, args.max_length)
    msg = f"--judge must be hhem, minicheck, hf:<repo>, mlx:<repo> or fake, got {args.judge!r}"
    raise SystemExit(msg)


# =============================================================================
# Statistics and report
# =============================================================================


def percentile(values: list[float], q: float) -> float:
    """Linear interpolation between order statistics (the default of numpy.percentile)."""
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q / 100.0
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def latency_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "p50_ms": round(percentile(values, 50), 2),
        "p90_ms": round(percentile(values, 90), 2),
        "mean_ms": round(sum(values) / len(values), 2),
        "max_ms": round(max(values), 2),
    }


def fifths_on_short_inputs(attempted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Latency by fifth of the run, on the inputs of at most ``SHORT_INPUT_TOKENS`` tokens.

    The inputs are not in random order: their length can change a lot along the run, so a
    fifth is compared with another on short inputs only. The latency per 100 tokens removes
    what is left of the difference in length. A rise along the run points to thermal drift.
    """
    result = []
    for k in range(5):
        part = attempted[k * len(attempted) // 5 : (k + 1) * len(attempted) // 5]
        short = [row for row in part if row.get("error") is None and row["tokens"] <= SHORT_INPUT_TOKENS]
        entry: dict[str, Any] = {"fifth": k + 1, "n": len(short)}
        if short:
            values = [row["latency_ms"] + row["release_ms"] for row in short]
            per_100 = [100 * value / row["tokens"] for value, row in zip(values, short, strict=True)]
            entry["p50_ms"] = round(percentile(values, 50), 2)
            entry["mean_tokens"] = round(sum(row["tokens"] for row in short) / len(short))
            entry["p50_ms_per_100_tokens"] = round(percentile(per_100, 50), 2)
        result.append(entry)
    return result


def rebuild_report(run_dir: Path) -> int:
    """Recompute the latency-by-fifth table of an existing run from its calls and rewrite its two summaries."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    attempted = [row for row in read_jsonl(run_dir / "calls.jsonl") if not str(row["id"]).startswith("synthetic-")]
    if len(attempted) != summary["n_calls"] + summary.get("n_failed", 0):
        raise SystemExit(f"{run_dir}: calls.jsonl does not match summary.json")
    summary.setdefault("n_failed", 0)
    summary.setdefault("reported_without_threshold", [])
    summary["latency"].pop("by_fifth", None)
    summary["latency"]["short_input_tokens"] = SHORT_INPUT_TOKENS
    summary["latency"]["by_fifth_short_inputs"] = fifths_on_short_inputs(attempted)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    write_markdown(summary, run_dir / "summary.md")
    print(f"rebuilt: {run_dir / 'summary.md'}")
    return 0


def thermal_note() -> list[str]:
    if sys.platform != "darwin":
        return []
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=10, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def machine() -> dict[str, Any]:
    info: dict[str, Any] = {"platform": platform.platform(), "python": platform.python_version()}
    if sys.platform == "darwin":
        for key, name in (("model", "hw.model"), ("memory_bytes", "hw.memsize"), ("chip", "machdep.cpu.brand_string")):
            try:
                out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=10, check=False)
                info[key] = out.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                continue
    return info


def previous_run_end(out_dir: Path) -> str | None:
    """Latest ``ended_at`` among the other runs of the same parent folder."""
    ends = []
    for path in out_dir.parent.glob("*/summary.json"):
        if path.parent == out_dir:
            continue
        try:
            ends.append(json.loads(path.read_text(encoding="utf-8"))["ended_at"])
        except (OSError, ValueError, KeyError):
            continue
    return max(ends) if ends else None


def gb(value: int | None) -> str:
    return "n/a" if value is None else f"{value / GB:.2f} GB ({value / MIB:.0f} MiB)"


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    judge = summary["judge"]
    main = summary["latency"]["call_plus_release"]
    regime = "released after each call" if summary["release_cache"] else "kept (harness behaviour)"
    tokens = summary["tokens"]
    lines = [
        f"# Cost run: {summary['judge_spec']}",
        "",
        f"Regime: one sentence per call, device cache {regime}.",
        f"Judge: {', '.join(f'{key} {value}' for key, value in judge.items())}.",
        f"Inputs: {summary['n_calls']} from `{summary['data']}`"
        + (f", order of `{summary['ids_from']}`" if summary["ids_from"] else "")
        + f"; lengths {tokens['min']} to {tokens['max']} tokens, {tokens['truncated']} truncated.",
        f"Started {summary['started_at']}, ended {summary['ended_at']}, {summary['duration_s']} s."
        + (
            f" Previous run of this folder ended {summary['minutes_since_previous_run']} min before the start."
            if summary["minutes_since_previous_run"] is not None
            else " No previous run in this folder."
        ),
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Median latency, call and cache release | {main['p50_ms']} ms |",
        f"| p90 latency | {main['p90_ms']} ms |",
        f"| Mean latency | {main['mean_ms']} ms |",
        f"| Median latency, call alone | {summary['latency']['call']['p50_ms']} ms |",
        f"| Footprint after loading | {gb(summary['memory']['footprint_after_load'])} |",
        f"| Peak footprint over the inputs | {gb(summary['memory']['peak_footprint_inputs'])} |",
    ]
    if summary["n_failed"]:
        lines.insert(5, f"Failed calls, left out of the figures below: {summary['n_failed']}.")
    extra = summary.get("max_length_input")
    if extra:
        lines += [
            f"| {'Maximum-length' if extra['exact'] else 'Long'} input ({extra['tokens']} tokens): latency "
            f"| {extra['latency_ms']} ms |",
            f"| Peak footprint including that input | {gb(summary['memory']['peak_footprint_total'])} |",
        ]
    if summary.get("max_length_input_error"):
        lines.append(f"| Maximum-length input | failed: {summary['max_length_input_error']} |")
    lines += [f"| Maximum resident memory (no GPU memory) | {gb(summary['memory']['max_resident'])} |", ""]
    short = summary["latency"]["short_input_tokens"]
    lines += [
        f"Latency by fifth of the run, on the inputs of at most {short} tokens only (call and release). The",
        "inputs are not in random order and their length changes along the run, so only short inputs are",
        "compared; the last column divides each latency by the length of its input. A rise from the first",
        "fifth to the last points to thermal drift.",
        "",
        "| Fifth | n | p50 ms | mean tokens | p50 ms per 100 tokens |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {row['fifth']} | {row['n']} | {row['p50_ms']} | {row['mean_tokens']} | {row['p50_ms_per_100_tokens']} |"
        if row["n"]
        else f"| {row['fifth']} | 0 | n/a | n/a | n/a |"
        for row in summary["latency"]["by_fifth_short_inputs"]
    ]
    lines += ["", "Latency by input length:", "", "| Tokens | n | p50 ms | p90 ms |", "|---|---|---|---|"]
    lines += [
        f"| {row['tokens']} | {row['n']} | {row['p50_ms']} | {row['p90_ms']} |"
        for row in summary["latency"]["by_tokens"]
    ]
    lines += ["", "Targets:", ""]
    lines += [
        f"- {target['name']}: {target['value']} against {target['limit']}, {target['verdict']}"
        for target in summary["targets"]
    ]
    if summary["reported_without_threshold"]:
        lines += ["", "Reported without threshold:", ""]
        lines += [f"- {item['name']}: {item['value']}" for item in summary["reported_without_threshold"]]
    if summary.get("score_check"):
        check = summary["score_check"]
        lines += [
            "",
            f"Scores against `{check['reference']}`: maximum absolute difference "
            f"{check['max_abs_diff']:.1e} over {check['n']} inputs.",
        ]
    if summary.get("thermal_before") or summary.get("thermal_after"):
        lines += [
            "",
            "`pmset -g therm` before: " + " / ".join(summary["thermal_before"]) + ".",
            "After: " + " / ".join(summary["thermal_after"]) + ".",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# Run
# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--judge", help="hhem, minicheck, hf:<repo>, mlx:<repo> or fake")
    parser.add_argument("--data", type=Path, help="JSONL with id, context, sentence")
    parser.add_argument("--ids-from", type=Path, help="judge-bench attempts.jsonl giving the inputs and their order")
    parser.add_argument("--out", type=Path, help="output folder, must not hold a summary yet")
    parser.add_argument(
        "--rebuild-report",
        type=Path,
        help="existing run folder: rewrite its summaries from its calls, without running anything",
    )
    parser.add_argument("--device", default="mps", help="torch device for hhem, minicheck and hf (default mps)")
    parser.add_argument("--dtype", default="float16", help="torch dtype for hf (default float16)")
    parser.add_argument("--revision", help="model revision for hf and mlx")
    parser.add_argument("--template", default="auto", help="auto, chat, pleias or plain (hf and mlx)")
    parser.add_argument("--max-length", type=int, default=4096, help="maximum input length in tokens (hf and mlx)")
    parser.add_argument(
        "--add-max-length-input",
        action="store_true",
        help="end with one input of --max-length tokens (exact for hf and mlx, approximate for hhem and minicheck)",
    )
    parser.add_argument(
        "--no-release", action="store_true", help="keep the device cache between calls (harness behaviour)"
    )
    parser.add_argument(
        "--compare-scores", type=Path, help="attempts.jsonl whose scores must be reproduced (hhem, minicheck)"
    )
    parser.add_argument("--limit", type=int, help="only the first N inputs (tests)")
    parser.add_argument("--latency-target-s", type=float, default=1.0)
    parser.add_argument("--memory-target-gb", type=float, default=3.0)
    args = parser.parse_args(argv)
    if args.rebuild_report is None and not (args.judge and args.data and args.out):
        parser.error("--judge, --data and --out are required for a measurement")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rebuild_report is not None:
        return rebuild_report(args.rebuild_report)
    out: Path = args.out
    if (out / "summary.json").exists():
        raise SystemExit(f"{out} already holds a run; choose a new folder")
    records = select_records(args.data, args.ids_from, args.limit)
    if not records:
        raise SystemExit("no input selected")
    reference = attempts_scores(args.compare_scores)[1] if args.compare_scores else None
    out.mkdir(parents=True, exist_ok=True)
    previous_end = previous_run_end(out)
    thermal_before = thermal_note()
    started = dt.datetime.now(dt.timezone.utc)
    clock = time.perf_counter()

    judge = build_judge(args)
    release = not args.no_release
    if release:
        judge.release()
    footprint_after_load = footprint()[0]
    load_seconds = time.perf_counter() - clock

    builder = getattr(judge, "builder", None)
    # A candidate base is not trained: its scores mean nothing, and the inputs may come from a
    # test set, so they are not written to the run files.
    keep_scores = builder is None
    rows: list[dict[str, Any]] = []
    peak_inputs: int | None = None
    extra_record = None
    with (out / "calls.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(len(records) + (1 if args.add_max_length_input else 0)):
            if index < len(records):
                record = records[index]
            elif builder is not None:
                record = extra_record = max_length_record(records, builder)
            else:
                record = extra_record = long_record(records, rows, args.max_length)
            t0 = time.perf_counter()
            error = None
            try:
                score, tokens, truncated = judge.call(record)
            except RuntimeError as exc:  # typically the device running out of memory on this input
                score, tokens, truncated = None, None, False
                error = (str(exc).splitlines() or [type(exc).__name__])[0][:300]
            t1 = time.perf_counter()
            before_release = judge.memory()
            if release:
                judge.release()
            t2 = time.perf_counter()
            current, peak = footprint()
            row = {
                "index": index,
                "id": record["id"],
                "tokens": tokens,
                "truncated": truncated,
                "score": score if keep_scores else None,
                "error": error,
                "latency_ms": round((t1 - t0) * 1000, 3),
                "release_ms": round((t2 - t1) * 1000, 3),
                "footprint": current,
                "footprint_max": peak,
                **{f"{key}_before_release": value for key, value in before_release.items()},
                **{f"{key}_after_release": value for key, value in judge.memory().items()},
            }
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            rows.append(row)
            if index == len(records) - 1:
                peak_inputs = peak
    ended = dt.datetime.now(dt.timezone.utc)

    attempted = rows[: len(records)]
    main_rows = [row for row in attempted if row["error"] is None]
    failures = len(attempted) - len(main_rows)
    if not main_rows:
        raise SystemExit(f"every call failed, first error: {attempted[0]['error']}")
    total = [row["latency_ms"] + row["release_ms"] for row in main_rows]
    fifths = fifths_on_short_inputs(attempted)
    by_tokens = []
    for low, high in TOKEN_BINS:
        values = [row["latency_ms"] + row["release_ms"] for row in main_rows if low <= row["tokens"] <= high]
        if values:
            label = f"{low} to {high}" if high < 10**9 else f"{low} and more"
            by_tokens.append(
                {
                    "tokens": label,
                    "n": len(values),
                    "p50_ms": round(percentile(values, 50), 2),
                    "p90_ms": round(percentile(values, 90), 2),
                }
            )

    peak_total = footprint()[1]
    median_s = percentile(total, 50) / 1000

    def memory_target(name: str, value: int | None) -> dict[str, str]:
        verdict = "n/a" if value is None else ("met" if value < args.memory_target_gb * GB else "MISSED")
        return {"name": name, "value": gb(value), "limit": f"< {args.memory_target_gb} GB", "verdict": verdict}

    targets = [
        {
            "name": "median latency per call",
            "value": f"{median_s:.3f} s",
            "limit": f"< {args.latency_target_s} s",
            "verdict": "met" if median_s < args.latency_target_s else "MISSED",
        },
        memory_target(f"peak memory footprint over the {len(attempted)} inputs", peak_inputs),
    ]
    extra_row = rows[-1] if extra_record else None
    reported: list[dict[str, str]] = []
    if extra_row and extra_row["error"] is None:
        kind = "maximum-length" if builder is not None else "long"
        reported.append(
            {
                "name": f"{kind} input of {extra_row['tokens']} tokens, latency",
                "value": f"{(extra_row['latency_ms'] + extra_row['release_ms']) / 1000:.3f} s",
            }
        )
        reported.append({"name": "peak memory footprint including that input", "value": gb(peak_total)})
    elif extra_row:
        reported.append({"name": "maximum-length input", "value": f"failed: {extra_row['error']}"})
    if failures:
        targets.append(
            {
                "name": "calls without error",
                "value": f"{len(main_rows)} of {len(attempted)}",
                "limit": "all",
                "verdict": "MISSED",
            }
        )
    summary: dict[str, Any] = {
        "judge_spec": args.judge,
        "judge": judge.describe(),
        "release_cache": release,
        "data": str(args.data),
        "data_sha256": sha256_of(args.data),
        "ids_from": str(args.ids_from) if args.ids_from else None,
        "n_calls": len(main_rows),
        "n_failed": failures,
        "tokens": {
            "min": min(row["tokens"] for row in main_rows),
            "max": max(row["tokens"] for row in main_rows),
            "truncated": sum(1 for row in main_rows if row["truncated"]),
        },
        "started_at": started.isoformat(timespec="seconds"),
        "ended_at": ended.isoformat(timespec="seconds"),
        "duration_s": round((ended - started).total_seconds(), 1),
        "load_s": round(load_seconds, 1),
        "previous_run_ended_at": previous_end,
        "minutes_since_previous_run": (
            round((started - dt.datetime.fromisoformat(previous_end)).total_seconds() / 60, 1) if previous_end else None
        ),
        "latency": {
            "call_plus_release": latency_stats(total),
            "call": latency_stats([row["latency_ms"] for row in main_rows]),
            "release": latency_stats([row["release_ms"] for row in main_rows]),
            "short_input_tokens": SHORT_INPUT_TOKENS,
            "by_fifth_short_inputs": fifths,
            "by_tokens": by_tokens,
        },
        "memory": {
            "footprint_after_load": footprint_after_load,
            "peak_footprint_inputs": peak_inputs,
            "peak_footprint_total": peak_total,
            "max_resident": max_resident(),
            "allocator_max_before_release": {
                key.removesuffix("_before_release"): max(row[key] for row in rows)
                for key in rows[0]
                if key.endswith("_before_release")
            },
        },
        "targets": targets,
        "reported_without_threshold": reported,
        "thermal_before": thermal_before,
        "thermal_after": thermal_note(),
        "machine": machine(),
        "environment": {key: value for key, value in os.environ.items() if key.startswith("PYTORCH_MPS")},
    }
    if extra_row and extra_row["error"] is None:
        summary["max_length_input"] = {
            "tokens": extra_row["tokens"],
            "truncated": extra_row["truncated"],
            "exact": builder is not None,
            "latency_ms": round(extra_row["latency_ms"] + extra_row["release_ms"], 2),
        }
    elif extra_row:
        summary["max_length_input_error"] = extra_row["error"]
    if reference is not None:
        diffs = [abs(row["score"] - reference[row["id"]]) for row in main_rows if row["id"] in reference]
        summary["score_check"] = {
            "reference": str(args.compare_scores),
            "n": len(diffs),
            "max_abs_diff": max(diffs) if diffs else float("nan"),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    write_markdown(summary, out / "summary.md")

    print(f"{args.judge}: {len(main_rows)} calls in {summary['duration_s']} s")
    for target in targets:
        print(f"  {target['name']}: {target['value']} against {target['limit']}, {target['verdict']}")
    for item in reported:
        print(f"  {item['name']}: {item['value']} (reported, no threshold)")
    print(f"  report: {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

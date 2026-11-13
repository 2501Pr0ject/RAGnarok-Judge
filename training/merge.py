"""Merging the adapters into the weights of the base: the convenience version (TRAINING.md,
section 9).

The judge that is selected, measured and evaluated is the base in bfloat16 with its separate
adapters. The merged model written here is a convenience for users who want one set of
weights; it is never evaluated, and its drift from the judge is measured here and declared.

    python training/merge.py --base pleias --adapters RUN_DIR/checkpoint-2597 --out MERGED_DIR --check-train TRAIN.jsonl

Each adapted linear module becomes a plain one whose weight is W + scale * (A B)^T, computed
in float32 and converted once to the precision of the base (bfloat16). MERGED_DIR receives a
local model that ``mlx_lm.load`` reads: weights, configuration and tokenizer of the base.

With ``--check-train`` the scores of four forms are compared on the first 50 examples of
epoch 1 of the training file, one sentence per forward pass:

    A  base in bfloat16 with the separate adapters, the form scored on dev during a run
    B  the same with the base weights converted to float32
    C  B after merging, in float32
    D  the merged model in bfloat16, read back from MERGED_DIR, the form of use

C against B checks the merge itself and has to agree within ``EXACT_TOLERANCE``; the command
fails otherwise. D against A is what the rounding of the merged weights to bfloat16 changes:
the two forms cannot be bit-identical, and their gap is printed next to B against A, which is
what bfloat16 arithmetic alone changes with nothing merged. Those two are reported, without
threshold. Training file only: neither the dev set nor the test set is read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

import judge
import sampler

CHECK_EXAMPLES = 50
EXACT_TOLERANCE = 1e-3  # largest absolute difference of P(Supported) accepted between B and C


def merged_linear(module: nn.Module) -> nn.Linear:
    """The plain linear module equal to an adapted one, weight in the precision of the base."""
    linear = module.linear
    weight = linear.weight
    delta = module.scale * (module.lora_b.astype(mx.float32).T @ module.lora_a.astype(mx.float32).T)
    out_dims, in_dims = weight.shape
    merged = nn.Linear(in_dims, out_dims, bias="bias" in linear)
    merged.weight = (weight.astype(mx.float32) + delta).astype(weight.dtype)
    if "bias" in linear:
        merged.bias = linear.bias
    return merged


def merge(model: nn.Module) -> int:
    """Replaces every adapted module by its merged form, in place. Returns their number."""
    merged = [(name, merged_linear(module)) for name, module in model.named_modules()
              if hasattr(module, "lora_a") and hasattr(module, "lora_b")]
    if merged:
        model.update_modules(tree_unflatten(merged))
        mx.eval(model.parameters())
    return len(merged)


def save(model: nn.Module, config: dict[str, Any], tokenizer: Any, folder: Path) -> None:
    """Writes the merged model as a local model: weights, configuration, tokenizer."""
    from mlx_lm.utils import save_config, save_model

    if any("lora" in name for name, _ in tree_flatten(model.parameters())):
        raise SystemExit("the model still carries adapters")
    dtypes = {str(value.dtype) for _, value in tree_flatten(model.parameters())}
    if dtypes != {"mlx.core.bfloat16"}:
        raise SystemExit(f"merged weights in {sorted(dtypes)}, bfloat16 expected")
    folder.mkdir(parents=True)
    save_model(folder, model)
    save_config(dict(config), folder / "config.json")
    tokenizer.save_pretrained(str(folder))


def load_merged(folder: Path) -> tuple[nn.Module, Any]:
    import mlx_lm
    from transformers import AutoTokenizer

    model, _ = mlx_lm.load(str(folder))
    return model, AutoTokenizer.from_pretrained(str(folder))


def to_float32(model: nn.Module) -> None:
    """Converts every weight of the model to float32, in place."""
    model.update(tree_unflatten([(name, value.astype(mx.float32)) for name, value in tree_flatten(model.parameters())]))
    mx.eval(model.parameters())


def gap(first: dict[str, float], second: dict[str, float]) -> dict[str, float]:
    if set(first) != set(second):
        raise SystemExit("the two forms were not scored on the same examples")
    gaps = [abs(first[i] - second[i]) for i in first]
    return {"max": max(gaps), "mean": sum(gaps) / len(gaps)}


def equivalence(
    a: dict[str, float], b: dict[str, float], c: dict[str, float], d: dict[str, float],
    tolerance: float = EXACT_TOLERANCE,
) -> dict[str, Any]:
    """The three comparisons of the check; ``passed`` bears on the merge in float32 only."""
    exact = gap(b, c)
    return {
        "examples": len(a),
        "merge_float32": exact,
        "merged_bfloat16_vs_adapters": gap(a, d),
        "float32_vs_bfloat16_adapters": gap(a, b),
        "tolerance": tolerance,
        "passed": exact["max"] <= tolerance,
    }


def check_records(train: Path, n: int = CHECK_EXAMPLES) -> list[dict[str, Any]]:
    """The first ``n`` examples of epoch 1, in training order."""
    with train.open(encoding="utf-8") as handle:
        records = {record["id"]: record for record in map(json.loads, filter(str.strip, handle))}
    labels = {example_id: record["label"] for example_id, record in records.items()}
    return [records[example_id] for example_id in sampler.epoch_ids(labels, 1)[:n]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", choices=("pleias", "qwen3"), required=True)
    parser.add_argument("--adapters", type=Path, required=True, help="a checkpoint or resume folder of train.py")
    parser.add_argument("--out", type=Path, required=True, help="folder of the merged model, must not exist")
    parser.add_argument("--check-train", type=Path, help="train-rule5-rule1.jsonl: equivalence check on 50 examples")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose another --out")
    state = json.loads((args.adapters / "state.json").read_text(encoding="utf-8"))
    if state["config"]["base"] != args.base:
        raise SystemExit(f"{args.adapters}: trained on {state['config']['base']}, not on {args.base}")

    import mlx_lm
    import prompt_templates as pt

    def adapted() -> tuple[nn.Module, dict[str, Any]]:
        model, _, config = mlx_lm.load(pt.BASES[args.base]["repo"], revision=pt.BASES[args.base]["revision"],
                                       return_config=True)
        judge.add_lora(model, state["config"]["seed"])
        judge.load_adapters(model, args.adapters)
        return model, config

    tokenizer = pt.load_tokenizer(args.base)
    records = check_records(args.check_train) if args.check_train else []
    scores: dict[str, dict[str, float]] = {}
    if records:
        model, _ = adapted()
        scores["a"] = judge.score_records(model, args.base, tokenizer, records)
        to_float32(model)
        scores["b"] = judge.score_records(model, args.base, tokenizer, records)
        merge(model)
        scores["c"] = judge.score_records(model, args.base, tokenizer, records)
        del model
        mx.clear_cache()

    model, config = adapted()
    modules = merge(model)
    save(model, config, tokenizer, args.out)
    print(f"{args.base}: adapters of {args.adapters.name} (update {state['updates_done']}) merged into "
          f"{modules} modules, written to {args.out}", flush=True)
    if not records:
        return 0

    del model
    mx.clear_cache()
    merged_model, merged_tokenizer = load_merged(args.out)
    for record in records:
        if judge.prompt_ids(args.base, merged_tokenizer, record) != judge.prompt_ids(args.base, tokenizer, record):
            raise SystemExit(f"{record['id']}: the saved tokenizer does not give the tokens of the base")
    scores["d"] = judge.score_records(merged_model, args.base, merged_tokenizer, records)
    result = equivalence(scores["a"], scores["b"], scores["c"], scores["d"])

    def line(block: dict[str, float]) -> str:
        return f"largest difference of the score {block['max']:.5f}, mean {block['mean']:.5f}"

    print(f"equivalence on {result['examples']} training examples")
    print(f"  merge in float32 (C against B): {line(result['merge_float32'])}, tolerance {result['tolerance']}: "
          f"{'passed' if result['passed'] else 'FAILED'}")
    print(f"  merged bfloat16 against separate adapters (D against A): {line(result['merged_bfloat16_vs_adapters'])}")
    print(f"  float32 against bfloat16, adapters separate (B against A): "
          f"{line(result['float32_vs_bfloat16_adapters'])}", flush=True)
    return 0 if result["passed"] else 1

if __name__ == "__main__":
    sys.exit(main())

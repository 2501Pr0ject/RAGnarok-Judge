"""The judge as trained and as scored on dev (TRAINING.md, sections 1 and 3).

One forward pass over the prompt; the final hidden state is read at the last position only
and the output head is applied to that single position; the logits of the first tokens of
the three labels are converted to float32. The score is P(Supported) among the three
(TARGETS.md, section 3) and the loss is the cross-entropy of the same three-way softmax.

Applying the head to the last position only is mathematically the same as computing the
logits of every position and reading the last one: the head is a linear map applied to each
position separately. ``cost/measure_cost.py`` reads the score the same way.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import prompt_templates as pt  # noqa: E402

LABELS = ("SUPPORTED", "UNSUPPORTED", "CONTRADICTED")  # as in the data files, in the order of pt.LABELS
TARGET = {label: k for k, label in enumerate(LABELS)}
LORA_RANK = 8
LORA_SCALE = 2.0
SEED = 13


def label_ids(base: str, tokenizer: Any) -> list[int]:
    """First token of each label, in the order Supported, Unsupported, Contradicted."""
    first = pt.label_first_tokens(base, tokenizer)
    ids = [first[label]["first_id"] for label in pt.LABELS]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{base}: the first tokens of the labels are not distinct")
    return ids


def label_logits(model: nn.Module, tokens: mx.array, labels: mx.array) -> mx.array:
    """Logits of the three label tokens at the last position, float32. ``tokens``: (1, length)."""
    hidden = model.model(tokens)[:, -1, :]
    project = model.lm_head if hasattr(model, "lm_head") else model.model.embed_tokens.as_linear
    return project(hidden)[:, labels].astype(mx.float32)


def loss(model: nn.Module, tokens: mx.array, target: mx.array, labels: mx.array) -> mx.array:
    """Cross-entropy of the three-way softmax against the label of the sentence."""
    return nn.losses.cross_entropy(label_logits(model, tokens, labels), target, reduction="mean")


def score(model: nn.Module, tokens: mx.array, labels: mx.array) -> float:
    """P(Supported) among the three labels."""
    return float(mx.softmax(label_logits(model, tokens, labels), axis=-1)[0, 0].item())


def load_base(base: str) -> tuple[nn.Module, Any]:
    """The pretrained base at its pinned revision, weights in bfloat16, and its tokenizer."""
    import mlx_lm
    from mlx.utils import tree_flatten

    model, _ = mlx_lm.load(pt.BASES[base]["repo"], revision=pt.BASES[base]["revision"])
    dtypes = {str(value.dtype) for _, value in tree_flatten(model.parameters())}
    if dtypes != {"mlx.core.bfloat16"}:
        raise SystemExit(f"{base}: weights in {sorted(dtypes)}, bfloat16 expected")
    return model, pt.load_tokenizer(base)


def add_lora(model: nn.Module, seed: int = SEED) -> int:
    """Freezes the base and adds LoRA to every linear module of every block. Returns the number
    of trainable parameters."""
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model.freeze()
    mx.random.seed(seed)
    linear_to_lora_layers(model, len(model.layers), {"rank": LORA_RANK, "scale": LORA_SCALE, "dropout": 0.0})
    return sum(value.size for _, value in tree_flatten(model.trainable_parameters()))


def load_adapters(model: nn.Module, folder: Path) -> int:
    """Loads the adapter weights of a state saved by ``train.py`` into a model carrying LoRA.
    Returns the number of tensors loaded."""
    from mlx.utils import tree_flatten

    weights = list(mx.load(str(folder / "adapters.safetensors")).items())
    expected = {name for name, _ in tree_flatten(model.trainable_parameters())}
    if {name for name, _ in weights} != expected:
        raise SystemExit(f"{folder}: adapter weights do not match the LoRA modules of the model")
    model.load_weights(weights, strict=False)
    mx.eval(model.parameters())
    return len(weights)


def prompt_ids(base: str, tokenizer: Any, record: dict[str, Any]) -> list[int]:
    ids, truncated = pt.build_input(base, tokenizer, record["context"], record["sentence"], pt.MAX_LENGTH)
    if truncated:
        raise SystemExit(f"{record['id']}: input beyond {pt.MAX_LENGTH} tokens, none is expected in RAGTruth")
    return ids


def score_records(
    model: nn.Module, base: str, tokenizer: Any, records: list[dict[str, Any]], progress: Any = None
) -> dict[str, float]:
    """Score of every record, one sentence per forward pass, cache released after each call."""
    labels = mx.array(label_ids(base, tokenizer))
    was_training = model.training
    model.eval()
    scores = {}
    for k, record in enumerate(records, 1):
        scores[record["id"]] = score(model, mx.array([prompt_ids(base, tokenizer, record)]), labels)
        mx.clear_cache()
        if progress is not None:
            progress(k, len(records))
    model.train(was_training)
    return scores

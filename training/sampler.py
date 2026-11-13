"""Deterministic sampler of the training examples (TRAINING.md, section 2).

An epoch holds every UNSUPPORTED and CONTRADICTED sentence of the training file and a fixed
number of SUPPORTED sentences drawn uniformly without replacement, renewed at every epoch.
The draw and the order depend on the seed and on the epoch number only, never on the base,
the learning rate or the progress of a run: a resumed run continues on the examples the
uninterrupted run would have seen. The draw reads the labels and nothing else.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np

SEED = 13
SUPPORTED_PER_EPOCH = 9232


def epoch_ids(
    labels: Mapping[str, str], epoch: int, seed: int = SEED, supported: int = SUPPORTED_PER_EPOCH
) -> list[str]:
    """Ids of the examples of one epoch (numbered from 1), in training order."""
    if epoch < 1:
        raise ValueError("epochs are numbered from 1")
    pool = sorted(i for i, label in labels.items() if label == "SUPPORTED")
    others = sorted(i for i, label in labels.items() if label != "SUPPORTED")
    if supported > len(pool):
        raise ValueError(f"{supported} SUPPORTED sentences asked, {len(pool)} available")
    rng = np.random.default_rng([seed, epoch])
    drawn = np.sort(rng.choice(len(pool), size=supported, replace=False))
    ids = others + [pool[k] for k in drawn]
    return [ids[k] for k in rng.permutation(len(ids))]


def epoch_size(labels: Mapping[str, str], supported: int = SUPPORTED_PER_EPOCH) -> int:
    return sum(label != "SUPPORTED" for label in labels.values()) + supported


def examples_from(
    labels: Mapping[str, str],
    position: int,
    epochs: int,
    seed: int = SEED,
    supported: int = SUPPORTED_PER_EPOCH,
) -> Iterator[tuple[int, str]]:
    """(position, id) from a position counted over the whole run, 0 being its first example."""
    size = epoch_size(labels, supported)
    for epoch in range(position // size + 1, epochs + 1):
        ids = epoch_ids(labels, epoch, seed, supported)
        start = position - (epoch - 1) * size
        for k in range(start, size):
            yield position, ids[k]
            position += 1

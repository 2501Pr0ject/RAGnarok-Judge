# Publication of the retained judge

The retained judge (RJ, tag `selection-v1`), evaluated once in `RESULTS.md` (tag `results-v1`),
is published as Gjallar 0.6B, as section 9 of `TRAINING.md` provides.

## What is published

| | |
|---|---|
| Repository | [`ragnarok-ai/gjallar-0.6b`](https://huggingface.co/ragnarok-ai/gjallar-0.6b) on the Hugging Face Hub |
| Commit | `aaff3979a173853037fd0f5f081d7fc9688b72d1`, 2026-11-16 09:29:57 UTC, "Gjallar 0.6B: retained adapters of RAGnarok Judge (results-v1)" |
| Date of publication | 2026-11-16 |
| `adapters.safetensors` SHA-256 | `44cdeecac3ae75c9bb53d538a14968667f9af2fa730a9dfae9477b6f2b1bed31` (20 227 058 bytes) |
| Base | `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca`, pinned in `adapter_config.json` and `gjallar.py` |
| Licence | Apache License 2.0, the licence of the base |

The SHA-256 of the published adapters is the one recorded at `selection-v1`
(`results/training/README.md`, section "The retained judge", and `ADAPTERS_SHA256` in
`training/rj_judge.py`). It is also the SHA-256 that the Hub reports for the file at the commit
above.

The repository holds the adapters, `adapter_config.json`, `gjallar.py` (the scorer for users,
SHA-256 `b3c501aea896abe68c4ebd9e91d4faced110cecba52e90f34f779a73259dffa1`), `LICENSE` and the
model card. No merged version is published. No data is published with the model, as section 9 of
`TRAINING.md` states.

## Check from the Hub, 2026-11-16

The published repository was downloaded from the Hub, and `gjallar.py` from that download scored
the first 50 examples of epoch 1 of the training file (`train-rule5-rule1.jsonl`, SHA-256
`bac1d36f…955edba`, section 2 of `TARGETS.md`), in training order (`check_records` in
`training/merge.py`). The same 50 examples were scored by `training/rj_judge.py`, unchanged since
`selection-v1`, with the retained adapters (same SHA-256).

- The three probabilities (Supported, Unsupported, Contradicted) are identical bit for bit on
  50 of 50 examples, compared as hexadecimal floats. The length of the prompt in tokens and the
  truncation flag are identical too.
- Only the training file was read. Neither dev nor test was read.

The two check scripts and their outputs were not committed. The check can be repeated with the
published repository, the training file and the retained checkpoint.

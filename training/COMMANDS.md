# Training: the commands

Every command fits on one line, without `\` or `&&`. They are run from the root of the
repository; the data folders they read (`ragtruth-out/`, `bench-1.13/`) and the run folders
they write are local and never committed. `python-mlx` stands for the interpreter of the MLX
environment (`requirements-frozen.txt`; `cost/COMMANDS.md` says how to recreate it). The
models are read from the local cache, hence `HF_HUB_OFFLINE=1`. The test set is never read.

## Tests

The test commands name their paths explicitly, for the local layout where the data folders
and the MLX environment sit in `../ragnarok-judge-ref`, next to the repository; they are run
from the root of the repository, with no link to create.

pytest 9.2.0, with iniconfig 2.3.1 and pluggy 1.7.0, is the test tool of the MLX environment:

```bash
uv pip install --python ../ragnarok-judge-ref/.venv-mlx/bin/python pytest==9.2.0 iniconfig==2.3.1 pluggy==1.7.0
```

All the tests of `training/`:

```bash
RJ_BASES=1 RJ_DEV=../ragnarok-judge-ref/ragtruth-out/dev.jsonl RJ_RUN_HHEM=../ragnarok-judge-ref/bench-1.13/full/hhem RJ_RUN_MINICHECK=../ragnarok-judge-ref/bench-1.13/full/minicheck HF_HUB_OFFLINE=1 ../ragnarok-judge-ref/.venv-mlx/bin/python -m pytest training -q
```

- `RJ_BASES=1` adds the tests that load the two pretrained bases from the local cache.
- `RJ_DEV`, `RJ_RUN_HHEM` and `RJ_RUN_MINICHECK` add the reproduction of the dev criterion of
  the two baselines (sections 12 and 13 of `TARGETS.md`) from the frozen dev file and the dev
  runs of the harness.
- Without these variables the corresponding tests are skipped; the others need no data file
  and no pretrained weights.

## Counts of meta sentences and abstentions per epoch

```bash
python-mlx training/epoch_counts.py --train ragtruth-out/train-rule5-rule1.jsonl
```

## Pilot (section 10 of `TRAINING.md`)

300 updates, no dev set. The same command resumes an interrupted pilot.

```bash
caffeinate -i python-mlx training/train.py --base pleias --lr 1e-4 --train ragtruth-out/train-rule5-rule1.jsonl --out training-runs/pilot --pilot
```

Drift of the merged convenience version, measured on the weights of the pilot before they are
deleted, on 50 training examples:

```bash
python-mlx training/merge.py --base pleias --adapters training-runs/pilot/resume --out training-runs/pilot-merged --check-train ragtruth-out/train-rule5-rule1.jsonl
```

## Runs (section 5)

The same command resumes an interrupted run from its latest state.

```bash
caffeinate -i python-mlx training/train.py --base pleias --lr 1e-4 --train ragtruth-out/train-rule5-rule1.jsonl --dev ragtruth-out/dev.jsonl --out training-runs/pleias-lr1e-4
```

```bash
caffeinate -i python-mlx training/train.py --base pleias --lr 3e-4 --train ragtruth-out/train-rule5-rule1.jsonl --dev ragtruth-out/dev.jsonl --out training-runs/pleias-lr3e-4
```

Run 3, with the learning rate of the winner of runs 1 and 2 in place of `LR`:

```bash
caffeinate -i python-mlx training/train.py --base qwen3 --lr LR --train ragtruth-out/train-rule5-rule1.jsonl --dev ragtruth-out/dev.jsonl --out training-runs/qwen3
```

## Selection between the bases (amendment 1 of `TARGETS.md`)

With the dev scores of the best checkpoint of each base, `CKPT_P` and `CKPT_Q`:

```bash
python-mlx training/dev_eval.py --dev ragtruth-out/dev.jsonl --scores pleias=training-runs/CKPT_P/dev_scores.json qwen3=training-runs/CKPT_Q/dev_scores.json --out training-runs/selection.json
```

## Cost of the retained judge (section 8 of `TARGETS.md`)

The base in bfloat16 with the adapters of the retained checkpoint, not merged; same regime as
for the candidate bases, from a cold start, on mains power. For Pleias-RAG-350M:

```bash
caffeinate -i /usr/bin/time -l python-mlx cost/measure_cost.py --judge mlx:PleIAs/Pleias-RAG-350M --revision db001b29a33532583d14979c3c38ef04b6d59352 --adapter training-runs/CKPT_P --max-length 4096 --add-max-length-input --data ragtruth-out/test.jsonl --ids-from bench-1.13/quota/hhem-mps/attempts.jsonl --out cost/runs/rj-pleias-adapter-mlx-739
```

For Qwen3-0.6B: `--judge mlx:Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca`
and its checkpoint. The run reads the inputs of the 739 sentences to time them and writes no
score.

## Merged convenience version (section 9 of `TRAINING.md`)

Never evaluated. Its drift from the judge with separate adapters is measured on 50 training
examples and declared with it:

```bash
python-mlx training/merge.py --base pleias --adapters training-runs/CKPT_P --out training-runs/rj-merged --check-train ragtruth-out/train-rule5-rule1.jsonl
```

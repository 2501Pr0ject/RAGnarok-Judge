# Cost measurement: the commands

Every command fits on one line, without `\` or `&&`. They are run from the root of the
repository; the data folders they read and write (`ragtruth-out/`, `bench-1.13/`, `cost/runs/`)
are local and never committed. Only the summaries kept in `results/cost/` are published.

Cost is measured on the 739-sentence sample for every judge. Rules: one measurement at a time,
from a cold start, that is no measurement and no other heavy load in the 15 minutes before, on
mains power. Each run writes a new folder under `cost/runs/` and refuses to overwrite an
existing run. The result is read in `cost/runs/<name>/summary.md`, which also gives the time
elapsed since the previous run.

Regime: one sentence per call, device cache released after each call. Targets checked by the
script on the 739 sentences: median latency < 1 s, peak footprint < 3 GB. The 4096-token input
added at the end of a run is reported without threshold.

## Environments

Two Python environments, versions in `requirements-frozen.txt`:

- PyTorch: torch 2.15.0, transformers 5.22.2, ragnarok-ai 1.13.0. Used for the baselines and the
  PyTorch MPS runs.
- MLX: Python 3.12.9, mlx 0.32.4, mlx-lm 0.32.2. Used for the MLX runs of the candidate bases.

The commands below are written for the PyTorch environment being active; `python-mlx` stands
for the interpreter of the MLX environment. To check it:

```bash
python-mlx -c "import mlx.core as mx, mlx_lm; print(mx.__version__, mlx_lm.__version__)"
```

To recreate it with uv, in two commands:

```bash
uv venv .venv-mlx --python 3.12
```

```bash
uv pip install --python .venv-mlx/bin/python mlx==0.32.4 mlx-lm==0.32.2
```

## 0. Download the weights, once, outside any measurement

Both stacks read the same Hugging Face cache.

```bash
hf download PleIAs/Pleias-RAG-350M --revision db001b29a33532583d14979c3c38ef04b6d59352
```

```bash
hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca
```

## The four runs of the candidate bases, in this order

Done on 2026-11-09, chained without a pause: results in section 8 of `TARGETS.md` and in
`results/cost/`. To redo them from a cold start, change the name of the output folder and wait
15 minutes between two runs.

### 1. Pleias-RAG-350M on MLX

```bash
caffeinate -i /usr/bin/time -l python-mlx cost/measure_cost.py --judge mlx:PleIAs/Pleias-RAG-350M --revision db001b29a33532583d14979c3c38ef04b6d59352 --max-length 4096 --add-max-length-input --data ragtruth-out/test.jsonl --ids-from bench-1.13/quota/hhem-mps/attempts.jsonl --out cost/runs/pleias-rag-350m-mlx-739
```

### 2. Qwen3-0.6B on MLX

```bash
caffeinate -i /usr/bin/time -l python-mlx cost/measure_cost.py --judge mlx:Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca --max-length 4096 --add-max-length-input --data ragtruth-out/test.jsonl --ids-from bench-1.13/quota/hhem-mps/attempts.jsonl --out cost/runs/qwen3-0.6b-mlx-739
```

### 3. Pleias-RAG-350M on PyTorch MPS in float16

```bash
caffeinate -i /usr/bin/time -l python cost/measure_cost.py --judge hf:PleIAs/Pleias-RAG-350M --revision db001b29a33532583d14979c3c38ef04b6d59352 --device mps --dtype float16 --max-length 4096 --add-max-length-input --data ragtruth-out/test.jsonl --ids-from bench-1.13/quota/hhem-mps/attempts.jsonl --out cost/runs/pleias-rag-350m-mps-float16-739
```

### 4. Qwen3-0.6B on PyTorch MPS in float16

```bash
caffeinate -i /usr/bin/time -l python cost/measure_cost.py --judge hf:Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca --device mps --dtype float16 --max-length 4096 --add-max-length-input --data ragtruth-out/test.jsonl --ids-from bench-1.13/quota/hhem-mps/attempts.jsonl --out cost/runs/qwen3-0.6b-mps-float16-739
```

## Baselines in the same regime, 739 sentences: the two reference runs

Run from a cold start, HHEM first, then MiniCheck 15 minutes after the end of the first.
`--compare-scores` checks that the scores of the harness are reproduced. Both done on
2026-11-09: results in section 8 of `TARGETS.md` and in `results/cost/`.

```bash
caffeinate -i /usr/bin/time -l python cost/measure_cost.py --judge hhem --device mps --data ragtruth-out/test.jsonl --ids-from bench-1.13/quota/hhem-mps/attempts.jsonl --compare-scores bench-1.13/quota/hhem-mps/attempts.jsonl --out cost/runs/hhem-mps-739
```

```bash
caffeinate -i /usr/bin/time -l python cost/measure_cost.py --judge minicheck --device mps --data ragtruth-out/test.jsonl --ids-from bench-1.13/quota/hhem-mps/attempts.jsonl --compare-scores bench-1.13/quota/minicheck/attempts.jsonl --out cost/runs/minicheck-mps-739
```

## What the script assumes

- Candidate bases: one forward pass per sentence, score read at the last position.
- On MLX the precision is bfloat16 for both bases, read from the official repositories at the
  revisions above. Both are published in bfloat16, so without conversion. Should weights come
  in another precision, the script would convert them to bfloat16 after loading and say so in
  the report, on the "Judge" line: `published_dtype` and `converted_to_bfloat16`. The
  elimination of a base is decided on MLX only.
- On PyTorch MPS the weights are converted to float16 at loading.
- Provisional templates: chat format for Qwen3 (43 tokens), native markers for Pleias (21
  tokens). The cost depends on the number of tokens, not on the text of the template; the
  declared templates of `TEMPLATES.md` are longer by about 41 and 62 tokens.
- Pretrained weights, not trained for the task: the score returned has no meaning and is not
  written to the files of the run.

## What was run

- The four runs of the candidate bases, on 2026-11-09, on MLX and on PyTorch MPS.
- The two baselines through this script, on 2026-11-09, each from a cold start; their scores
  reproduce those of the harness exactly.

# Endless Terminals runtime

This package runs stateful, multi-turn terminal trajectories against local
persistent Apptainer sessions and returns the official binary final-test
reward. The code integration targets the
[official Endless Terminals repository](https://github.com/kanishkg/endless-terminals)
at revision `26ecf78458e7f756e4d06780d5fbf3dd78e91815`.

## Data

Nine tasks requiring live public-network behavior are excluded. The remaining
2,483 tasks use a deterministic seed-42 split:

| Split | Tasks | Use |
|---|---:|---|
| train | 2,083 | policy optimization |
| validation | 100 | in-training diagnostics |
| test | 300 | final held-out evaluation |
| eval | 400 | compatibility alias: validation + test |

Download the public 406 MB source dataset from Hugging Face at the exact
revision used for these experiments:

```bash
python scripts/endless_terminals/download_dataset.py
```

The task sources are stored under `tasks/endless_terminals/data/source/`, and
the exact split IDs are included in `tasks/endless_terminals/splits/`.

## Prepare the runtime

Clone the pinned Endless Terminals runtime code into the standard local path:

```bash
git clone https://github.com/kanishkg/endless-terminals external/endless-terminals
git -C external/endless-terminals checkout 26ecf78458e7f756e4d06780d5fbf3dd78e91815
```

The downloaded tasks contain `container.def` files. Build local SIF images for
both the training and held-out tasks on a Linux host with Apptainer:

```bash
python scripts/endless_terminals/build_sifs.py \
  --source-dir tasks/endless_terminals/data/source \
  --split-file tasks/endless_terminals/splits/train.txt \
  --direct-docker-base \
  --fakeroot \
  --workers 4

python scripts/endless_terminals/build_sifs.py \
  --source-dir tasks/endless_terminals/data/source \
  --split-file tasks/endless_terminals/splits/eval.txt \
  --direct-docker-base \
  --fakeroot \
  --workers 4
```

Build the fixed train, validation, and test Parquet files consumed by training
and evaluation:

```bash
python scripts/endless_terminals/build_skyrl_parquets.py \
  --official-repo external/endless-terminals
```

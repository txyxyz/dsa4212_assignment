# Training a small language model under a fixed compute budget

Start with [REPORT_HANDOFF.md](REPORT_HANDOFF.md) for the report-writing handoff,
verified results, limitations, and outstanding tasks. Group details are in authors.md.

## Current status

Six final GPT models were trained from random weights on TinyStories: three sizes
and two seeds, each with a declared 1800-second training allowance on Modal's A10
GPU option. Full validation and held-out test evaluation are complete.
The submitted report has not been written yet. See report/README.md.

## Files

- code/: dataset preparation, tokenization, training, Modal launcher and evaluation.
- dataset_cleaning.ipynb: exact-overlap removal used before training.
- EXPERIMENT_PROTOCOL.md: final experiment design and budget definition.
- results/: small numerical artifacts, settings, metrics and evaluation summaries.
- results/full-validation/ and results/full-test/: authoritative full evaluation.
- report/: report source will go here; no final report exists yet.

Large data files, model checkpoints and virtual environments are deliberately
excluded. The original checkpoints remain on the experiment owner's Mac and
Modal volume `small-llm-results`; request access/copies before local evaluation.
Do not run training just to write the report: use the saved numerical results.

## Environment

Create a Python 3.13 environment and install `requirements.txt`. This file is an
unpinned convenience list, not a lockfile. The cloud image pins torch 2.14.1,
transformers 5.21.0 and numpy 2.5.4 in code/modal_train.py. Full dependency locks
remain to be recorded. On the original Apple Silicon Mac, commands use
`arch -arm64 .venv-arm/bin/python` because the original notebook launched Intel
Python against ARM packages. Other systems should use their own environment's Python.

## Reproduction workflow

Run from the repository root:

1. `python code/prepare_date.py` downloads/streams the subset. Historical filename
   is retained. This uses buffered shuffle seed 42, not a uniform whole-corpus shuffle.
2. Run dataset_cleaning.ipynb to create data/clean/{train,valid,test}.txt.
3. Obtain/cache the GPT-2 tokenizer (`GPT2TokenizerFast.from_pretrained("gpt2")`),
   then run `python code/tokenize_data.py`. Compare hashes against the saved run
   data_manifest.json files before claiming exact reproduction.
4. Authenticate via `python -m modal setup`. Training uses paid compute/credits.
   Follow EXPERIMENT_PROTOCOL.md for model sizes and seeds. Example:
   `python -m modal run --detach code/modal_train.py --run-id final-small-s42 --seconds 1800 --model-size small --seed 42 --purpose final`.
   Existing run directories cannot be overwritten; use unique IDs for fresh reruns.
5. Download checkpoints into the corresponding results run directories (create
   the local destination directory first). Retain only one completed copy of each
   size/seed in the evaluation search tree.
6. `python code/evaluate_validation.py` evaluates full validation.
7. `python code/evaluate_validation.py --split test` reproduces the fixed test
   evaluation; do not use the observed test scores to tune the models.

Evaluation checks checkpoint steps and data manifests. It requires all six final
checkpoints, their configs/summaries, token files and the saved tokenizer. Report
writing and analysis of the saved numerical files require none of the checkpoints.

## Known reproducibility gaps

The initial dataset revision and complete package environment were not pinned.
Data hashes are preserved but the original text/token artifacts are not in Git.
Archive the exact data/tokenizer under the dataset's terms or identify the source
revision before final submission. The downloader counts source indices rather
than accepted nonempty stories, so other source revisions may yield different counts.
Cloud hardware labels differ between A10 and A10G; report this explicitly.

# Training a small language model under a fixed compute budget

This project investigates how model size affects language-model performance under a fixed training-time allowance. We train three GPT-style models from randomly initialized weights on a subset of TinyStories, using two random seeds per model.

Each final run has a declared allowance of **1,800 seconds on one GPU requested through Modal’s `A10` option**. Recorded device names include NVIDIA A10 and A10G.

Group members are listed in [authors.md](authors.md). The experimental protocol is documented in [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md).
## Repository structure

```text
authors.md                   Group members and student numbers
requirements.txt             Python dependencies
EXPERIMENT_PROTOCOL.md       Experimental design and budget definition
code/
    prepare_date.py          Download the dataset subset
    tokenize_data.py         Tokenize cleaned training and validation data
    train_baseline.py        Model initialization and training
    modal_train.py           Modal GPU training launcher
    evaluate_validation.py   Full validation and test evaluation
dataset_cleaning.ipynb        Inspect data and remove exact cross-split overlaps
results/                     Configurations, training logs and evaluation results
report/                      Report source, bibliography and figures
```

Large datasets, token files, model checkpoints and virtual environments are excluded from Git. Numerical results can be inspected without downloading checkpoints or rerunning training.
  

## Environment and dependencies

The experiments used Python 3.13. The Modal training image specifies:

- PyTorch `2.14.1`
- Transformers `5.21.0`
- NumPy `2.5.4`

Other dependencies are listed in `requirements.txt`, including `datasets`, `tokenizers`, `modal`, `matplotlib`, `jupyterlab` and `ipykernel`.

### Local setup

Run from the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "torch==2.14.1" "transformers==5.21.0" "numpy==2.5.4"

python -m pip check
```

`requirements.txt` is currently an unpinned dependency list, not a complete environment lock. Installing the three pinned packages does not fully reproduce every original dependency version.

Register the environment for notebooks:

```bash
python -m ipykernel install --user \
  --name small-llm \
  --display-name "Python (small-llm)"
```

Select this environment as the notebook kernel.

On Apple Silicon, Python and compiled packages must use the same architecture. The original project used ARM Python explicitly:

```bash
arch -arm64 .venv-arm/bin/python
```

The commands below use `python`, assuming the intended environment is activated. Training runs remotely on Modal; evaluation and plotting can run locally on a CPU.


## Dataset and preparation
We use the official [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories).

TinyStories contains synthetic short English stories. Our conclusions concern language modelling in this domain rather than general-purpose language understanding.

### 1. Download the subset

```bash
python code/prepare_date.py
```

The filename `prepare_date.py` is retained from the original implementation.

The script streams the dataset using:

- Shuffle seed: `42`
- Shuffle buffer size: `10,000`
- Initial training subset: `100,000` stories
- Initial validation subset: `2,000` stories
- Initial test subset: `2,000` stories

Streaming uses a buffered shuffle rather than a uniform shuffle of the entire corpus.

Output files:

```text
data/train.txt
data/valid.txt
data/test.txt
```

Stories are separated by `<|endoftext|>`.

### 2. Inspect and clean the splits

Open `dataset_cleaning.ipynb` and run its cells in order.

The cleaning procedure:

1. Preserves the test split.
2. Removes test stories from validation.
3. Removes validation and test stories from training.

Comparisons use stripped story text and remove exact cross-split overlaps only.

The cleaned dataset contains:

| Split | Stories |
|---|---:|
| Training | 99,868 |
| Validation | 1,994 |
| Test | 2,000 |

Cleaned files are saved in:

```text
data/clean/train.txt
data/clean/valid.txt
data/clean/test.txt
```

Near-duplicate stories and duplicates within individual splits are not necessarily removed.

### 3. Prepare the tokenizer and token files

Download/cache the GPT-2 tokenizer:

```bash
python -c 'from transformers import GPT2TokenizerFast; GPT2TokenizerFast.from_pretrained("gpt2")'
```

Then tokenize the cleaned training and validation splits:

```bash
python code/tokenize_data.py
```

This saves:

```text
data/tokens/train.bin
data/tokens/valid.bin
data/tokens/tokenizer/
data/tokens/manifest.json
```

One end-of-story token is appended to each story. Token files use little-endian unsigned 32-bit integers.

The original preparation produced:

- Training: `22,323,961` tokens
- Validation: `455,842` tokens

The manifest records counts and SHA-256 hashes. Compare these with the `data_manifest.json` files saved alongside the original runs before claiming exact data reproduction.

The test split is tokenized separately when test evaluation is requested.

## Training experiments

We compare three model sizes (`small`, `medium`, and `large`), each trained from scratch with seeds 42 and 43 under a 1,800-second allowance on Modal's A10 GPU option.

See [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) for the model architectures, shared training settings, budget definition, and experimental controls.

### Launch training on Modal

Authenticate:

```bash
python -m modal setup
```

Example: train the small model with seed 42.

```bash
python -m modal run --detach code/modal_train.py \
  --run-id final-small-s42 \
  --seconds 1800 \
  --model-size small \
  --seed 42 \
  --purpose final
```

Run the following six configurations:

| Model size | Seed | Suggested fresh run ID |
|---|---:|---|
| small | 42 | final-small-s42 |
| small | 43 | final-small-s43 |
| medium | 42 | final-medium-s42 |
| medium | 43 | final-medium-s43 |
| large | 42 | final-large-s42 |
| large | 43 | final-large-s43 |

Change `--model-size`, `--seed` and `--run-id` accordingly.

Existing remote output directories cannot be overwritten. Choose a new run ID if a name has already been used.

These runs consume Modal credits. `--detach` allows a run to continue after the launching terminal disconnects.

### Download training artifacts

Results are stored in the Modal volume `small-llm-results`.

Create the local destination directory before downloading:

```bash
mkdir -p results/final-small-s42-files

python -m modal volume get \
  small-llm-results final-small-s42 \
  results/final-small-s42-files/
```

Each completed run should contain:

```text
config.json
settings.json
data_manifest.json
metrics.jsonl
checkpoint.pt
summary.json
```

Check that `summary.json` reports completion before including a run in the final comparison.

The original completed runs used for evaluation are:

| Model | Seed | Directory under `results/` |
|---|---:|---|
| Small | 42 | final-small-s42-files/final-small-s42 |
| Small | 43 | final-small-s43-retry1-files/final-small-s43-retry1 |
| Medium | 42 | final-medium-s42-files/final-medium-s42 |
| Medium | 43 | final-medium-s43-files/final-medium-s43 |
| Large | 42 | final-large-s42-files/final-large-s42 |
| Large | 43 | final-large-s43-files/final-large-s43-retry1 |

Preliminary pilots and interrupted runs are excluded from final averages.

The evaluator expects exactly one completed run per model-size/seed combination. If reproducing training, place the new runs in a separate checkout or move the original completed-run metadata outside that checkout’s `results/` directory.

The original checkpoints are not included in Git. Access to the original Modal workspace or checkpoint copies from the experiment owner is required to reproduce their exact scores. Alternatively, retrain all six configurations; newly trained checkpoints may produce slightly different results.

## Evaluation

### Full validation

```bash
python code/evaluate_validation.py
```

Results are saved in:

```text
results/full-validation/
```

### Final test evaluation

After fixing the models and evaluation procedure:

```bash
python code/evaluate_validation.py --split test
```

Results are saved in:

```text
results/full-test/
```

The test results have already been produced for this project. Rerunning the command is for verification or reproduction, not for tuning models against the test set.

### Scoring procedure

Validation and test evaluation use the same fixed procedure described in [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md).

The evaluator checks data manifests and checkpoint steps and requires all six final checkpoints, their configuration and summary files, and the prepared token data.

Training-time monitoring scores are computed on a small validation subset and differ from the full evaluation results reported here.


## Reproducibility and limitations

- The original dataset revision was not pinned. Saved hashes identify the artifacts used, but a fresh download is not guaranteed to reproduce them.
- The exact original data, tokenizer and checkpoints are excluded from Git. Their availability must be arranged separately for exact score reproduction.
- The complete original dependency environment was not locked.
- The downloader counts source indices rather than accepted nonempty stories; different source content may change the resulting counts.
- GPU device labels include both A10 and A10G, despite all runs requesting Modal’s A10 option.
- Fixed-time stopping and GPU numerical behavior can cause reruns to differ even with the same seed.
- Checkpoints do not include random-generator states, and the training script does not implement resuming interrupted runs.
- Exact duplicate removal does not eliminate near duplicates.
- Conclusions are limited to this synthetic dataset subset, three architectures, two seeds, and the shared training recipe.


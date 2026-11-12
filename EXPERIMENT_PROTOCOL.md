# Experiment Protocol

This document describes the experimental design for comparing three GPT model sizes under a fixed compute budget. It specifies the data preparation, model configurations, training settings, evaluation procedure, and limitations.

For environment setup, data-preparation commands, training commands, and reproduction instructions, see [README.md](README.md).

## 1. Research question

Which of three GPT model sizes achieves the lowest held-out next-token prediction loss under a fixed training-time allowance?

Each final run receives an allowance of **1,800 seconds on one GPU requested through Modal's A10 option**.

Increasing model size provides more capacity but reduces the number of training tokens processed within the same time allowance. This experiment investigates that trade-off on a subset of TinyStories.

## 2. Data and tokenization

We use a subset of the official [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories), which contains synthetic short English stories.

The initial subset was obtained using streaming with shuffle seed 42 and a shuffle buffer of 10,000 stories. This is a buffered shuffle, not a uniform sample of the complete corpus.

Initial split sizes were:

- Training: 100,000 stories
- Validation: 2,000 stories
- Test: 2,000 stories

### Preventing exact cross-split leakage

Stories were compared after stripping leading and trailing whitespace.

The cleaning procedure:

1. Preserved the test split.
2. Removed stories from validation if they appeared in test.
3. Removed stories from training if they appeared in validation or test.

The resulting splits contain:

| Split | Stories |
|---|---:|
| Training | 99,868 |
| Validation | 1,994 |
| Test | 2,000 |

This removes exact cross-split overlaps. It does not guarantee removal of near duplicates or duplicates within a split.

### Tokenization

All models use the same GPT-2 tokenizer. One end-of-story token is appended to each story before the token sequences are concatenated.

The prepared data contain:

- Training: 22,323,961 tokens
- Validation: 455,842 tokens
- Test: 456,877 tokens

Training and validation counts and source hashes are recorded in the saved `data_manifest.json` files. Test preparation is recorded in `results/full-test/test_manifest.json`.

The tokenizer is reused, but all neural weights—including token embeddings—are randomly initialized. No pretrained neural weights are loaded.

## 3. Model configurations

| Size | Layers | Embedding width | Attention heads | Parameters | Seeds |
|---|---:|---:|---:|---:|---|
| Small | 4 | 128 | 4 | 7,242,624 | 42, 43 |
| Medium | 6 | 256 | 4 | 17,637,632 | 42, 43 |
| Large | 8 | 384 | 6 | 33,544,320 | 42, 43 |

Depth and width change together. Therefore, this experiment compares three model configurations rather than isolating the effects of depth or width.

The same two seeds are used for every model size. Seeds control random initialization, training-window sampling, and other random operations such as dropout.

## 4. Shared training settings

| Setting | Value |
|---|---|
| Context length | 128 tokens |
| Batch size | 16 sequences |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Learning-rate schedule | Constant |
| Gradient clipping | Maximum norm 1.0 |
| GPU precision | bfloat16 |
| Monitoring validation interval | Approximately 30 seconds |
| Checkpoint interval | Approximately 300 seconds, plus final save |
| Training allowance | 1,800 seconds per run |

Training samples random fixed-length windows from the concatenated training tokens.

The dataset, tokenizer, context length, batch size, optimizer settings, and time allowance are held fixed across model sizes. Model size and random seed are the planned experimental variables.

Other architecture and optimizer defaults are determined by the saved configuration and training implementation. The Modal image pins PyTorch 2.14.1, Transformers 5.21.0, and NumPy 2.5.4.

The shared training recipe is not separately optimized for each model.

## 5. Compute-budget definition

The allowance begins immediately before initial monitoring validation.

It includes:

- Initial monitoring validation
- Training updates
- Periodic monitoring validation
- Periodic checkpoint saving

It excludes:

- Environment startup and model initialization
- Final evaluation after training stops
- Final checkpoint saving

Excluded operations can still incur cloud charges.

The loop stops with a conservative margin before the deadline. Completed final runs used approximately 1,798 seconds of the declared allowance.

This allowance is not equivalent to 1,800 seconds of optimizer-only updates or exactly 30 minutes of total billed GPU time.

Actual elapsed time, training steps, tokens processed, and peak allocated GPU memory are recorded for each run.

There are six completed final runs, corresponding to three GPU-hours of declared training allowance. Preliminary experiments, interrupted attempts, startup, and final evaluation/saving are additional compute.

## 6. Preliminary and interrupted runs

The five-minute and fifteen-minute small-model pilots were used to check the pipeline and inform the final budget. They are excluded from the final comparison.

Interrupted runs are not treated as completed final runs. They must not silently receive a new allowance as if continuing the same run.

The first final small-model seed-43 attempt was canceled at approximately 1,727 seconds. Its latest saved checkpoint was from approximately 1,518 seconds. A fresh run with the same seed was completed and used in the comparison.

The completed large-model seed-43 run is stored under a `final-large-s43-retry1` run identifier. Its name is preserved for traceability; the name alone does not establish why a retry occurred.

The current training implementation does not support automatic resumption and does not save random-generator states in checkpoints.

## 7. Monitoring and checkpoint selection

During training, validation is measured on 40 fixed windows of 128 tokens each.

These scores are used to inspect learning curves. They are not the full validation or test metrics.

For each completed run, evaluation uses the **final checkpoint**, not the checkpoint with the lowest monitoring loss.

This selection rule is consistent across all six runs.

## 8. Full validation and test evaluation

All models are evaluated using the same tokenizer, evaluation text, windowing procedure, and numerical precision.

Evaluation uses:

- CPU float32 computation
- Evaluation mode, which disables dropout
- Windows of at most 128 tokens
- A stride of 127 tokens
- One score for every next-token target
- Token-weighted aggregation of prediction loss
- End-of-story tokens included in scoring

Windows overlap by one token so each target after the first token of the concatenated sequence is predicted exactly once.

Context resets at each window. The reported metric therefore measures windowed prediction performance rather than performance with unlimited document context.

### Metrics

The primary metric is mean next-token cross-entropy in natural-log units. Lower values indicate better predictive performance.

For each run:

```text
perplexity = exp(mean cross-entropy)
```

Perplexity is computed separately for each run before averaging across seeds. Mean perplexity is therefore not necessarily equal to the exponential of mean loss.

Because every model uses the same tokenizer, per-token loss and perplexity are directly comparable within this experiment.

### Validation and test roles

Validation data were used during development and for comparing completed models.

After model configurations and the scoring procedure were fixed, all six models were evaluated on the held-out test split.

Test scores must not be used to select new hyperparameters or repeatedly modify the models.

Evaluation records include checkpoint hashes and data hashes so that scores can be associated with the exact evaluated artifacts.

## 9. Reporting across seeds

Report individual seed results and the mean and sample standard deviation across seeds 42 and 43.

These standard deviations describe variation across only two runs.
They are not confidence intervals and do not quantify uncertainty across possible datasets.

Alongside predictive performance, report:

- Number of parameters
- Actual elapsed training time
- Tokens processed
- Recorded GPU model
- Peak allocated GPU memory
- Monitoring learning curves

## 10. Results

Full validation and test results are saved in `results/full-validation/` and `results/full-test/`.

Each folder contains individual run scores, a comparison CSV, and a summary of means and standard deviations across seeds. Analysis and interpretation are provided in the report.

## 11. Qualitative story generation

The final checkpoints are compared using three fixed prompts:

1. "Once upon a time, Mia found a small red key."
2. "Tom wanted to help his friend, but"
3. "The little dog was afraid of the rain."

Shared generation settings:

- Sampling enabled
- Temperature: 0.8
- Top-k: 50
- Maximum new tokens: 80
- Generation seeds: 2026, 2027, and 2028 for the respective prompts

Generation seeds are separate from training seeds.

All three prompts are used for all six checkpoints, producing 18 saved continuations. Outputs may stop at an end-of-story token or be truncated at the generation limit.

Samples are saved in:

```text
results/story-examples/samples.json
```

Inspect relevance to the prompt, grammar, character consistency, cause-and-effect relationships, and repetition. Include failures as well as successes.

These prompts were selected after numerical evaluation. The examples are qualitative illustrations, not an independent benchmark or a quantitative measure of story quality.

## 12. Limitations

- TinyStories is a synthetic, narrow-domain dataset, so conclusions may not generalize to broader language tasks.
- Buffered streaming shuffle is not uniform whole-corpus sampling.
- Exact-overlap removal does not eliminate near duplicates.
- Only three model configurations and two seeds were evaluated.
- Depth and width change together, so their effects are not isolated.
- All models share one training recipe without separate hyperparameter optimization.
- The short context and window resets limit evaluation of long-range dependencies.
- Recorded GPU labels include A10 and A10G; hardware and throughput differences may affect fixed-time comparisons.

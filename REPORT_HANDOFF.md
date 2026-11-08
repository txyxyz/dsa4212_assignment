# Report-writing handoff — 8 November 2026

## What we studied

Research question: under the same 30-minute training allowance on Modal's A10
option, how does GPT model size affect held-out prediction on TinyStories?
This is next-token story continuation, not an instruction-tuned chatbot.
All neural weights, including embeddings, start randomly. GPT-2 tokenization
rules and the Hugging Face GPT-2 implementation are reused, which the assignment permits.

## Completed work

1. Streamed 104,000 stories from roneneldan/TinyStories, using buffered shuffle
   seed 42 and buffer 10,000. Initial partitions were 100,000 / 2,000 / 2,000.
2. Checked exact cross-split overlaps: 55 train/validation, 77 train/test, and
   6 validation/test unique overlaps. Kept test unchanged, removed test stories
   from validation and all held-out stories from training. Final train: 99,868
   stories; validation: 1,994. Test initially had 2,000 stories; consult
   results/full-test/test_manifest.json for evaluated counts and hashes.
3. Tokenized using GPT-2, appending one EOS per story. Training has 22,323,961
   tokens; validation has 455,842. All models use the same tokenized text.
4. Debugged the local environment and short training loop. Ran 5- and 15-minute
   preliminary small-model pilots, then fixed the final protocol BEFORE final runs.
5. Completed six final runs and full validation/test evaluation. No further
   training is required for this main comparison. Test results have been seen;
   do not tune methods against them.

## Final setup

| Model | Layers | Width | Heads | Parameters |
|---|---:|---:|---:|---:|
| Small | 4 | 128 | 4 | 7,242,624 |
| Medium | 6 | 256 | 4 | 17,637,632 |
| Large | 8 | 384 | 6 | 33,544,320 |

Each has seeds 42 and 43, context 128, batch 16, AdamW constant learning rate
0.0003, bfloat16 GPU training, dropout per saved config, and gradient clipping
at 1.0. Depth and width change together; this is not an isolated depth study.
The vocabulary embedding accounts for much of the smallest model's parameters.

Budget is wall time starting BEFORE initial validation, including periodic
validation and checkpointing. Initialization/startup and final evaluation/saving
are excluded from that allowance but incur cloud charges. Each final loop stops
near 1,798 seconds, leaving a conservative margin. Do not claim exactly 1,800
seconds of optimizer updates or exactly 30 minutes of total billed GPU time.
Validation monitoring runs roughly every 30 seconds on only 40 fixed windows;
checkpoints overwrite every five minutes and at the end. Final checkpoint,
not minimum monitoring-loss checkpoint, is used for comparison.

## Results ready to cite

Full evaluation uses CPU float32, 128-token windows with stride 127, scoring each
next-token target once. Context resets at every window; EOS tokens count toward
loss. Loss is token-weighted cross-entropy in natural-log units (nats).
Perplexity for each run is exp(loss). Aggregate perplexity is the arithmetic
mean of per-run perplexities, not necessarily exp(mean loss).

| Model | Mean validation loss | Mean test loss ± sample SD | Mean test perplexity |
|---|---:|---:|---:|
| Small | 2.163921 | 2.164119 ± 0.000622 | 8.706929 |
| Medium | 1.938332 | 1.939559 ± 0.001429 | 6.955688 |
| Large | 1.876061 | 1.878328 ± 0.000637 | 6.542556 |

Source of truth: results/full-validation/summary.json and results/full-test/summary.json.
Individual scores, checkpoint hashes and run paths are in per_run.csv and the
per-model JSON files in each directory. SD is across only TWO training seeds;
it is not a confidence interval or a measure of general dataset uncertainty.

The larger model performed best among these three choices despite fewer tokens
processed under the same time allowance. This supports a capacity benefit in
this experiment, not a universal scaling law or proof that larger is always better.

## Which run files to use

- Small 42: results/final-small-s42-files/final-small-s42/
- Small 43: results/final-small-s43-retry1-files/final-small-s43-retry1/
- Medium 42: results/final-medium-s42-files/final-medium-s42/
- Medium 43: results/final-medium-s43-files/final-medium-s43/
- Large 42: results/final-large-s42-files/final-large-s42/
- Large 43: results/final-large-s43-files/final-large-s43-retry1/

Every run includes settings/config, summary, data manifest and metrics.jsonl.
The JSONL rows distinguish training_loss and validation_loss. Plot against
seconds/60 for learning curves; the monitoring scores are not full validation.
Use tokens from the final metric row for compute-allocation comparisons.
The interrupted small-43 attempt and all pilots are excluded from final averages.

## Failed attempts and caveats to explain honestly

- Small seed 43 was canceled around 1,727 seconds. Its last saved checkpoint was
   at 1,518 seconds. It was rerun from scratch with the same seed; the completed
   retry is used. Keep failed compute separate from per-run final allowances.
- Large seed 43 is a retry-named run. The reason for that retry has not been
   established in the chat; confirm with the experiment owner rather than invent it.
- Some devices report NVIDIA A10, others A10G. All requested Modal's A10 option.
   Record exact labels and throughput; do not claim perfectly identical hardware.
- Exact duplicate removal does not remove near duplicates. Repetitions within a
   split may remain. The synthetic narrow domain limits generalization claims.
- The subset uses buffered streaming shuffle, not a uniform sample of the full corpus.
- Only two seeds, one data subset, one context length and one shared optimizer
   recipe were tested. Models were not individually hyperparameter-optimized.
- Actual dollar cost has not been extracted from Modal billing; do not invent costs.
- Initial source revision and full dependency lock were not saved. Hashes support
   checking artifacts but do not by themselves guarantee fresh-download reproduction.
- Training checkpoints lack RNG state, and no resume implementation exists.

## What remains for the report

1. Read the original sources and explain the question, tokenizer, Transformer,
   next-token objective, fixed-time trade-off and metric in your own words.
2. Generate figures from saved metrics: monitoring loss vs time (show individual
   seeds), full test loss vs parameter count, and tokens processed vs model size.
3. Generate and save systematic qualitative examples from final checkpoints using
   the same prompts/decoding settings across models. Earlier pilot examples were
   informal; they are not a saved final-model qualitative comparison.
4. Draft concise discussion of capacity versus throughput, diminishing gains,
   seed variability, failures and limitations. Identify open questions.
5. Add complete report source/bibliography/figures in report/. Maximum PDF length
   is 10 pages including everything. Include names/student IDs and repo URL up front.
6. Resolve reproducibility gaps and checkpoint/data access before final submission.
   Repository may be private, but invite alexxthiery if so. Confirm grading access.

Suggested report allocation: intro/objective 1 page; data/evaluation 1–1.5;
architecture/training/budget 1–1.5; results and figures 2–3; discussion/limitations
1–1.5; references within the remaining space. This is a guide, not a required template.

## Sources to read and cite

- Assignment: https://alexxthiery.github.io/teaching/small-llm/small-llm.html
- Guidelines: https://alexxthiery.github.io/teaching/guidelines/guidelines.html
- TinyStories dataset: https://huggingface.co/datasets/roneneldan/TinyStories
- Compute-optimal training: https://arxiv.org/abs/2203.15556
- Data-constrained training: https://arxiv.org/abs/2305.16264
- nanoGPT: https://github.com/karpathy/nanoGPT (starter reference; our code uses HF GPT-2)
- Smol playbook: https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook

AI assistance was used for implementation, troubleshooting, evaluation and this
handoff. Each group member must verify sources/results and understand the work;
the handoff is not a finished submission or a substitute for your own analysis.

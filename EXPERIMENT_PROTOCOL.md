# Final comparison protocol

Question: Which of three GPT model sizes gives the lowest held-out predictive
loss with a fixed allowance of 1800 seconds on one NVIDIA A10 per training run?

All weights and embeddings are randomly initialized. Existing GPT-2 tokenization
rules are reused. No pretrained neural weights are loaded.

| Size | Layers | Embedding width | Attention heads | Seeds |
|---|---:|---:|---:|---|
| small | 4 | 128 | 4 | 42, 43 |
| medium | 6 | 256 | 4 | 42, 43 |
| large | 8 | 384 | 6 | 42, 43 |

Model size changes depth and width together; this does not isolate their effects.
Exact parameter counts are recorded automatically in settings.json.

Fixed across all runs: tokenized data and source hashes in data/tokens/manifest.json,
GPT-2 vocabulary, 128-token context, batch size 16, AdamW learning rate 0.0003,
bfloat16 precision, constant learning rate, 30-second validation interval and
300-second checkpoint interval. Other architecture/optimizer defaults are those
of the pinned code environment; save config.json and source with results.

The allowance begins before initial monitoring validation, includes periodic
validation and checkpointing, and ends before final evaluation/saving. A small
conservative stopping margin avoids overshoot. Initialization, startup, final
evaluation and saving incur charges but are excluded from this training allowance.
Report actual elapsed time and tokens processed. Interrupted runs are incomplete;
do not silently extend or restart their allowance. No automatic resume is provided.

There are six final runs (three GPU-hours of declared training in total). Existing
pilot-001 and pilot-002 remain preliminary and do not enter the final comparison.
Run one configuration at a time using unique output names. Check available credits
before launch. An OOM or hardware substitution requires documenting the failure
and revising the protocol consistently before final comparisons proceed.

Use the final checkpoint from every completed run. Monitoring uses only 40 fixed
validation windows and is not the final reported validation metric. Evaluate every
checkpoint on the full validation split using identical 128-token windows with
stride 127, weighting loss by the number of predicted tokens. After methods are
fixed, evaluate all final runs on the same untouched test data with the same rule.
Report cross-entropy in nats and perplexity; the tokenizer is identical across runs.
These windowed metrics reset context at each boundary; describe that limitation.

Report each seed individually and mean and standard deviation across the two
seeds. Two seeds offer only limited evidence of stability. Save tokens processed,
time, memory and loss curves. Qualitative prompts/decoding settings must be fixed
before comparing outputs. Conclusions apply to this synthetic story domain and
this particular budget/hardware/training recipe.

Command pattern (from project root):

```bash
arch -arm64 .venv-arm/bin/python -m modal run code/modal_train.py --run-id final-small-s42 --seconds 1800 --model-size small --seed 42 --purpose final
```

Repeat with (small,43), (medium,42), (medium,43), (large,42), (large,43),
changing run-id accordingly. No final runs are launched by editing this protocol.

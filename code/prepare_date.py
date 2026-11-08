from pathlib import Path
from datasets import load_dataset

SEED = 42
BUFFER_SIZE = 10_000
N_TRAIN = 100_000
N_VALID = 2_000
N_TEST = 2_000

output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

# streaming=True means the full dataset is not downloaded
dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train",
    streaming=True,
)

# Shuffle before taking the split. This is reproducible with the same seed.
dataset = dataset.shuffle(seed=SEED, buffer_size=BUFFER_SIZE)

files = {
    "train": open(output_dir / "train.txt", "w", encoding="utf-8"),
    "valid": open(output_dir / "valid.txt", "w", encoding="utf-8"),
    "test": open(output_dir / "test.txt", "w", encoding="utf-8"),
}

try:
    for index, example in enumerate(dataset):
        text = example["text"].strip()

        if not text:
            continue

        if index < N_TRAIN:
            split = "train"
        elif index < N_TRAIN + N_VALID:
            split = "valid"
        elif index < N_TRAIN + N_VALID + N_TEST:
            split = "test"
        else:
            break

        files[split].write(text + "\n<|endoftext|>\n")
finally:
    for file in files.values():
        file.close()

print("Done.")
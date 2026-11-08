"""Evaluate six final checkpoints locally. Test data requires --split test.

128-token windows, stride 127: every next-token target is scored once.
Context resets at each window; story EOS tokens are included in the metric.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

ROOT = Path(__file__).resolve().parents[1]


def evaluate(model, tokens, batch_size):
    total_nll, targets = 0.0, 0
    with torch.inference_mode():
        # Batch complete windows together; handle the final short window separately.
        starts = list(range(0, len(tokens) - 127, 127))
        for offset in range(0, len(starts), batch_size):
            selected = starts[offset:offset + batch_size]
            x = torch.from_numpy(np.stack([
                tokens[i:i + 128] for i in selected
            ]).astype('int64'))
            loss = model(x, labels=x, use_cache=False).loss.item()
            n = len(selected) * 127
            total_nll += loss * n
            targets += n
        last_start = len(starts) * 127
        if len(tokens) - last_start > 1:
            x = torch.tensor(np.array(tokens[last_start:], dtype=np.int64))[None, :]
            n = x.shape[1] - 1
            total_nll += model(x, labels=x, use_cache=False).loss.item() * n
            targets += n
    assert targets == len(tokens) - 1, (targets, len(tokens))
    loss = total_nll / targets
    if not math.isfinite(loss):
        raise RuntimeError('Non-finite evaluation loss')
    return dict(loss_nats=loss, perplexity=math.exp(loss), predicted_tokens=targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    torch.set_num_threads(4)
    if args.output is None:
        args.output = ROOT / f'results/full-{args.split}'
    manifest = json.loads((ROOT / 'data/tokens/manifest.json').read_text())
    runs = {}
    for path in (ROOT / 'results').rglob('summary.json'):
        settings = json.loads(path.read_text())
        if settings.get('purpose') != 'final' or settings.get('status') != 'completed':
            continue
        key = (settings['model_size'], settings['seed'])
        if key in runs:
            raise RuntimeError(f'Multiple completed copies of {key}; resolve duplicates first.')
        if json.loads((path.parent / 'data_manifest.json').read_text()) != manifest:
            raise RuntimeError(f'Dataset mismatch: {path}')
        runs[key] = (path.parent, settings)
    expected = {(size, seed) for size in ['small', 'medium', 'large'] for seed in [42, 43]}
    if set(runs) != expected:
        raise RuntimeError(f'Expected six final runs; found {sorted(runs)}')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.split == 'test':
        # Record fixed methods and checkpoint identities before computing test scores.
        protocol = dict(split='test', context=128, stride=127, precision='float32',
                        checkpoint_selection='Final checkpoint for each of six completed runs',
                        checkpoints={f'{s}-{seed}': hashlib.sha256((folder/'checkpoint.pt').read_bytes()).hexdigest()
                                     for (s, seed), (folder, _) in runs.items()})
        (args.output/'evaluation_protocol.json').write_text(json.dumps(protocol, indent=2))
        source = ROOT / 'data/clean/test.txt'
        raw = source.read_bytes()
        stories = [s.strip() for s in raw.decode('utf-8').split('<|endoftext|>') if s.strip()]
        tokenizer = GPT2TokenizerFast.from_pretrained(ROOT/'data/tokens/tokenizer', local_files_only=True)
        ids = []
        for story in stories:
            ids.extend(tokenizer.encode(story, add_special_tokens=False))
            ids.append(tokenizer.eos_token_id)
        token_path = ROOT/'data/tokens/test.bin'
        np.asarray(ids, dtype='<u4').tofile(token_path)
        split_manifest = dict(stories=len(stories), tokens=len(ids),
                              source_sha256=hashlib.sha256(raw).hexdigest(),
                              tokens_sha256=hashlib.sha256(token_path.read_bytes()).hexdigest())
        (args.output/'test_manifest.json').write_text(json.dumps(split_manifest, indent=2))
    else:
        token_path = ROOT/'data/tokens/valid.bin'
        split_manifest = manifest['splits']['valid']
    if hashlib.sha256(token_path.read_bytes()).hexdigest() != split_manifest['tokens_sha256']:
        raise RuntimeError('Evaluation token checksum differs from manifest')
    tokens = np.memmap(token_path, dtype='<u4', mode='r')
    rows = []
    for size in ['small', 'medium', 'large']:
        for seed in [42, 43]:
            folder, settings = runs[(size, seed)]
            print(f'Evaluating {size}, seed {seed}...', flush=True)
            checkpoint_path = folder / 'checkpoint.pt'
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            if checkpoint['step'] != settings['steps']:
                raise RuntimeError(f'Checkpoint is not the final step: {folder}')
            model = GPT2LMHeadModel(GPT2Config.from_pretrained(folder)).eval()
            model.load_state_dict(checkpoint['model'], strict=True)
            start = time.monotonic()
            scores = evaluate(model, tokens, args.batch_size)
            row = dict(model_size=size, seed=seed, parameters=settings['parameters'],
                       run_directory=str(folder.relative_to(ROOT)), **scores,
                       evaluation_seconds=time.monotonic() - start,
                       checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest())
            (args.output / f'{size}-s{seed}.json').write_text(json.dumps(row, indent=2))
            rows.append(row)
            print(f"  loss={scores['loss_nats']:.6f}, perplexity={scores['perplexity']:.4f}", flush=True)
            del model, checkpoint
    with (args.output / 'per_run.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped = []
    for size in ['small', 'medium', 'large']:
        selected = [r for r in rows if r['model_size'] == size]
        item = dict(model_size=size, seeds=[r['seed'] for r in selected])
        for metric in ['loss_nats', 'perplexity']:
            values = [r[metric] for r in selected]
            item[f'{metric}_mean'] = statistics.mean(values)
            item[f'{metric}_sample_std'] = statistics.stdev(values)
        grouped.append(item)
    report = dict(split=args.split, context=128, stride=127, device='cpu',
                  precision='float32', batch_size=args.batch_size,
                  torch_version=torch.__version__,
                  evaluation_tokens_sha256=split_manifest['tokens_sha256'],
                  note='Windowed next-token loss in nats, including EOS; context resets each window. Two-seed sample standard deviations have limited precision.',
                  results=grouped)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(grouped, indent=2), flush=True)
    print(f'Saved results to {args.output}')


if __name__ == '__main__':
    main()

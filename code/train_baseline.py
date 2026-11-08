"""Fresh initialization per run; elapsed budget includes validation/checkpointing.
No automatic resume: interrupted runs are incomplete and must not be reported
as complete budget comparisons. Existing run directories are never overwritten.
"""
import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

MODEL_SIZES = {
    'small': dict(n_embd=128, n_layer=4, n_head=4),
    'medium': dict(n_embd=256, n_layer=6, n_head=4),
    'large': dict(n_embd=384, n_layer=8, n_head=6),
}

def train(data, output, seconds=300, max_steps=0, batch_size=16, device='cpu', persist=None,
          eval_interval=30, checkpoint_interval=300, model_size='small', seed=42,
          purpose='preliminary'):
    if model_size not in MODEL_SIZES:
        raise ValueError(f'Unknown model size: {model_size}')
    if purpose not in ['preliminary', 'final']:
        raise ValueError('purpose must be preliminary or final')
    if purpose == 'final' and (seconds != 1800 or max_steps):
        raise ValueError('Final protocol requires 1800 seconds and no step limit.')
    if eval_interval <= 0 or checkpoint_interval <= 0:
        raise ValueError('Monitoring intervals must be positive.')
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((data / 'manifest.json').read_text())
    arrays = {s: np.memmap(data / f'{s}.bin', dtype='<u4', mode='r') for s in ['train', 'valid']}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    length = 128
    config = GPT2Config(vocab_size=manifest['vocab_size'], n_positions=length,
                       n_ctx=length, **MODEL_SIZES[model_size],
                       eos_token_id=manifest['eos_id'], bos_token_id=manifest['eos_id'])
    model = GPT2LMHeadModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    use_amp = device == 'cuda'
    def amp():
        return torch.autocast('cuda', dtype=torch.bfloat16) if use_amp else nullcontext()
    def sync():
        if device == 'cuda': torch.cuda.synchronize()
    def batch(split, starts):
        return torch.from_numpy(np.stack([arrays[split][int(i):int(i)+length] for i in starts]).astype('int64')).to(device)
    # Fixed validation windows, same for every run. All windows have equal length.
    val_starts = np.arange(min(40, len(arrays['valid']) // length)) * length
    assert len(val_starts) and len(arrays['train']) > length
    def evaluate():
        model.eval()
        total = 0.0
        with torch.no_grad():
            for offset in range(0, len(val_starts), batch_size):
                x = batch('valid', val_starts[offset:offset+batch_size])
                with amp(): loss = model(x, labels=x, use_cache=False).loss
                total += loss.item() * len(x)
        model.train()
        return total / len(val_starts)
    settings = dict(seed=seed, model_size=model_size, purpose=purpose,
                    architecture=MODEL_SIZES[model_size],
                    seconds=seconds, batch_size=batch_size, context=length,
                    learning_rate=3e-4, precision='bfloat16' if use_amp else 'float32',
                    device=torch.cuda.get_device_name() if use_amp else device,
                    parameters=sum(p.numel() for p in model.parameters()),
                    validation_windows=len(val_starts), max_steps=max_steps,
                    eval_interval=eval_interval, checkpoint_interval=checkpoint_interval,
                    budget_policy='Wall time from before initial validation; includes periodic evaluation and checkpointing. Final saving is outside the training allowance.')
    (output / 'settings.json').write_text(json.dumps(settings, indent=2))
    (output / 'data_manifest.json').write_text(json.dumps(manifest, indent=2))
    config.save_pretrained(output)
    sync()
    start = time.monotonic()
    step, last_save, last_eval, longest_step = 0, 0.0, 0.0, 1.0
    log = output / 'metrics.jsonl'
    def record(row):
        sync()
        row.update(step=step, seconds=time.monotonic()-start, tokens=step*batch_size*length)
        with log.open('a') as f: f.write(json.dumps(row)+'\n')
        print(row, flush=True)
    def save():
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'step': step, 'elapsed_seconds': time.monotonic()-start}, output / 'checkpoint.pt')
        if persist: persist()
    record({'validation_loss': evaluate()})
    last_eval = time.monotonic()-start
    model.train()
    while time.monotonic()-start + max(2.0, 2*longest_step) < seconds:
        if max_steps and step >= max_steps: break
        tick = time.monotonic()
        x = batch('train', rng.integers(0, len(arrays['train'])-length+1, batch_size))
        optimizer.zero_grad(set_to_none=True)
        with amp(): loss = model(x, labels=x, use_cache=False).loss
        if not torch.isfinite(loss).item(): raise RuntimeError('Non-finite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        sync()
        longest_step = max(longest_step, time.monotonic()-tick)
        step += 1
        if step == 1 or step % 25 == 0: record({'training_loss': loss.item()})
        elapsed = time.monotonic()-start
        if elapsed-last_eval >= eval_interval and seconds-elapsed > 30:
            record({'validation_loss': evaluate()})
            last_eval = time.monotonic()-start
        elapsed = time.monotonic()-start
        if elapsed-last_save >= checkpoint_interval and seconds-elapsed > 30:
            save()
            last_save = time.monotonic()-start
    training_elapsed = time.monotonic()-start
    record({'validation_loss': evaluate(), 'training_finished': True})
    save()
    summary = dict(settings, steps=step, training_elapsed_seconds=training_elapsed,
                   total_elapsed_seconds=time.monotonic()-start,
                   peak_gpu_bytes=torch.cuda.max_memory_allocated() if use_amp else None,
                   status='completed' if purpose == 'final' else 'preliminary')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    if persist: persist()
    return summary

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/tokens')
    p.add_argument('--output', required=True)
    p.add_argument('--seconds', type=int, default=60)
    p.add_argument('--max-steps', type=int, default=2)
    p.add_argument('--model-size', choices=MODEL_SIZES, default='small')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    train(args.data, args.output, args.seconds, args.max_steps, batch_size=2,
          model_size=args.model_size, seed=args.seed)

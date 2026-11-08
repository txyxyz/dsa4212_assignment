"""Run once locally. Test data is intentionally left untouched."""
import hashlib
import json
from pathlib import Path
import numpy as np
from transformers import GPT2TokenizerFast

ROOT = Path(__file__).resolve().parents[1]

def main():
    out = ROOT / 'data/tokens'
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', local_files_only=True)
    tokenizer.save_pretrained(out / 'tokenizer')
    manifest = {'vocab_size': len(tokenizer), 'eos_id': tokenizer.eos_token_id,
                'separator': '<|endoftext|>', 'splits': {}}
    for split in ['train', 'valid']:
        source = ROOT / 'data/clean' / f'{split}.txt'
        raw = source.read_bytes()
        stories = [s.strip() for s in raw.decode('utf-8').split('<|endoftext|>') if s.strip()]
        target = out / f'{split}.bin'
        count = 0
        with target.open('wb') as f:
            for offset in range(0, len(stories), 256):
                encoded = tokenizer(stories[offset:offset+256], add_special_tokens=False,
                                    truncation=False, return_attention_mask=False)['input_ids']
                for ids in encoded:
                    ids.append(tokenizer.eos_token_id)
                    np.asarray(ids, dtype='<u4').tofile(f)
                    count += len(ids)
        manifest['splits'][split] = {'stories': len(stories), 'tokens': count,
            'source_sha256': hashlib.sha256(raw).hexdigest(),
            'tokens_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
        print(split, manifest['splits'][split], flush=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()

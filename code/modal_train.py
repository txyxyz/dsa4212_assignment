"""Explicit cloud launch only; default is a five-minute preliminary test."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App('small-llm-baseline')
volume = modal.Volume.from_name('small-llm-results', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.13')
         .pip_install('torch==2.14.1', 'transformers==5.21.0', 'numpy==2.5.4')
         .add_local_file(ROOT / 'code/train_baseline.py', '/root/train_baseline.py')
         .add_local_dir(ROOT / 'data/tokens', '/data/tokens'))

@app.function(image=image, gpu='A10', timeout=9600, volumes={'/results': volume})
def run(run_id: str, seconds: int, model_size: str, seed: int, purpose: str):
    from train_baseline import train
    return train('/data/tokens', f'/results/{run_id}', seconds=seconds,
                 device='cuda', persist=volume.commit,
                 model_size=model_size, seed=seed, purpose=purpose)

@app.local_entrypoint()
def main(run_id: str = 'pilot-001', seconds: int = 300,
         model_size: str = 'small', seed: int = 42, purpose: str = 'preliminary'):
    if not run_id.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Use letters, digits, hyphens and underscores for run-id.')
    if not 30 <= seconds <= 9000:
        raise ValueError('seconds must be between 30 and 9000')
    if model_size not in ['small', 'medium', 'large']:
        raise ValueError('model-size must be small, medium or large')
    if purpose not in ['preliminary', 'final']:
        raise ValueError('purpose must be preliminary or final')
    if purpose == 'final' and seconds != 1800:
        raise ValueError('Final protocol requires --seconds 1800')
    print(run.remote(run_id, seconds, model_size, seed, purpose))


import os, sys, time
from huggingface_hub import hf_hub_download

local_dir = 'models/cogvideox-2b'
log = open('download_cogvideo.log', 'w')

files = [
    ('transformer/diffusion_pytorch_model.safetensors', 3.15),
    ('text_encoder/model-00001-of-00002.safetensors', 4.65),
]

for fname, expected_gb in files:
    fp = os.path.join(local_dir, fname)
    if os.path.exists(fp) and os.path.getsize(fp) > expected_gb * 0.95 * 1024**3:
        log.write(f'SKIP {fname}: already exists\n')
        log.flush()
        continue
    log.write(f'Downloading {fname} ({expected_gb} GB)...\n')
    log.flush()
    t0 = time.time()
    try:
        hf_hub_download(repo_id='THUDM/CogVideoX-2b', filename=fname, local_dir=local_dir)
        sz = os.path.getsize(fp) / (1024**3) if os.path.exists(fp) else 0
        log.write(f'  Done in {time.time()-t0:.0f}s ({sz:.2f} GB)\n')
    except Exception as e:
        log.write(f'  ERROR: {e}\n')
    log.flush()

log.write('All done.\n')
log.close()

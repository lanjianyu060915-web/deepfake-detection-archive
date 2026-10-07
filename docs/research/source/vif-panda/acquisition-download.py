import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
from huggingface_hub import get_hf_file_metadata, hf_hub_url

SIZE = 1294907616
SHA = '6f47fc9935c6545c7c237a6d15ef8c0e7e26a6826160bb54a6f2f5e83d2c6fae'
REV = 'c9e9b30e47e216f72c50664c43750674a79c0ded'
base = Path('cache/downloads/vif')
parts = base / 'ranges'
parts.mkdir(parents=True, exist_ok=True)
final = base / 'assets/source_videos.zip'
if final.exists():
    raise RuntimeError('final archive already exists; verify existing artifact')
metadata = get_hf_file_metadata(hf_hub_url(repo_id='JoeLeelyf/ViF-CoT-4K', filename='source_videos.zip', repo_type='dataset', revision=REV))
assert metadata.size == SIZE and metadata.etag == SHA
url = metadata.location
chunk_size = 5 * 1024**2

def fetch(start):
    end = min(start + chunk_size, SIZE) - 1
    target = parts / f'{start:012d}-{end:012d}.part'
    if target.exists() and target.stat().st_size == end - start + 1:
        return target
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
            with urllib.request.urlopen(request, timeout=45) as response:
                assert response.status == 206
                assert response.headers.get('Content-Range') == f'bytes {start}-{end}/{SIZE}'
                temporary = target.with_suffix('.tmp')
                with temporary.open('wb') as output:
                    count = 0
                    while block := response.read(1024**2):
                        count += len(block)
                        if count > end - start + 1:
                            raise RuntimeError('range exceeded requested bytes')
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
                assert count == end - start + 1
            temporary.replace(target)
            return target
        except Exception as exc:
            if attempt == 3:
                raise RuntimeError(f'range {start} failed: {type(exc).__name__}') from None
            time.sleep(2**attempt)

started = time.monotonic()
completed = 0
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    futures = [pool.submit(fetch, start) for start in range(0, SIZE, chunk_size)]
    for future in concurrent.futures.as_completed(futures):
        completed += future.result().stat().st_size
        if completed % (10 * chunk_size) == 0 or completed == SIZE:
            print(json.dumps({'completed_bytes': completed, 'total_bytes': SIZE, 'elapsed_seconds': round(time.monotonic()-started, 1)}), flush=True)
digest = hashlib.sha256()
assembly = final.with_suffix('.zip.part')
with assembly.open('xb') as output:
    for start in range(0, SIZE, chunk_size):
        end = min(start + chunk_size, SIZE) - 1
        with (parts / f'{start:012d}-{end:012d}.part').open('rb') as source:
            while block := source.read(1024**2):
                digest.update(block)
                output.write(block)
    output.flush()
    os.fsync(output.fileno())
assert assembly.stat().st_size == SIZE and digest.hexdigest() == SHA
os.link(assembly, final)
assembly.unlink()
print(json.dumps({'archive': str(final), 'bytes': SIZE, 'sha256': digest.hexdigest(), 'revision': REV}), flush=True)

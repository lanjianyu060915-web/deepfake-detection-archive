# -*- coding: utf-8 -*-
"""M3 二代深伪检测器：模型下载（缓存只写入升级工作区）。

用法：
    set HF_HOME=D:\刑科技项目升级2.0\experiments\models_cache
    python download_models.py [模型ID ...]
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import bootstrap  # noqa: E402

bootstrap()

ROOT = Path(r"D:\刑科技项目升级2.0")
MODELS_CACHE = ROOT / "experiments" / "models_cache"
LOG = ROOT / "experiments" / "logs" / "m3_download.log"

CANDIDATES = [
    "buildborderless/CommunityForensics-DeepfakeDet-ViT",
    "dima806/deepfake_vs_real_image_detection",
    "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "Wvolf/ViT_Deepfake_Detection",
]


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> None:
    from huggingface_hub import snapshot_download

    targets = sys.argv[1:] or CANDIDATES
    log(f"HF_HOME={os.environ['HF_HOME']}")
    for repo_id in targets:
        started = time.time()
        try:
            path = snapshot_download(
                repo_id,
                allow_patterns=[
                    "*.json", "*.txt", "*.md", "*.py", "*.safetensors",
                    "*.bin", "*.model", "*.jpg", "*.png",
                ],
                max_workers=4,
            )
            size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
            log(f"OK {repo_id} -> {path} ({size / 1e6:.1f} MB, {time.time() - started:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            log(f"FAIL {repo_id}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()

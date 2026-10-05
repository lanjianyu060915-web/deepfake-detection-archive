# -*- coding: utf-8 -*-
"""把工作区隔离依赖目录（pylibs）接到 sys.path 末尾。

追加而非前插，保证 venv 自带的 numpy / torch / pyyaml 优先，
只补齐 venv 里没有的 transformers / huggingface_hub / tokenizers / safetensors。
"""
import os
import sys
from pathlib import Path

UPGRADE_ROOT = Path(r"D:\刑科技项目升级2.0")
PYLIBS = UPGRADE_ROOT / "experiments" / "pylibs"
MODELS_CACHE = UPGRADE_ROOT / "experiments" / "models_cache"


def bootstrap() -> None:
    if PYLIBS.is_dir() and str(PYLIBS) not in sys.path:
        sys.path.append(str(PYLIBS))
    os.environ.setdefault("HF_HOME", str(MODELS_CACHE))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

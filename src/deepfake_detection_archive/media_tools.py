"""Select local media executables consistently for API calls and launchers."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path


_MODULE = Path(__file__).resolve()
_BUNDLE_BIN = (
    _MODULE.parents[2] / "runtime/media-tools/bin"
    if _MODULE.parent.parent.name == "src"
    else None
)


@cache
def _selected_bin(explicit: str | None, bundle: Path | None) -> Path | None:
    """Validate each configured immutable tool directory once per process."""
    if explicit is not None:
        selected = Path(explicit)
        origin = "DEEPFAKE_MEDIA_TOOLS"
    elif bundle is not None and bundle.parent.exists():
        selected = bundle
        origin = "bundled media tools"
    else:
        return None
    for name in ("ffmpeg", "ffprobe"):
        binary = selected / name
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise RuntimeError(f"{origin} requires executable {binary}")
    return selected


def _configured_bin() -> Path | None:
    explicit = os.environ.get("DEEPFAKE_MEDIA_TOOLS")
    if explicit:
        explicit = str(Path(explicit).expanduser().absolute())
    return _selected_bin(explicit or None, _BUNDLE_BIN)


def media_tool(name: str) -> str:
    """Keep host command lookup unchanged unless local tools are selected."""
    if name not in ("ffmpeg", "ffprobe"):
        return name
    selected = _configured_bin()
    return str(selected / name) if selected is not None else name


def prepare_media_tools_path() -> None:
    """Expose the same selected tools to ordinary child processes."""
    selected = _configured_bin()
    if selected is not None:
        os.environ["PATH"] = str(selected) + os.pathsep + os.environ.get("PATH", "")

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


APPLICATION_DATA_DIRECTORY = "DeepfakeDetectionSystem"
RESOURCE_ROOT_ENV = "DEEPFAKE_RESOURCE_ROOT"
USER_DATA_ROOT_ENV = "DEEPFAKE_USER_DATA_ROOT"


def is_frozen_application() -> bool:
    """返回当前进程是否由 PyInstaller 冻结。"""
    return bool(getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"))


def resolve_bundle_root() -> Path:
    """返回只读包内资源根；开发环境返回项目根。"""
    if is_frozen_application():
        return Path(str(getattr(sys, "_MEIPASS"))).resolve()
    return Path(__file__).resolve().parents[2]


def resolve_resource_root(source_root: str | Path | None = None) -> Path:
    """返回模型、配置和示例素材所在目录。"""
    configured = os.environ.get(RESOURCE_ROOT_ENV, "").strip()
    if configured:
        return Path(configured).resolve()
    if is_frozen_application():
        return Path(sys.executable).resolve().parent
    if source_root is not None:
        return Path(source_root).resolve()
    return Path(__file__).resolve().parents[2]


def resolve_user_data_root(resource_root: str | Path | None = None) -> Path:
    """冻结环境写入 LOCALAPPDATA；开发环境保持原有项目内行为。"""
    configured = os.environ.get(USER_DATA_ROOT_ENV, "").strip()
    if configured:
        return Path(configured).resolve()
    if is_frozen_application():
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if not local_app_data:
            raise RuntimeError("无法确定LOCALAPPDATA，不能创建安全的运行结果目录。")
        return (Path(local_app_data) / APPLICATION_DATA_DIRECTORY).resolve()
    return resolve_resource_root(resource_root)


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def sample_inputs_dir(self) -> Path:
        return self.root / "sample_inputs"

    @property
    def demo_data_dir(self) -> Path:
        return self.root / "data_all" / "data_demo"

    @property
    def preprocessing_test_dir(self) -> Path:
        return self.outputs_dir / "preprocessing_test"

# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata


REPO_ROOT = Path(SPECPATH).resolve().parents[1]
FRONTEND_ROOT = REPO_ROOT / "frontend"
COMPONENT_ROOT = FRONTEND_ROOT / "components"
REPORTING_ROOT = REPO_ROOT / "reporting"


def include_path(path):
    """Keep formal runtime files while excluding caches, tests and demo trees."""
    lowered = [part.lower() for part in path.parts]
    if "__pycache__" in lowered or "tests" in lowered:
        return False
    return not any(part.startswith(("mock", "sample", "demo")) for part in lowered)


def append_data_tree(target, source, destination):
    if not source.is_dir():
        return
    for path in sorted(source.rglob("*")):
        if not path.is_file() or not include_path(path.relative_to(source)):
            continue
        relative_parent = path.relative_to(source).parent
        target.append((str(path), str(Path(destination) / relative_parent)))


def package_modules(source, prefix):
    modules = []
    if not source.is_dir():
        return modules
    for path in sorted(source.rglob("*.py")):
        relative = path.relative_to(source)
        if not include_path(relative):
            continue
        if path.stem == "__init__":
            parts = relative.parent.parts
        else:
            parts = relative.with_suffix("").parts
        modules.append(".".join((prefix, *parts)) if parts else prefix)
    return modules

streamlit_datas, streamlit_binaries, streamlit_hiddenimports = collect_all(
    "streamlit", include_py_files=True
)

# Streamlit executes app.py as a data file. Collect formal components and the
# deterministic reporting package generically so later explainability/progress
# modules are picked up without adding one-off entries.
datas = [
    (str(FRONTEND_ROOT / "app.py"), "frontend"),
]
append_data_tree(datas, COMPONENT_ROOT, "frontend/components")
append_data_tree(datas, REPORTING_ROOT, "reporting")
for runtime_name in (
    "orchestrator.py",
    "runtime_services.py",
    "system_status.py",
    "progress_events.py",
):
    runtime_path = REPO_ROOT / runtime_name
    if runtime_path.is_file():
        datas.append((str(runtime_path), "."))
for optional_name in ("assets", "templates"):
    optional_path = FRONTEND_ROOT / optional_name
    append_data_tree(datas, optional_path, f"frontend/{optional_name}")

datas += streamlit_datas
datas += copy_metadata("streamlit", recursive=True)

local_hiddenimports = package_modules(COMPONENT_ROOT, "components")
local_hiddenimports += package_modules(REPORTING_ROOT, "reporting")

a = Analysis(
    [str(REPO_ROOT / "packaging_v0.3" / "src" / "app_launcher.py")],
    pathex=[str(REPO_ROOT), str(FRONTEND_ROOT)],
    binaries=streamlit_binaries,
    datas=datas,
    hiddenimports=streamlit_hiddenimports
    + local_hiddenimports
    + [
        "orchestrator",
        "progress_events",
        "runtime_services",
        "system_status",
        "streamlit.web.cli",
        "streamlit.runtime.scriptrunner",
        "streamlit.components.v1",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "IPython", "notebook", "jupyter"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="换脸寻源系统_v0.4.0",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version=str(REPO_ROOT / "packaging_v0.3" / "pyinstaller" / "frontend_version_info.txt"),
    uac_admin=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="换脸寻源系统_v0.4.0",
)

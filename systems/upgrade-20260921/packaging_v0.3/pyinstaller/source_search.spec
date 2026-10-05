# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


REPO_ROOT = Path(SPECPATH).resolve().parents[1]
MODULE_ROOT = REPO_ROOT / "modules" / "source_search"

cv2_datas = collect_data_files("cv2", includes=["data/*.xml"])

a = Analysis(
    [str(MODULE_ROOT / "run_source_search.py")],
    pathex=[str(REPO_ROOT), str(MODULE_ROOT)],
    binaries=[],
    datas=cv2_datas,
    hiddenimports=["cv2", "numpy", "PIL.Image", "progress_events"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "IPython",
        "notebook",
        "jupyter",
        "torch",
        "torchvision",
        "tensorflow",
        "PyQt5",
        "PyQt6",
        "tkinter",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="source_search",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version=str(REPO_ROOT / "packaging_v0.3" / "pyinstaller" / "source_version_info.txt"),
    uac_admin=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="source_search",
)

# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


REPO_ROOT = Path(SPECPATH).resolve().parents[1]
MODULE_ROOT = REPO_ROOT / "modules" / "identity_search"

datas = []
binaries = []
hiddenimports = []

datas += collect_data_files("insightface", include_py_files=True)
hiddenimports += collect_submodules("insightface")
datas += collect_data_files("onnxruntime", include_py_files=False)
binaries += collect_dynamic_libs("onnxruntime")
hiddenimports += [
    "onnxruntime.capi._pybind_state",
    "onnxruntime.capi.onnxruntime_pybind11_state",
    "cv2",
    "numpy",
    "PIL.Image",
    "skimage.transform",
    "sklearn.preprocessing",
    "albumentations",
]
for distribution in (
    "insightface",
    "onnxruntime",
    "albumentations",
    "opencv-python-headless",
):
    datas += copy_metadata(distribution, recursive=True)

a = Analysis(
    [str(MODULE_ROOT / "run_identity_search.py")],
    pathex=[str(MODULE_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
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
    name="identity_search",
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
    version=str(REPO_ROOT / "packaging" / "pyinstaller" / "identity_version_info.txt"),
    uac_admin=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="identity_search",
)

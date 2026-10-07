# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata


REPO_ROOT = Path(SPECPATH).resolve().parents[1]

streamlit_datas, streamlit_binaries, streamlit_hiddenimports = collect_all(
    "streamlit", include_py_files=True
)

datas = [
    (str(REPO_ROOT / "frontend"), "frontend"),
    (str(REPO_ROOT / "orchestrator.py"), "."),
]
datas += streamlit_datas
datas += copy_metadata("streamlit", recursive=True)

a = Analysis(
    [str(REPO_ROOT / "packaging" / "src" / "app_launcher.py")],
    pathex=[str(REPO_ROOT), str(REPO_ROOT / "frontend")],
    binaries=streamlit_binaries,
    datas=datas,
    hiddenimports=streamlit_hiddenimports + [
        "orchestrator",
        "components.data_loader",
        "components.path_security",
        "components.result_views",
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
    name="换脸寻源双搜索系统",
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
    version=str(REPO_ROOT / "packaging" / "pyinstaller" / "frontend_version_info.txt"),
    uac_admin=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="换脸寻源双搜索系统",
)

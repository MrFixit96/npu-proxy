# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for NPU Proxy.

Build with: pyinstaller npu_proxy.pyinstaller.spec

This creates a standalone executable that includes:
- OpenVINO runtime
- All Python dependencies
- Pre-configured for Intel NPU inference
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Get the project root
PROJ_ROOT = Path(SPECPATH)

# Analysis configuration
block_cipher = None

try:
    openvino_hiddenimports = collect_submodules('openvino') + collect_submodules('openvino_genai')
except Exception as e:
    print(f"Warning: Could not collect OpenVINO submodules: {e}")
    openvino_hiddenimports = []

try:
    openvino_datas = (
        collect_data_files('openvino', include_py_files=False)
        + collect_data_files('openvino_genai', include_py_files=False)
    )
except Exception as e:
    print(f"Warning: Could not collect OpenVINO data files: {e}")
    openvino_datas = []

try:
    npu_proxy_hiddenimports = collect_submodules(
        'npu_proxy',
        filter=lambda name: not (name.startswith('npu_proxy.tests') or name.startswith('tests')),
    )
except Exception as e:
    print(f"Warning: Could not collect npu_proxy submodules: {e}")
    npu_proxy_hiddenimports = ['npu_proxy']

# Collect all npu_proxy package files
a = Analysis(
    ['npu_proxy/cli.py'],
    pathex=[str(PROJ_ROOT)],
    binaries=[],
    datas=openvino_datas,
    hiddenimports=[
        # FastAPI and dependencies
        'fastapi',
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.http.h11_impl',
        'uvicorn.protocols.http.httptools_impl',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'uvicorn.lifespan.off',
        'starlette',
        'starlette.routing',
        'starlette.middleware',
        'starlette.responses',
        'sse_starlette',
        'pydantic',
        'pydantic.deprecated',
        'pydantic_core',
        'orjson',
        'httpx',
        'anyio',
        'sniffio',
        'openvino',
        'openvino_genai',
    ] + npu_proxy_hiddenimports + openvino_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce size
        'tkinter',
        'matplotlib',
        'PIL',
        'numpy.distutils',
        'setuptools',
        'wheel',
        'pip',
        'pytest',
        'test',
        'tests',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='npu-proxy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if desired
    version=None,  # Can add version_file.txt for Windows
)

# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for NPU Proxy.

Build with: pyinstaller npu_proxy.spec

This creates a standalone executable that includes:
- OpenVINO runtime
- All Python dependencies
- Pre-configured for Intel NPU inference
"""

import sys
from pathlib import Path

# Get the project root
PROJ_ROOT = Path(SPECPATH)

# Analysis configuration
block_cipher = None

# Collect all npu_proxy package files
a = Analysis(
    ['npu_proxy/cli.py'],
    pathex=[str(PROJ_ROOT)],
    binaries=[],
    datas=[
        # Include any data files (e.g., default configs)
    ],
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
        
        # OpenVINO - critical for NPU support
        'openvino',
        'openvino.runtime',
        'openvino_genai',
        
        # Prometheus metrics (optional)
        'prometheus_client',
        
        # NPU Proxy modules
        'npu_proxy',
        'npu_proxy.main',
        'npu_proxy.api',
        'npu_proxy.api.chat',
        'npu_proxy.api.embeddings',
        'npu_proxy.api.health',
        'npu_proxy.api.models',
        'npu_proxy.api.ollama',
        'npu_proxy.api.metrics',
        'npu_proxy.inference',
        'npu_proxy.inference.engine',
        'npu_proxy.inference.embedding_engine',
        'npu_proxy.inference.streaming',
        'npu_proxy.inference.tokenizer',
        'npu_proxy.models',
        'npu_proxy.models.registry',
        'npu_proxy.models.ollama_defaults',
        'npu_proxy.models.parameter_mapper',
        'npu_proxy.routing',
        'npu_proxy.routing.context_router',
        'npu_proxy.metrics',
    ],
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

# Collect OpenVINO submodules
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

try:
    a.hiddenimports += collect_submodules('openvino')
    a.hiddenimports += collect_submodules('openvino_genai')
except Exception as e:
    print(f"Warning: Could not collect OpenVINO submodules: {e}")

# Collect OpenVINO data files (runtime libraries)
try:
    a.datas += collect_data_files('openvino', include_py_files=False)
    a.datas += collect_data_files('openvino_genai', include_py_files=False)
except Exception as e:
    print(f"Warning: Could not collect OpenVINO data files: {e}")

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
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

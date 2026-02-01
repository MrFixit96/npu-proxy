# NPU Proxy Improvement Research Report

**Document Type**: Technical Research Report  
**Date**: 2026-02-01  
**Author**: AI-Assisted Research  
**Status**: COMPLETE

---

## Executive Summary

This report provides comprehensive research on 5 recommended improvements for the npu-proxy project. Research sources include OpenVINO official repositories, vLLM project, Ollama architecture, WinGet/apt packaging best practices, and community discussions.

---

## 1. Context-Aware Routing

### Current State
- `MAX_PROMPT_LEN=4096` configurable via environment variable
- Token counting via regex-based approximation in `tokenizer.py`
- Device fallback chain: NPU → GPU → CPU (load-time only)
- **Gap**: No runtime routing based on prompt length

### Research Findings

#### Industry Patterns (vLLM, TGI, Ollama)

| Server | Context Handling | Fallback Strategy |
|--------|------------------|-------------------|
| **vLLM** | Truncates to `max_model_len`, logs warning | No device fallback (single device) |
| **TGI** | Returns 400 error if exceeds limit | No automatic fallback |
| **Ollama** | Model-specific context windows | Graceful truncation |

#### Recommended Implementation

```python
# npu_proxy/inference/router.py

class ContextAwareRouter:
    """Routes requests based on prompt length and device capabilities."""
    
    NPU_SAFE_TOKENS = 1800  # Conservative limit from testing
    
    def __init__(self, npu_engine, cpu_engine):
        self.npu_engine = npu_engine
        self.cpu_engine = cpu_engine
    
    def route(self, prompt: str) -> tuple[InferenceEngine, dict]:
        """Returns (engine, metadata) based on prompt analysis."""
        token_count = count_tokens(prompt)
        
        if token_count <= self.NPU_SAFE_TOKENS:
            return self.npu_engine, {"device": "NPU", "routed_reason": "within_limit"}
        else:
            return self.cpu_engine, {
                "device": "CPU",
                "routed_reason": "prompt_too_long",
                "token_count": token_count,
                "limit": self.NPU_SAFE_TOKENS,
            }
```

#### HTTP Response Headers

Add routing metadata to responses:
```
X-NPU-Proxy-Device: CPU
X-NPU-Proxy-Route-Reason: prompt_too_long
X-NPU-Proxy-Token-Count: 2341
```

### Implementation Effort: 2-3 hours
### Priority: HIGH (prevents garbled output)

---

## 2. Native OS Packaging

### Overview

Native OS packages provide the best user experience for system services that require hardware access (NPU drivers). Unlike Docker, native packages can properly integrate with system services, hardware drivers, and startup management.

### Research Findings

#### Packaging Options Comparison

| Platform | Package Format | Package Manager | Service Manager | Best For |
|----------|---------------|-----------------|-----------------|----------|
| **Windows** | MSI / MSIX | winget, chocolatey | Windows Service | End users, enterprises |
| **Windows** | PyInstaller EXE | GitHub Releases | Task Scheduler | Developers, quick install |
| **Linux (Debian/Ubuntu)** | .deb | apt-get | systemd | Server deployments |
| **Linux (RHEL/Fedora)** | .rpm | dnf/yum | systemd | Enterprise Linux |
| **Linux (Any)** | Snap | snapd | snapd | Cross-distro, sandboxed |
| **Linux (Any)** | pipx | pipx | systemd (manual) | Python developers |
| **macOS** | Homebrew formula | brew | launchd | Mac developers |

---

### Windows Packaging

#### Option A: WinGet Package (Recommended for Distribution)

WinGet is Microsoft's official package manager. Submitting to `microsoft/winget-pkgs` provides:
- Discovery via `winget search npu-proxy`
- One-command install: `winget install npu-proxy`
- Automatic updates

**Manifest Structure** (`manifests/n/NPUProxy/NPUProxy/1.0.0/`):

```yaml
# NPUProxy.installer.yaml
PackageIdentifier: NPUProxy.NPUProxy
PackageVersion: 1.0.0
InstallerType: exe  # or msi
Installers:
  - Architecture: x64
    InstallerUrl: https://github.com/yourorg/npu-proxy/releases/download/v1.0.0/npu-proxy-1.0.0-win64.exe
    InstallerSha256: <sha256>
    InstallerSwitches:
      Silent: /S
      SilentWithProgress: /S
ManifestType: installer
ManifestVersion: 1.6.0
```

```yaml
# NPUProxy.locale.en-US.yaml
PackageIdentifier: NPUProxy.NPUProxy
PackageVersion: 1.0.0
PackageLocale: en-US
Publisher: NPU Proxy Contributors
PackageName: NPU Proxy
License: Apache-2.0
ShortDescription: Ollama-compatible API server for Intel NPU inference
Tags:
  - ai
  - llm
  - intel
  - npu
  - inference
ManifestType: defaultLocale
ManifestVersion: 1.6.0
```

#### Option B: PyInstaller Standalone EXE (Quickest Implementation)

Create a single `.exe` that bundles Python + all dependencies:

```python
# scripts/build_windows.py
"""Build Windows executable using PyInstaller."""
import PyInstaller.__main__
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

PyInstaller.__main__.run([
    str(ROOT / 'npu_proxy' / '__main__.py'),
    '--name=npu-proxy',
    '--onefile',
    '--console',  # Show console for server logs
    '--icon=assets/npu-proxy.ico',
    # Collect OpenVINO binaries
    '--collect-submodules=openvino',
    '--collect-data=openvino',
    '--collect-submodules=openvino_genai',
    # Hidden imports for dynamic loading
    '--hidden-import=uvicorn.logging',
    '--hidden-import=uvicorn.protocols.http',
    '--hidden-import=uvicorn.protocols.http.auto',
    '--hidden-import=uvicorn.lifespan.on',
    # Exclude unused heavy packages
    '--exclude-module=tkinter',
    '--exclude-module=matplotlib',
    '--exclude-module=pandas',
    '--exclude-module=scipy',
])
```

**Build command:**
```powershell
pip install pyinstaller
python scripts/build_windows.py
# Output: dist/npu-proxy.exe (~150-300MB)
```

#### Option C: Windows Service (Production)

For running as a background service that starts on boot:

```python
# scripts/windows_service.py
"""Windows Service wrapper for npu-proxy using pywin32."""
import win32serviceutil
import win32service
import win32event
import servicemanager
import subprocess
import sys
import os

class NPUProxyService(win32serviceutil.ServiceFramework):
    _svc_name_ = "NPUProxy"
    _svc_display_name_ = "NPU Proxy Server"
    _svc_description_ = "Ollama-compatible API server for Intel NPU inference"
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        self.process = None
    
    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.stop_event)
        if self.process:
            self.process.terminate()
    
    def SvcDoRun(self):
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, '')
        )
        self.main()
    
    def main(self):
        # Run uvicorn in subprocess
        self.process = subprocess.Popen([
            sys.executable, '-m', 'uvicorn',
            'npu_proxy.main:app',
            '--host', '0.0.0.0',
            '--port', '11435'
        ])
        # Wait for stop signal
        win32event.WaitForSingleObject(self.stop_event, win32event.INFINITE)

if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(NPUProxyService)
```

**Install service:**
```powershell
python scripts/windows_service.py install
python scripts/windows_service.py start
# Or via SC:
sc create NPUProxy binPath= "C:\path\to\npu-proxy.exe serve" start= auto
```

---

### Linux Packaging

#### Option A: Debian Package (.deb) - Ubuntu/Debian

**Package structure:**
```
npu-proxy_1.0.0-1_amd64/
├── DEBIAN/
│   ├── control
│   ├── postinst
│   ├── prerm
│   └── conffiles
├── usr/
│   ├── bin/
│   │   └── npu-proxy -> ../lib/npu-proxy/venv/bin/npu-proxy
│   └── lib/
│       └── npu-proxy/
│           ├── venv/           # Python virtual environment
│           └── npu_proxy/      # Application code
├── etc/
│   └── npu-proxy/
│       └── config.yaml
└── lib/
    └── systemd/
        └── system/
            └── npu-proxy.service
```

**DEBIAN/control:**
```
Package: npu-proxy
Version: 1.0.0-1
Section: utils
Priority: optional
Architecture: amd64
Depends: python3 (>= 3.10), python3-venv, intel-npu-driver
Maintainer: NPU Proxy Team <npu-proxy@example.com>
Description: Ollama-compatible API server for Intel NPU
 NPU Proxy provides OpenAI and Ollama-compatible APIs
 for running LLM inference on Intel Neural Processing Units.
Homepage: https://github.com/yourorg/npu-proxy
```

**systemd service file:**
```ini
# /lib/systemd/system/npu-proxy.service
[Unit]
Description=NPU Proxy - Intel NPU Inference Server
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=npu-proxy
Group=npu-proxy
Environment="NPU_PROXY_DEVICE=NPU"
Environment="NPU_PROXY_HOST=0.0.0.0"
Environment="NPU_PROXY_PORT=11435"
ExecStart=/usr/bin/npu-proxy serve
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/cache/npu-proxy

[Install]
WantedBy=multi-user.target
```

**Build script:**
```bash
#!/bin/bash
# scripts/build_deb.sh
set -e

VERSION="1.0.0"
PKG_NAME="npu-proxy"
PKG_DIR="${PKG_NAME}_${VERSION}-1_amd64"

# Create package structure
mkdir -p "$PKG_DIR"/{DEBIAN,usr/lib/npu-proxy,usr/bin,etc/npu-proxy,lib/systemd/system}

# Create virtual environment with dependencies
python3 -m venv "$PKG_DIR/usr/lib/npu-proxy/venv"
"$PKG_DIR/usr/lib/npu-proxy/venv/bin/pip" install -U pip
"$PKG_DIR/usr/lib/npu-proxy/venv/bin/pip" install .

# Create symlink
ln -s ../lib/npu-proxy/venv/bin/npu-proxy "$PKG_DIR/usr/bin/npu-proxy"

# Copy config and service files
cp packaging/linux/config.yaml "$PKG_DIR/etc/npu-proxy/"
cp packaging/linux/npu-proxy.service "$PKG_DIR/lib/systemd/system/"
cp packaging/linux/control "$PKG_DIR/DEBIAN/"

# Build package
dpkg-deb --build "$PKG_DIR"
# Output: npu-proxy_1.0.0-1_amd64.deb
```

**Installation:**
```bash
sudo dpkg -i npu-proxy_1.0.0-1_amd64.deb
sudo systemctl enable --now npu-proxy
```

#### Option B: pipx (Developer-Friendly)

For Python developers who want the latest version:

```bash
# Install
pipx install npu-proxy

# Or from git
pipx install git+https://github.com/yourorg/npu-proxy.git

# Run as service (manual systemd)
npu-proxy serve --daemon
```

**pyproject.toml entry point:**
```toml
[project.scripts]
npu-proxy = "npu_proxy.cli:main"
```

---

### macOS Packaging (Homebrew)

**Formula** (`Formula/npu-proxy.rb`):
```ruby
class NpuProxy < Formula
  desc "Ollama-compatible API server for Intel NPU inference"
  homepage "https://github.com/yourorg/npu-proxy"
  url "https://github.com/yourorg/npu-proxy/archive/refs/tags/v1.0.0.tar.gz"
  sha256 "<sha256>"
  license "Apache-2.0"

  depends_on "python@3.11"

  def install
    virtualenv_install_with_resources
    
    # Install launchd plist for service
    (prefix/"homebrew.mxcl.npu-proxy.plist").write <<~EOS
      <?xml version="1.0" encoding="UTF-8"?>
      <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
      <plist version="1.0">
      <dict>
        <key>Label</key>
        <string>homebrew.mxcl.npu-proxy</string>
        <key>ProgramArguments</key>
        <array>
          <string>#{opt_bin}/npu-proxy</string>
          <string>serve</string>
        </array>
        <key>RunAtLoad</key>
        <true/>
        <key>KeepAlive</key>
        <true/>
      </dict>
      </plist>
    EOS
  end

  service do
    run [opt_bin/"npu-proxy", "serve"]
    keep_alive true
  end

  test do
    system "#{bin}/npu-proxy", "--version"
  end
end
```

**Installation:**
```bash
brew tap yourorg/npu-proxy
brew install npu-proxy
brew services start npu-proxy
```

---

### Recommended Implementation Strategy

| Phase | Platform | Format | Effort | Priority |
|-------|----------|--------|--------|----------|
| 1 | Windows | PyInstaller EXE | 2-3 hrs | HIGH |
| 2 | Linux | pipx + systemd unit | 1-2 hrs | HIGH |
| 3 | Windows | WinGet manifest | 2 hrs | MEDIUM |
| 4 | Linux | .deb package | 3-4 hrs | MEDIUM |
| 5 | macOS | Homebrew formula | 2 hrs | LOW |

### Implementation Effort: 4-6 hours (Phase 1-2)
### Priority: HIGH (enables easy installation)

---

## 3. Prometheus Metrics

### Research Findings

#### vLLM Metrics (Industry Standard)

From `vllm/v1/metrics/loggers.py`:

```python
from prometheus_client import Counter, Gauge, Histogram

# Core metrics vLLM exposes
request_success = Counter('vllm:request_success_total', 'Successful requests')
request_failure = Counter('vllm:request_failure_total', 'Failed requests')
request_latency = Histogram('vllm:request_latency_seconds', 'Request latency',
                           buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10])
tokens_per_second = Gauge('vllm:tokens_per_second', 'Generation speed')
num_requests_running = Gauge('vllm:num_requests_running', 'Active requests')
```

#### Recommended Metrics for npu-proxy

```python
# npu_proxy/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info

# Latency (buckets tuned for NPU - slower than GPU)
inference_latency = Histogram(
    'npu_proxy_inference_duration_seconds',
    'Inference request latency',
    labelnames=['device', 'model', 'endpoint'],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
)

first_token_latency = Histogram(
    'npu_proxy_first_token_seconds',
    'Time to first token (streaming)',
    labelnames=['model'],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
)

# Throughput
requests_total = Counter(
    'npu_proxy_requests_total',
    'Total requests',
    labelnames=['device', 'model', 'status', 'endpoint']
)

tokens_generated = Counter(
    'npu_proxy_tokens_generated_total',
    'Total tokens generated',
    labelnames=['model', 'device']
)

# Resource utilization
active_requests = Gauge(
    'npu_proxy_active_requests',
    'Currently processing requests',
    labelnames=['device']
)

model_loaded = Gauge(
    'npu_proxy_model_loaded',
    'Model load status (1=loaded)',
    labelnames=['model', 'device']
)

# Info
build_info = Info(
    'npu_proxy_build',
    'Build information'
)
```

#### FastAPI Integration

```python
# npu_proxy/api/metrics.py
from fastapi import APIRouter
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

router = APIRouter()

@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
```

### Implementation Effort: 2-3 hours
### Priority: MEDIUM (observability)

---

## 4. Dual-Engine Architecture

### Problem Statement

The current architecture uses a single global engine:
```python
_llm_engine: InferenceEngine | None = None  # Only ONE engine active
```

This prevents running LLM inference and embeddings simultaneously - a common requirement for RAG (Retrieval-Augmented Generation) pipelines.

### Why NOT Multi-Model LRU Cache?

We explicitly evaluated and rejected an LRU cache approach:

| Factor | Reality on NPU |
|--------|----------------|
| Memory | 2-4GB max (can only hold ONE model) |
| Load time | 80-130 seconds per model |
| Cache hit rate | ~0% with varied workloads |
| User experience | Hidden 80s waits = frustration |

**For multi-tenant scenarios**: Run multiple proxy instances with a load balancer, not dynamic model swapping.

### Recommended: Separate Engines by Type

```python
# npu_proxy/inference/engines.py

# Separate engine instances by model type
_llm_engine: InferenceEngine | None = None      # LLM on NPU
_embedding_engine: EmbeddingEngine | None = None  # Embeddings on CPU

def get_llm_engine() -> InferenceEngine:
    """Get LLM engine (runs on NPU)."""
    global _llm_engine
    if _llm_engine is None:
        _llm_engine = InferenceEngine(model_path, device="NPU")
    return _llm_engine

def get_embedding_engine() -> EmbeddingEngine:
    """Get embedding engine (runs on CPU for reliability)."""
    global _embedding_engine
    if _embedding_engine is None:
        _embedding_engine = EmbeddingEngine(model_path, device="CPU")
    return _embedding_engine
```

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      NPU Proxy                               │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │   LLM Engine        │    │   Embedding Engine          │ │
│  │   (/v1/chat/...)    │    │   (/v1/embeddings)          │ │
│  │   Device: NPU       │    │   Device: CPU               │ │
│  │   Model: Mistral-7B │    │   Model: BGE-Small          │ │
│  └──────────┬──────────┘    └──────────────┬──────────────┘ │
│             │                              │                 │
│             ▼                              ▼                 │
│     ┌───────────────┐              ┌───────────────┐        │
│     │     NPU       │              │     CPU       │        │
│     │  (2-4GB)      │              │  (unlimited)  │        │
│     └───────────────┘              └───────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Benefits

| Benefit | Description |
|---------|-------------|
| **Simultaneous execution** | LLM + embeddings run in parallel |
| **No swapping** | Each engine stays loaded |
| **Optimal device usage** | NPU for LLM, CPU for embeddings |
| **Simple implementation** | No LRU logic, no eviction policies |
| **Predictable latency** | No hidden 80s model loads |

### Implementation Tasks

1. **Refactor engine globals** (1 hr)
   - Split `_llm_engine` into type-specific globals
   - Add `get_embedding_engine()` function

2. **Update API handlers** (1 hr)
   - `/v1/chat/completions` → uses `get_llm_engine()`
   - `/v1/embeddings` → uses `get_embedding_engine()`

3. **Add device configuration** (30 min)
   - `NPU_PROXY_LLM_DEVICE=NPU`
   - `NPU_PROXY_EMBED_DEVICE=CPU`

4. **Update health endpoint** (30 min)
   - Report both engines in `/health` response

### Implementation Effort: 2-3 hours
### Priority: HIGH (enables RAG pipelines)

---

## 5. VLMPipeline Integration

### Research Findings

#### OpenVINO VLMPipeline API

From `openvino_genai/py_openvino_genai.pyi`:

```python
class VLMPipeline:
    def __init__(
        self,
        models_path: str,
        device: str,
        **kwargs
    ) -> None: ...
    
    def generate(
        self,
        prompt: str,
        images: list[ov.Tensor] | None = None,
        generation_config: GenerationConfig | None = None,
        streamer: Callable | None = None,
    ) -> str: ...
```

#### Supported Models
- LLaVA (llava-1.5-7b, llava-v1.6-mistral-7b)
- InternVL (internvl-chat-v1.5)
- Qwen-VL (qwen-vl-chat)
- Phi-3-Vision
- MiniCPM-V

#### OpenAI Vision API Format

```json
{
  "model": "llava-v1.6-7b",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What's in this image?"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/..."}}
    ]
  }]
}
```

#### Recommended Implementation

```python
# npu_proxy/inference/vision_engine.py
import base64
from PIL import Image
import numpy as np
import openvino as ov
import openvino_genai as ov_genai

class VisionEngine:
    """Wrapper for OpenVINO VLMPipeline."""
    
    def __init__(self, model_path: str, device: str = "CPU"):
        self.pipeline = ov_genai.VLMPipeline(model_path, device)
        self.device = device
    
    def generate(
        self,
        prompt: str,
        images: list[str] | None = None,  # Base64 or URL
        max_tokens: int = 256,
    ) -> str:
        # Process images
        ov_images = []
        if images:
            for img_data in images:
                pil_image = self._decode_image(img_data)
                ov_tensor = self._to_tensor(pil_image)
                ov_images.append(ov_tensor)
        
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = max_tokens
        
        return self.pipeline.generate(prompt, images=ov_images, generation_config=config)
    
    def _decode_image(self, data: str) -> Image.Image:
        """Decode base64 or fetch URL."""
        if data.startswith("data:image"):
            # Base64 data URL
            _, encoded = data.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            return Image.open(io.BytesIO(img_bytes))
        else:
            # URL
            response = requests.get(data)
            return Image.open(io.BytesIO(response.content))
    
    def _to_tensor(self, image: Image.Image) -> ov.Tensor:
        """Convert PIL Image to OpenVINO Tensor."""
        np_image = np.array(image.convert("RGB"))
        return ov.Tensor(np_image)
```

#### Updated Message Model

```python
# npu_proxy/api/chat.py
class ContentPart(BaseModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: dict | None = None  # {"url": "..."}

class Message(BaseModel):
    role: str
    content: str | list[ContentPart]  # Support both formats
```

### Implementation Effort: 4-6 hours
### Priority: MEDIUM (feature expansion)

---

## Implementation Roadmap

| Phase | Feature | Effort | Dependencies |
|-------|---------|--------|--------------|
| 1 | Context-Aware Routing | 2-3 hrs | None |
| 2 | Native OS Packaging (Win EXE + Linux pipx) | 4-6 hrs | None |
| 3 | Prometheus Metrics | 2-3 hrs | None |
| 4 | Dual-Engine Architecture | 2-3 hrs | None |
| 5 | VLMPipeline | 4-6 hrs | Multi-model |

**Total Estimated Effort**: 15-21 hours

---

## Sources

1. **Microsoft WinGet Docs**: https://github.com/microsoft/winget-pkgs/blob/master/doc/README.md
2. **PyInstaller with OpenVINO**: https://github.com/zhaohb/paddleocr_vl_ov (reference spec file)
3. **vLLM Metrics**: https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/loggers.py
4. **OpenVINO GenAI**: https://github.com/openvinotoolkit/openvino.genai
5. **npu-proxy codebase**: Local analysis
6. **Prometheus Python Client**: https://github.com/prometheus/client_python
7. **Debian Packaging Guide**: https://wiki.debian.org/Packaging
8. **systemd Service Files**: https://www.freedesktop.org/software/systemd/man/systemd.service.html

---

*Research completed: 2026-02-01*

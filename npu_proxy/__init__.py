"""NPU Proxy package metadata."""

from importlib.metadata import PackageNotFoundError, version


def _detect_version() -> str:
    """Return the installed package version or the in-tree fallback."""
    try:
        return version("npu-proxy")
    except PackageNotFoundError:
        return "0.2.0"


__version__ = _detect_version()
OLLAMA_VERSION = f"{__version__}-npu-proxy"

__all__ = ["__version__", "OLLAMA_VERSION"]

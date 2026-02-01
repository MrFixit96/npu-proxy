"""Health endpoint handlers for the NPU Proxy service.

This module provides health check endpoints for monitoring service status,
device availability, and engine readiness.

Endpoints:
    GET /health: Primary health check with service status and engine info.
    GET /health/devices: Detailed device information and fallback chain.

Deployment Note:
    NPU Proxy runs as a **native host service**, not in containers.
    Intel NPU drivers require direct hardware access and cannot be
    containerized (no Docker, Kubernetes, or WSL2 passthrough for NPU).
    
    For WSL2 workloads, the proxy runs on Windows and WSL2 clients
    connect via HTTP bridge (see NPU_WSL2_PROXY_SPEC.md).

Service Management:
    - **Windows**: Use Windows Service (sc.exe) or Task Scheduler
    - **Linux**: Use systemd (see packaging/npu-proxy.service)

Load Balancer Health Checks:
    Configure your load balancer to probe GET /health and check:
    - ``status == 'healthy'`` for basic availability
    - ``engines.llm.status == 'loaded'`` for inference readiness
"""

from fastapi import APIRouter
import openvino as ov

from npu_proxy.inference.engine import is_model_loaded, get_loaded_models
from npu_proxy.inference.embedding_engine import get_embedding_engine

router = APIRouter(tags=["health"])

# Module-level cached OpenVINO Core instance for efficient device queries
_ov_core: ov.Core | None = None


def get_ov_core() -> ov.Core:
    """Get or create a cached OpenVINO Core instance.

    Uses module-level caching to avoid repeated Core initialization,
    which can be expensive on systems with many devices.

    Returns:
        ov.Core: Cached OpenVINO Core instance for device queries.

    Note:
        The Core instance is created lazily on first access and reused
        for all subsequent calls within the module lifetime.
    """
    global _ov_core
    if _ov_core is None:
        _ov_core = ov.Core()
    return _ov_core


def check_npu_available() -> bool:
    """Check if an Intel NPU device is available via OpenVINO.

    Queries the OpenVINO runtime to determine if an NPU (Neural Processing
    Unit) is present and accessible on the current system.

    Returns:
        bool: True if NPU device is detected, False otherwise.

    Note:
        Returns False on any exception (device busy, driver issues, etc.)
        to ensure graceful degradation.
    """
    try:
        return "NPU" in get_ov_core().available_devices
    except Exception:
        return False


def check_gpu_available() -> bool:
    """Check if an Intel integrated GPU is available via OpenVINO.

    Queries the OpenVINO runtime to determine if an Intel GPU (iGPU or
    discrete) is present and accessible for inference.

    Returns:
        bool: True if Intel GPU device is detected, False otherwise.

    Note:
        This checks for OpenVINO-compatible Intel GPUs only, not NVIDIA
        or AMD GPUs. Returns False on any exception.
    """
    try:
        return "GPU" in get_ov_core().available_devices
    except Exception:
        return False


@router.get("/health")
async def health() -> dict:
    """Check service health status and device availability.

    Provides a comprehensive health check for the NPU Proxy service,
    including device detection, engine status, and version information.
    Use this endpoint for load balancer health checks and monitoring.

    Returns:
        dict: Health response containing service status and diagnostics.

    Response Fields:
        status (str): Overall service status. Values:
            - ``'healthy'``: Service is fully operational
            - ``'degraded'``: Service running with reduced capability
            - ``'unhealthy'``: Service is not functional
        engines (dict): Status of inference engines:
            - ``llm``: LLM engine status object
            - ``embedding``: Embedding engine status object
        version (str): API version string (e.g., ``'0.1.0'``)
        npu_available (bool): Whether Intel NPU device is detected
        gpu_available (bool): Whether Intel GPU device is detected
        cpu_available (bool): Whether CPU is available (always True)
        devices (list[str]): List of all available OpenVINO devices
        openvino_version (str): OpenVINO runtime version

    Engine Status Object:
        status (str): Engine state. Values:
            - ``'loaded'``: Model loaded and ready for inference
            - ``'not_loaded'``: No model currently loaded
            - ``'fallback'``: Using fallback model (embedding only)
            - ``'error'``: Engine in error state
        device (str | None): Device running the model (e.g., ``'NPU'``, ``'CPU'``)
        model (str | None): Name/path of the loaded model

    Health Check Usage:
        - **Load Balancer**: Check ``status != 'unhealthy'`` for availability
        - **Monitoring**: Check ``engines.llm.status == 'loaded'`` for readiness
        - **Alerting**: Monitor for ``status == 'degraded'`` or ``'unhealthy'``

    Note:
        NPU Proxy runs as a native host service (Windows Service or systemd),
        not in containers. NPU hardware cannot be virtualized or containerized.

    Example:
        Response for a healthy service with NPU-accelerated LLM::

            {
                "status": "healthy",
                "engines": {
                    "llm": {
                        "status": "loaded",
                        "device": "NPU",
                        "model": "Phi-3-mini-4k-instruct-int4-ov"
                    },
                    "embedding": {
                        "status": "loaded",
                        "device": "CPU",
                        "model": "all-MiniLM-L6-v2"
                    }
                },
                "version": "0.1.0",
                "npu_available": true,
                "gpu_available": true,
                "cpu_available": true,
                "devices": ["CPU", "GPU", "NPU"],
                "openvino_version": "2024.0.0"
            }
    """
    core = get_ov_core()
    devices = core.available_devices
    
    # Check LLM engine status
    llm_status = "not_loaded"
    llm_device = None
    llm_model = None
    if is_model_loaded():
        try:
            loaded = get_loaded_models()
            for name, engine in loaded.items():
                info = engine.get_engine_info()
                llm_status = "loaded"
                llm_device = info.get("device", "unknown")
                llm_model = info.get("model_name", name)
                break
        except Exception:
            llm_status = "error"
    
    # Check embedding engine status
    embedding_status = "not_loaded"
    embedding_device = None
    embedding_model = None
    try:
        emb_engine = get_embedding_engine()
        if emb_engine is not None:
            info = emb_engine.get_engine_info()
            embedding_status = "fallback" if info.get("fallback", False) else "loaded"
            embedding_device = info.get("device", "CPU")
            embedding_model = info.get("model_name", "unknown")
    except Exception:
        embedding_status = "not_loaded"
    
    return {
        "status": "healthy",
        "engines": {
            "llm": {
                "status": llm_status,
                "device": llm_device,
                "model": llm_model,
            },
            "embedding": {
                "status": embedding_status,
                "device": embedding_device,
                "model": embedding_model,
            },
        },
        "version": "0.1.0",
        # Maintain backward compatibility
        "npu_available": "NPU" in devices,
        "gpu_available": "GPU" in devices,
        "cpu_available": "CPU" in devices,
        "devices": devices,
        "openvino_version": ov.__version__,
    }


@router.get("/health/devices")
async def get_devices() -> dict:
    """Get detailed device information and fallback chain status.

    Provides comprehensive device diagnostics including all available
    OpenVINO devices, the currently active inference device, and the
    device fallback priority chain.

    Returns:
        dict: Device information and status.

    Response Fields:
        available_devices (list[str]): All OpenVINO-compatible devices
            detected on the system (e.g., ``['CPU', 'GPU', 'NPU']``).
        active_device (str | None): The device currently running inference,
            or None if no model is loaded.
        device_info (dict | None): Detailed info about the active device:
            - ``actual_device``: Device identifier string
            - ``device_name``: Human-readable device name
            - Additional device-specific properties
        fallback_chain (list[str]): Priority order for device selection
            when loading models (``['NPU', 'GPU', 'CPU']``).

    Use Cases:
        - Debugging device detection issues
        - Verifying NPU/GPU is being utilized
        - Monitoring device fallback behavior

    Example:
        Response when LLM is running on NPU::

            {
                "available_devices": ["CPU", "GPU", "NPU"],
                "active_device": "NPU",
                "device_info": {
                    "actual_device": "NPU",
                    "device_name": "Intel(R) AI Boost"
                },
                "fallback_chain": ["NPU", "GPU", "CPU"]
            }
    """
    from npu_proxy.inference.engine import get_available_devices, is_model_loaded, get_loaded_models
    
    devices = get_available_devices()
    
    # Get active device info if model loaded
    active_device = None
    device_info = None
    if is_model_loaded():
        loaded = get_loaded_models()
        for name, engine in loaded.items():
            device_info = engine.get_device_info()
            active_device = device_info["actual_device"]
            break
    
    return {
        "available_devices": devices,
        "active_device": active_device,
        "device_info": device_info,
        "fallback_chain": ["NPU", "GPU", "CPU"],
    }

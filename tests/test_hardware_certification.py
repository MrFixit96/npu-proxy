"""Tests for hardware certification report evaluation."""

from npu_proxy.hardware_certification import evaluate_hardware_certification


def test_certification_passes_for_real_npu_run():
    runtime_details = {
        "openvino_version": "2026.1.0",
        "available_devices": ["CPU", "NPU"],
        "npu_visible": True,
        "npu_device_name": "Intel(R) AI Boost",
    }
    health_data = {
        "status": "healthy",
        "engines": {
            "llm": {
                "status": "loaded",
                "device": "NPU",
                "backend": "openvino",
                "requested_device": "NPU",
                "compile_cache_dir": "build\\runtime-cache",
                "compile_cache_mode": "OPTIMIZE_SPEED",
                "prefix_cache_mode": "on",
                "runtime_features": {
                    "compile_cache_enabled": True,
                    "prefix_cache_enabled": True,
                    "degraded_features": [],
                },
                "last_generation_stats": {
                    "ttft_seconds": 0.4,
                    "throughput_tokens_per_second": 25.0,
                },
            },
            "embedding": {
                "status": "loaded",
                "device": "CPU",
                "backend": "openvino",
                "model": "BAAI/bge-small-en-v1.5",
                "requested_device": "CPU",
                "is_production": True,
                "is_fallback": False,
                "cache": {
                    "enabled": True,
                    "kind": "lru",
                    "configured_max_entries": 1024,
                },
            }
        },
    }
    devices_data = {
        "active_device": "NPU",
        "device_info": {"used_fallback": False},
        "embedding": {
            "device_info": {
                "device": "CPU",
                "requested_device": "CPU",
            }
        },
    }
    generate_data = {"response": "Certified on Intel NPU."}

    report = evaluate_hardware_certification(
        runtime_details=runtime_details,
        health_data=health_data,
        devices_data=devices_data,
        generate_data=generate_data,
        requested_device="NPU",
        model="tinyllama-1.1b-chat-int4-ov",
        startup_seconds=1.2,
        inference_seconds=3.4,
    )

    assert report.certified is True
    assert report.failures == []
    assert report.active_device == "NPU"
    assert report.used_fallback is False
    assert report.llm_backend == "openvino"
    assert report.llm_runtime_state["compile_cache_mode"] == "OPTIMIZE_SPEED"
    assert report.embedding_backend == "openvino"
    assert report.embedding_runtime_state["cache"]["configured_max_entries"] == 1024


def test_certification_fails_when_proxy_falls_back():
    runtime_details = {
        "openvino_version": "2026.1.0",
        "available_devices": ["CPU", "NPU"],
        "npu_visible": True,
        "npu_device_name": "Intel(R) AI Boost",
    }
    health_data = {
        "status": "healthy",
        "engines": {
            "llm": {
                "status": "loaded",
                "device": "CPU",
                "backend": "llama_cpp",
            }
        },
    }
    devices_data = {
        "active_device": "CPU",
        "device_info": {"used_fallback": True},
    }
    generate_data = {"response": "Fallback response."}

    report = evaluate_hardware_certification(
        runtime_details=runtime_details,
        health_data=health_data,
        devices_data=devices_data,
        generate_data=generate_data,
        requested_device="NPU",
        model="tinyllama-1.1b-chat-int4-ov",
        startup_seconds=1.2,
        inference_seconds=3.4,
    )

    assert report.certified is False
    assert any("fallback" in failure.lower() for failure in report.failures)
    assert any("backend" in failure.lower() for failure in report.failures)

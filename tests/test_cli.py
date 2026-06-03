"""Focused CLI tests for startup and runtime-cache options."""

import logging
import os
import sys
from unittest.mock import patch

import pytest

from npu_proxy.config import get_active_llm_runtime_config, reset_active_proxy_bootstrap_config
from npu_proxy.cli import build_bootstrap_config, parse_args, set_environment
from npu_proxy.main import create_app


def test_parse_args_accepts_runtime_cache_options():
    """CLI flags should expose runtime cache options."""
    args = parse_args(
        [
            "--compile-cache-dir",
            "build\\runtime-cache",
            "--compile-cache-mode",
            "OPTIMIZE_SPEED",
            "--prefix-cache-mode",
            "on",
        ]
    )

    assert args.compile_cache_dir == "build\\runtime-cache"
    assert args.compile_cache_mode == "OPTIMIZE_SPEED"
    assert args.prefix_cache_mode == "on"


def test_parse_args_reads_runtime_cache_env():
    """Environment variables should populate runtime cache CLI defaults."""
    with patch.dict(
        os.environ,
        {
            "NPU_PROXY_COMPILE_CACHE_DIR": "build\\cache-from-env",
            "NPU_PROXY_COMPILE_CACHE_MODE": "optimize_size",
            "NPU_PROXY_PREFIX_CACHE_MODE": "off",
        },
        clear=False,
    ):
        args = parse_args([])

    assert args.compile_cache_dir == "build\\cache-from-env"
    assert args.compile_cache_mode == "OPTIMIZE_SIZE"
    assert args.prefix_cache_mode == "off"


def test_set_environment_sets_runtime_cache_vars():
    """set_environment should propagate runtime cache configuration."""
    args = parse_args(
        [
            "--device",
            "CPU",
            "--compile-cache-dir",
            "build\\runtime-cache",
            "--compile-cache-mode",
            "OPTIMIZE_SPEED",
            "--prefix-cache-mode",
            "off",
        ]
    )

    previous = {
        "NPU_PROXY_DEVICE": os.environ.get("NPU_PROXY_DEVICE"),
        "NPU_PROXY_COMPILE_CACHE_DIR": os.environ.get("NPU_PROXY_COMPILE_CACHE_DIR"),
        "NPU_PROXY_COMPILE_CACHE_MODE": os.environ.get("NPU_PROXY_COMPILE_CACHE_MODE"),
        "NPU_PROXY_PREFIX_CACHE_MODE": os.environ.get("NPU_PROXY_PREFIX_CACHE_MODE"),
    }

    try:
        set_environment(args)
        assert os.environ["NPU_PROXY_DEVICE"] == "CPU"
        assert os.environ["NPU_PROXY_COMPILE_CACHE_DIR"] == "build\\runtime-cache"
        assert os.environ["NPU_PROXY_COMPILE_CACHE_MODE"] == "OPTIMIZE_SPEED"
        assert os.environ["NPU_PROXY_PREFIX_CACHE_MODE"] == "off"
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_build_bootstrap_config_resolves_auto_to_authoritative_runtime_defaults():
    """CLI AUTO should resolve through the bootstrap control plane to the real runtime default."""
    args = parse_args([])

    config = build_bootstrap_config(args, env={})

    assert config.host == "127.0.0.1"
    assert config.port == 8080
    assert config.token_limit == 1800
    assert config.llm.device == "NPU"


def test_create_app_activates_explicit_bootstrap_config():
    """Programmatic app creation should activate the supplied control-plane config."""
    reset_active_proxy_bootstrap_config()
    try:
        config = build_bootstrap_config(parse_args(["--device", "CPU"]), env={})
        app = create_app(config)

        assert app.state.proxy_config.llm.device == "CPU"
        assert get_active_llm_runtime_config().device == "CPU"
    finally:
        reset_active_proxy_bootstrap_config()



def test_setup_logging_configures_stdout_handler(monkeypatch):
    """Console logging should use stdout and the requested level."""
    import npu_proxy.cli as cli

    captured = {}

    def fake_basic_config(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)

    cli.setup_logging("debug")

    assert captured["level"] == logging.DEBUG
    assert captured["force"] is True
    assert len(captured["handlers"]) == 1
    assert isinstance(captured["handlers"][0], logging.StreamHandler)
    assert captured["handlers"][0].stream is sys.stdout


def test_setup_logging_configures_file_handler(monkeypatch, tmp_path):
    """File logging should install a FileHandler for the requested path."""
    import npu_proxy.cli as cli

    captured = {}

    def fake_basic_config(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)
    log_file = tmp_path / "npu-proxy.log"

    cli.setup_logging("warning", str(log_file))

    try:
        assert captured["level"] == logging.WARNING
        assert len(captured["handlers"]) == 1
        assert isinstance(captured["handlers"][0], logging.FileHandler)
        assert captured["handlers"][0].baseFilename == str(log_file)
    finally:
        captured["handlers"][0].close()


def test_run_server_uses_app_instance_for_single_worker(monkeypatch):
    """Single-process mode should pass an app instance to uvicorn."""
    import npu_proxy.cli as cli
    import npu_proxy.main as main_module
    import uvicorn

    app = object()
    calls = {}

    def fake_create_app(config):
        calls["config"] = config
        return app

    def fake_run(*args, **kwargs):
        calls["run_args"] = args
        calls["run_kwargs"] = kwargs

    config = build_bootstrap_config(
        parse_args(["--host", "127.0.0.1", "--port", "9090", "--device", "CPU", "--log-level", "debug"]),
        env={},
    )
    monkeypatch.setattr(main_module, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", fake_run)

    cli.run_server(config)

    assert calls["config"] is config
    assert calls["run_args"] == (app,)
    assert calls["run_kwargs"] == {
        "host": "127.0.0.1",
        "port": 9090,
        "workers": 1,
        "reload": False,
        "log_level": "debug",
    }


@pytest.mark.parametrize("argv", [["--reload"], ["--workers", "2"]])
def test_run_server_uses_import_string_for_reload_or_workers(monkeypatch, argv):
    """Reload and multi-worker modes require an import-string app target."""
    import npu_proxy.cli as cli
    import npu_proxy.main as main_module
    import uvicorn

    calls = {}

    def fail_create_app(config):  # pragma: no cover - should not be called
        raise AssertionError("create_app should not be called")

    def fake_run(*args, **kwargs):
        calls["run_args"] = args
        calls["run_kwargs"] = kwargs

    config = build_bootstrap_config(parse_args(argv), env={})
    monkeypatch.setattr(main_module, "create_app", fail_create_app)
    monkeypatch.setattr(uvicorn, "run", fake_run)

    cli.run_server(config)

    assert calls["run_args"] == ("npu_proxy.main:app",)
    assert calls["run_kwargs"]["reload"] is config.reload
    assert calls["run_kwargs"]["workers"] == config.workers


def test_run_server_warns_for_non_loopback_default_allowed_hosts(monkeypatch, caplog):
    """Binding remotely with default Host allow-list should emit the operator warning."""
    import npu_proxy.cli as cli
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    config = build_bootstrap_config(parse_args(["--host", "0.0.0.0"]), env={})

    with caplog.at_level(logging.WARNING, logger="npu-proxy"):
        cli.run_server(config)

    assert "Binding to non-loopback host 0.0.0.0" in caplog.text


def test_main_bootstraps_runtime_mirrors_env_and_runs_server(monkeypatch):
    """main should bootstrap once, mirror the resolved config into env, and return success."""
    import npu_proxy.cli as cli
    import npu_proxy.main as main_module

    calls = {}
    monkeypatch.setenv("NPU_PROXY_COMPILE_CACHE_MODE", "OPTIMIZE_SPEED")

    def fake_bootstrap_runtime(config):
        calls["bootstrap_config"] = config
        return config

    def fake_run_server(config):
        calls["run_config"] = config

    monkeypatch.setattr(main_module, "bootstrap_runtime", fake_bootstrap_runtime)
    monkeypatch.setattr(cli, "run_server", fake_run_server)

    code = cli.main(
        [
            "--host",
            "localhost",
            "--port",
            "11434",
            "--workers",
            "2",
            "--device",
            "CPU",
            "--token-limit",
            "123",
            "--real-inference",
            "--compile-cache-dir",
            "build\\cache",
            "--prefix-cache-mode",
            "off",
            "--allowed-hosts",
            "localhost,127.0.0.1",
        ]
    )

    assert code == 0
    assert calls["bootstrap_config"] is calls["run_config"]
    assert calls["run_config"].llm.device == "CPU"
    assert os.environ["NPU_PROXY_HOST"] == "localhost"
    assert os.environ["NPU_PROXY_PORT"] == "11434"
    assert os.environ["NPU_PROXY_DEVICE"] == "CPU"
    assert os.environ["NPU_PROXY_TOKEN_LIMIT"] == "123"
    assert os.environ["NPU_PROXY_REAL_INFERENCE"] == "1"
    assert os.environ["NPU_PROXY_COMPILE_CACHE_DIR"] == "build\\cache"
    assert os.environ["NPU_PROXY_COMPILE_CACHE_MODE"] == "OPTIMIZE_SPEED"
    assert os.environ["NPU_PROXY_PREFIX_CACHE_MODE"] == "off"
    assert os.environ["NPU_PROXY_ALLOWED_HOSTS"] == "localhost,127.0.0.1"


def test_main_returns_zero_on_keyboard_interrupt(monkeypatch):
    """Ctrl+C during server startup should be treated as a clean shutdown."""
    import npu_proxy.cli as cli
    import npu_proxy.main as main_module

    monkeypatch.setattr(main_module, "bootstrap_runtime", lambda config: config)

    def raise_keyboard_interrupt(config):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_server", raise_keyboard_interrupt)

    assert cli.main([]) == 0


def test_main_returns_one_and_logs_fatal_error(monkeypatch, caplog):
    """Unexpected startup failures should become process exit code 1."""
    import npu_proxy.cli as cli
    import npu_proxy.main as main_module

    monkeypatch.setattr(main_module, "bootstrap_runtime", lambda config: config)
    monkeypatch.setattr(cli, "setup_logging", lambda level, log_file=None: None)

    def raise_runtime_error(config):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "run_server", raise_runtime_error)

    with caplog.at_level(logging.ERROR, logger="npu-proxy"):
        assert cli.main([]) == 1

    assert "Fatal error: boom" in caplog.text


@pytest.mark.parametrize(
    "argv",
    [
        ["--port", "0"],
        ["--port", "65536"],
        ["--workers", "0"],
        ["--token-limit", "0"],
    ],
)
def test_main_returns_one_for_invalid_numeric_startup_arguments(argv, monkeypatch):
    """Post-parse startup validation failures should be reported as fatal CLI errors."""
    import npu_proxy.cli as cli

    monkeypatch.setattr(
        cli,
        "run_server",
        lambda config: (_ for _ in ()).throw(AssertionError("server should not start")),
    )

    assert cli.main(argv) == 1


@pytest.mark.parametrize(
    "argv",
    [
        ["--port", "not-a-number"],
        ["--workers", "not-a-number"],
        ["--token-limit", "not-a-number"],
    ],
)
def test_parse_args_rejects_non_integer_numeric_arguments(argv):
    """argparse should reject non-integer numeric arguments before startup."""
    with pytest.raises(SystemExit) as exc_info:
        parse_args(argv)

    assert exc_info.value.code == 2

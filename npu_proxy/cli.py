#!/usr/bin/env python3
"""NPU Proxy command-line interface.

This module provides the main CLI entry point for running the NPU Proxy server.
It handles argument parsing, configuration, and server startup. The CLI is designed
to be compatible with multiple deployment scenarios including direct execution,
PyInstaller binaries, and package manager installations.

Usage:
    npu-proxy                      Start with defaults (localhost:8080)
    npu-proxy --port 8080          Custom port
    npu-proxy --host 0.0.0.0       Bind to all interfaces
    npu-proxy --workers 4          Multiple worker processes
    npu-proxy --device NPU         Force NPU device
    npu-proxy --log-level debug    Verbose logging
    npu-proxy --real-inference     Enable real model inference

Environment Variables:
    NPU_PROXY_HOST: Bind address (default: 127.0.0.1)
    NPU_PROXY_PORT: Port number (default: 8080)
    NPU_PROXY_DEVICE: Inference device - NPU, GPU, CPU, or AUTO (default: AUTO)
    NPU_PROXY_TOKEN_LIMIT: Token limit for NPU context routing (default: 1800)
    NPU_PROXY_COMPILE_CACHE_DIR: Optional OpenVINO compile cache directory
    NPU_PROXY_COMPILE_CACHE_MODE: Optional compile cache mode
        (OPTIMIZE_SIZE, OPTIMIZE_SPEED)
    NPU_PROXY_PREFIX_CACHE_MODE: Prefix cache mode (auto, on, off)
    NPU_PROXY_REAL_INFERENCE: Enable real inference when set to "1" (default: 0)
    NPU_PROXY_ALLOWED_HOSTS: Comma-separated Host-header allow-list

Example:
    # Start server with real inference on NPU
    $ npu-proxy --device NPU --real-inference

    # Development mode with auto-reload and debug logging
    $ npu-proxy --reload --log-level debug

    # Production deployment on all interfaces
    $ npu-proxy --host 0.0.0.0 --port 11434 --workers 4

    # Use Ollama's default port for drop-in replacement
    $ npu-proxy --port 11434 --device AUTO
"""
import argparse
import logging
import os
import sys

from npu_proxy import __version__
from npu_proxy.config import (
    DEFAULT_ALLOWED_HOSTS,
    ENV_ALLOWED_HOSTS,
    ProxyBootstrapConfig,
    VALID_COMPILE_CACHE_MODES,
    VALID_PREFIX_CACHE_MODES,
    apply_proxy_bootstrap_config_to_env,
    is_loopback_host,
    load_cli_environment_defaults,
    load_proxy_bootstrap_config,
    normalize_allowed_hosts,
)

# Configure logging before imports to catch early errors
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("npu-proxy")

def get_version() -> str:
    """Get the installed package version.

    Returns the shared package version used across the CLI, API, and docs.

    Returns:
        str: The semantic version string (e.g., "0.2.0").

    Example:
        >>> version = get_version()
        >>> print(f"Running NPU Proxy v{version}")
        Running NPU Proxy v0.2.0
    """
    return __version__

def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments into a configuration namespace.

    Defines and parses all CLI arguments organized into logical groups:
    - Server Options: host, port, workers, reload
    - Device Options: device, token-limit, real-inference
    - Logging Options: log-level, log-file

    Args:
        args: Optional list of argument strings to parse. If None, uses sys.argv.
            Primarily used for testing to inject specific argument sets.

    Returns:
        argparse.Namespace: Parsed arguments with the following attributes:
            - host (str): Bind address (default: 127.0.0.1)
            - port (int): Port number (default: 8080)
            - workers (int): Number of uvicorn workers (default: 1)
            - reload (bool): Enable auto-reload for development
            - device (str): Inference device - NPU, GPU, CPU, or AUTO
            - token_limit (int): Token limit for NPU routing (default: 1800)
            - compile_cache_dir (str | None): Optional compile cache directory
            - compile_cache_mode (str | None): Compile cache mode
            - prefix_cache_mode (str): Prefix cache mode (auto, on, off)
            - real_inference (bool): Enable real model inference
            - log_level (str): Logging verbosity level
            - log_file (str | None): Optional log file path

    Example:
        >>> args = parse_args(['--port', '11434', '--device', 'NPU'])
        >>> args.port
        11434
        >>> args.device
        'NPU'

        >>> # Testing with mock arguments
        >>> args = parse_args(['--real-inference', '--log-level', 'debug'])
        >>> args.real_inference
        True
    """
    defaults = load_cli_environment_defaults()

    parser = argparse.ArgumentParser(
        prog="npu-proxy",
        description="Ollama-compatible API proxy for Intel NPU inference via OpenVINO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  npu-proxy                         Start server on localhost:8080
  npu-proxy --port 11434            Use Ollama's default port
  npu-proxy --host 0.0.0.0          Accept connections from any interface
  npu-proxy --device CPU            Force CPU device (skip NPU)
  npu-proxy --real-inference        Enable real model inference
  
Environment Variables:
  NPU_PROXY_DEVICE         Default device (NPU, GPU, CPU)
  NPU_PROXY_TOKEN_LIMIT    Token limit for NPU (default: 1800)
  NPU_PROXY_COMPILE_CACHE_DIR  Compile cache directory
  NPU_PROXY_COMPILE_CACHE_MODE Compile cache mode
  NPU_PROXY_PREFIX_CACHE_MODE  Prefix cache mode (auto, on, off)
  NPU_PROXY_REAL_INFERENCE Enable real inference (1 to enable)
  NPU_PROXY_ALLOWED_HOSTS  Comma-separated Host-header allow-list
        """,
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"%(prog)s {get_version()}",
    )
    
    # Server options
    server_group = parser.add_argument_group("Server Options")
    server_group.add_argument(
        "--host",
        default=defaults.host,
        help="Host to bind to (default: 127.0.0.1)",
    )
    server_group.add_argument(
        "--port", "-p",
        type=int,
        default=defaults.port,
        help="Port to bind to (default: 8080)",
    )
    server_group.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )
    server_group.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    server_group.add_argument(
        "--allowed-hosts",
        default=",".join(defaults.allowed_hosts),
        help="Comma-separated Host-header allow-list (default: localhost/loopback/test clients)",
    )
    
    # Device options
    device_group = parser.add_argument_group("Device Options")
    device_group.add_argument(
        "--device", "-d",
        choices=["NPU", "GPU", "CPU", "AUTO"],
        default=defaults.device,
        help="Inference device (default: AUTO, tries NPU first)",
    )
    device_group.add_argument(
        "--token-limit",
        type=int,
        default=defaults.token_limit,
        help="Token limit for NPU context routing (default: 1800)",
    )
    device_group.add_argument(
        "--compile-cache-dir",
        default=defaults.compile_cache_dir,
        help="Enable OpenVINO compile cache in this directory",
    )
    device_group.add_argument(
        "--compile-cache-mode",
        choices=list(VALID_COMPILE_CACHE_MODES),
        default=defaults.compile_cache_mode,
        help="Compile cache mode (default: OpenVINO runtime default)",
    )
    device_group.add_argument(
        "--prefix-cache-mode",
        choices=list(VALID_PREFIX_CACHE_MODES),
        default=defaults.prefix_cache_mode,
        help="Prefix cache mode: auto keeps runtime defaults",
    )
    device_group.add_argument(
        "--real-inference",
        action="store_true",
        default=defaults.real_inference,
        help="Enable real model inference (default: mock mode)",
    )
    
    # Logging options
    log_group = parser.add_argument_group("Logging Options")
    log_group.add_argument(
        "--log-level", "-l",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Log level (default: info)",
    )
    log_group.add_argument(
        "--log-file",
        help="Log to file instead of stdout",
    )
    
    return parser.parse_args(args)


def build_bootstrap_config(
    args: argparse.Namespace,
    env: dict[str, str] | None = None,
) -> ProxyBootstrapConfig:
    """Resolve CLI arguments into the authoritative bootstrap config."""
    return load_proxy_bootstrap_config(
        env=env,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        token_limit=args.token_limit,
        real_inference=args.real_inference,
        log_level=args.log_level,
        log_file=args.log_file,
        allowed_hosts=args.allowed_hosts,
        device=args.device,
        compile_cache_dir=args.compile_cache_dir,
        compile_cache_mode=args.compile_cache_mode,
        prefix_cache_mode=args.prefix_cache_mode,
    )


def setup_logging(level: str, log_file: str | None = None) -> None:
    """Configure logging based on CLI options.

    Sets up the Python logging system with the specified verbosity level
    and output destination. Uses force=True to override any existing
    logging configuration.

    Args:
        level: Log level string (debug, info, warning, error, critical).
            Case-insensitive; will be converted to uppercase.
        log_file: Optional path to a log file. If provided, logs are written
            to this file instead of stdout. Parent directories must exist.

    Raises:
        PermissionError: If log_file path is not writable.
        FileNotFoundError: If log_file directory does not exist.

    Example:
        >>> # Console logging at debug level
        >>> setup_logging('debug')

        >>> # File logging for production
        >>> setup_logging('info', '/var/log/npu-proxy.log')
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = []
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    else:
        handlers.append(logging.StreamHandler(sys.stdout))
    
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=log_level,
        handlers=handlers,
        force=True,
    )


def set_environment(args: argparse.Namespace | ProxyBootstrapConfig) -> None:
    """Set environment variables from CLI arguments.

    Propagates CLI configuration to environment variables so that
    downstream modules (inference engine, API handlers) can access
    settings without direct argument passing. This enables consistent
    configuration across the application.

    The following environment variables are set:
        - NPU_PROXY_DEVICE: Only set if device is not AUTO
        - NPU_PROXY_TOKEN_LIMIT: Always set from token_limit argument
        - NPU_PROXY_COMPILE_CACHE_DIR: Set if compile_cache_dir is provided
        - NPU_PROXY_COMPILE_CACHE_MODE: Set if compile_cache_mode is provided
        - NPU_PROXY_PREFIX_CACHE_MODE: Always set from prefix_cache_mode
        - NPU_PROXY_REAL_INFERENCE: Set to "1" if real_inference is True

    Args:
        args: Parsed CLI arguments from parse_args().

    Example:
        >>> import os
        >>> args = parse_args(['--device', 'NPU', '--real-inference'])
        >>> set_environment(args)
        >>> os.environ.get('NPU_PROXY_DEVICE')
        'NPU'
        >>> os.environ.get('NPU_PROXY_REAL_INFERENCE')
        '1'
    """
    config = args if isinstance(args, ProxyBootstrapConfig) else build_bootstrap_config(args)
    apply_proxy_bootstrap_config_to_env(config)


def run_server(args: argparse.Namespace | ProxyBootstrapConfig) -> None:
    """Start the uvicorn ASGI server with the FastAPI application.

    Launches the NPU Proxy server using uvicorn with configuration derived
    from CLI arguments. Logs startup information including version, bind
    address, device configuration, and inference mode.

    The server runs the FastAPI application defined in npu_proxy.main:app.
    In development mode (--reload), file changes trigger automatic restarts.
    In production mode (--workers > 1), multiple worker processes are spawned.

    Args:
        args: Parsed CLI arguments containing server configuration:
            - host: Bind address
            - port: Port number
            - workers: Number of worker processes
            - reload: Enable auto-reload for development
            - log_level: Uvicorn log verbosity
            - device: Inference device for logging
            - token_limit: Token limit for logging
            - real_inference: Inference mode for logging

    Note:
        This function blocks until the server is shut down (Ctrl+C or SIGTERM).
        Workers > 1 is incompatible with reload=True.

    Example:
        >>> args = parse_args(['--port', '8080', '--device', 'NPU'])
        >>> run_server(args)  # Blocks until shutdown
    """
    import uvicorn

    config = args if isinstance(args, ProxyBootstrapConfig) else build_bootstrap_config(args)

    logger.info(f"Starting NPU Proxy v{get_version()}")
    logger.info(f"Binding to {config.host}:{config.port}")
    logger.info(f"Device: {config.llm.device}, Token limit: {config.token_limit}")
    if config.llm.compile_cache_dir:
        cache_mode = config.llm.compile_cache_mode or "runtime default"
        logger.info(f"Compile cache: {config.llm.compile_cache_dir} ({cache_mode})")
    else:
        logger.info("Compile cache: disabled")
    logger.info(f"Prefix cache mode: {config.llm.prefix_cache_mode}")
    logger.info(
        f"Real inference: {'enabled' if config.real_inference else 'disabled (mock mode)'}"
    )
    if (
        not is_loopback_host(config.host)
        and normalize_allowed_hosts(config.allowed_hosts) == normalize_allowed_hosts(DEFAULT_ALLOWED_HOSTS)
    ):
        logger.warning(
            "Binding to non-loopback host %s without Host allow-list relaxation; "
            "remote clients must use one of %s or set %s/--allowed-hosts.",
            config.host,
            ", ".join(config.allowed_hosts),
            ENV_ALLOWED_HOSTS,
        )

    if config.reload or config.workers > 1:
        app_target = "npu_proxy.main:app"
    else:
        from npu_proxy.main import create_app

        app_target = create_app(config)

    uvicorn.run(
        app_target,
        host=config.host,
        port=config.port,
        workers=config.workers,
        reload=config.reload,
        log_level=config.log_level,
    )


def main(args: list[str] | None = None) -> int:
    """Main entry point for the NPU Proxy CLI.

    Orchestrates the complete startup sequence: argument parsing, logging
    configuration, environment setup, and server launch. Handles graceful
    shutdown on keyboard interrupt and catches fatal errors.

    This function is registered as the console script entry point in
    pyproject.toml and can also be called directly for testing or
    programmatic invocation.

    Args:
        args: Optional list of CLI argument strings. If None, arguments
            are read from sys.argv. Useful for testing or embedding.

    Returns:
        int: Exit code for the process:
            - 0: Successful execution or clean shutdown
            - 1: Fatal error occurred

    Example:
        >>> # Normal invocation (reads sys.argv)
        >>> exit_code = main()

        >>> # Programmatic invocation with custom args
        >>> exit_code = main(['--port', '8080', '--device', 'NPU'])

        >>> # Testing with mock configuration
        >>> exit_code = main(['--log-level', 'debug', '--real-inference'])
    """
    try:
        parsed = parse_args(args)
        config = build_bootstrap_config(parsed)
        setup_logging(config.log_level, config.log_file)

        from npu_proxy.main import bootstrap_runtime

        bootstrap_runtime(config)
        set_environment(config)
        run_server(config)
        return 0
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        return 0
    except Exception as e:
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("Fatal error")
        else:
            logger.error("Fatal error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())

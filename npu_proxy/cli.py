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
    NPU_PROXY_REAL_INFERENCE: Enable real inference when set to "1" (default: 0)

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

# Configure logging before imports to catch early errors
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("npu-proxy")


def get_version() -> str:
    """Get the installed package version.

    Retrieves the version from package metadata using importlib.metadata.
    Falls back to a hardcoded version if metadata is unavailable (e.g.,
    when running from source without installation).

    Returns:
        str: The semantic version string (e.g., "0.1.0").

    Example:
        >>> version = get_version()
        >>> print(f"Running NPU Proxy v{version}")
        Running NPU Proxy v0.1.0
    """
    try:
        from importlib.metadata import version
        return version("npu-proxy")
    except Exception:
        return "0.1.0"


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
  NPU_PROXY_REAL_INFERENCE Enable real inference (1 to enable)
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
        default=os.environ.get("NPU_PROXY_HOST", "127.0.0.1"),
        help="Host to bind to (default: 127.0.0.1)",
    )
    server_group.add_argument(
        "--port", "-p",
        type=int,
        default=int(os.environ.get("NPU_PROXY_PORT", "8080")),
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
    
    # Device options
    device_group = parser.add_argument_group("Device Options")
    device_group.add_argument(
        "--device", "-d",
        choices=["NPU", "GPU", "CPU", "AUTO"],
        default=os.environ.get("NPU_PROXY_DEVICE", "AUTO"),
        help="Inference device (default: AUTO, tries NPU first)",
    )
    device_group.add_argument(
        "--token-limit",
        type=int,
        default=int(os.environ.get("NPU_PROXY_TOKEN_LIMIT", "1800")),
        help="Token limit for NPU context routing (default: 1800)",
    )
    device_group.add_argument(
        "--real-inference",
        action="store_true",
        default=os.environ.get("NPU_PROXY_REAL_INFERENCE", "0") == "1",
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


def set_environment(args: argparse.Namespace) -> None:
    """Set environment variables from CLI arguments.

    Propagates CLI configuration to environment variables so that
    downstream modules (inference engine, API handlers) can access
    settings without direct argument passing. This enables consistent
    configuration across the application.

    The following environment variables are set:
        - NPU_PROXY_DEVICE: Only set if device is not AUTO
        - NPU_PROXY_TOKEN_LIMIT: Always set from token_limit argument
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
    if args.device != "AUTO":
        os.environ["NPU_PROXY_DEVICE"] = args.device
    
    os.environ["NPU_PROXY_TOKEN_LIMIT"] = str(args.token_limit)
    
    if args.real_inference:
        os.environ["NPU_PROXY_REAL_INFERENCE"] = "1"


def run_server(args: argparse.Namespace) -> None:
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
    
    logger.info(f"Starting NPU Proxy v{get_version()}")
    logger.info(f"Binding to {args.host}:{args.port}")
    logger.info(f"Device: {args.device}, Token limit: {args.token_limit}")
    logger.info(f"Real inference: {'enabled' if args.real_inference else 'disabled (mock mode)'}")
    
    uvicorn.run(
        "npu_proxy.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level,
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
        setup_logging(parsed.log_level, parsed.log_file)
        set_environment(parsed)
        run_server(parsed)
        return 0
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

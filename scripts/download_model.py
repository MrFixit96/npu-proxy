#!/usr/bin/env python3
"""
CLI tool for downloading and exporting embedding models to OpenVINO format.

This script provides utilities to:
- Download embedding models from HuggingFace or Ollama
- Export models to OpenVINO format for optimized inference
- List cached models
- Display model metadata and information

Usage:
    python download_model.py download <model_name> [--force]
    python download_model.py list
    python download_model.py info <model_name>
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    from npu_proxy.models.mapper import OLLAMA_TO_HUGGINGFACE
except ImportError:
    # Fallback mapping if import fails
    OLLAMA_TO_HUGGINGFACE = {
        "bge-small": "BAAI/bge-small-en-v1.5",
        "bge-base": "BAAI/bge-base-en-v1.5",
        "bge-large": "BAAI/bge-large-en-v1.5",
        "nomic-embed-text": "nomic-ai/nomic-embed-text-v1",
        "all-minilm": "sentence-transformers/all-MiniLM-L6-v2",
        "all-mpnet": "sentence-transformers/all-mpnet-base-v2",
    }


class ModelManager:
    """Manages downloading and exporting embedding models to OpenVINO format."""

    def __init__(self):
        """Initialize model manager with cache paths."""
        self.cache_dir = Path.home() / ".cache" / "npu-proxy" / "models" / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def resolve_model_name(self, model_name: str) -> str:
        """
        Resolve Ollama-style model names to HuggingFace repository names.

        Args:
            model_name: Model name in Ollama format (e.g., "bge-small") or
                       HuggingFace format (e.g., "BAAI/bge-small-en-v1.5")

        Returns:
            HuggingFace repository name.
        """
        # If it already looks like a HuggingFace repo (contains /), return as-is
        if "/" in model_name:
            return model_name

        # Try to resolve from Ollama mapping
        if model_name in OLLAMA_TO_HUGGINGFACE:
            return OLLAMA_TO_HUGGINGFACE[model_name]

        # If not found, assume it's a HuggingFace model name and return as-is
        return model_name

    def get_safe_model_name(self, model_name: str) -> str:
        """
        Convert model name to safe directory name.

        Args:
            model_name: Original model name.

        Returns:
            Safe directory name with special characters replaced.
        """
        # Replace / with -- for HuggingFace format
        safe_name = model_name.replace("/", "--")
        # Replace other special characters
        safe_name = safe_name.replace(".", "-")
        return safe_name.lower()

    def get_model_cache_path(self, model_name: str) -> Path:
        """
        Get the cache directory path for a model.

        Args:
            model_name: HuggingFace model name.

        Returns:
            Path to the model cache directory.
        """
        safe_name = self.get_safe_model_name(model_name)
        return self.cache_dir / safe_name

    def download_model(self, model_name: str, force: bool = False) -> bool:
        """
        Download and export a model to OpenVINO format.

        Args:
            model_name: Model name in Ollama or HuggingFace format.
            force: Force re-download even if model exists.

        Returns:
            True if successful, False otherwise.
        """
        # Resolve model name
        hf_model = self.resolve_model_name(model_name)
        output_path = self.get_model_cache_path(hf_model)

        # Check if already downloaded
        if output_path.exists() and not force:
            print(f"✓ Model '{model_name}' already cached at {output_path}")
            return True

        if output_path.exists() and force:
            print(f"→ Removing existing cache for '{model_name}'...")
            import shutil
            shutil.rmtree(output_path)

        print(f"→ Downloading and exporting '{model_name}' (HuggingFace: {hf_model})...")
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Build optimum-cli command
            cmd = [
                "optimum-cli",
                "export",
                "openvino",
                "--model_name_or_path",
                hf_model,
                "--task",
                "feature-extraction",
                str(output_path),
            ]

            print(f"→ Running: {' '.join(cmd)}")

            # Run the export command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                print(f"✗ Export failed with return code {result.returncode}")
                print(f"STDERR: {result.stderr}")
                if result.stdout:
                    print(f"STDOUT: {result.stdout}")
                return False

            print(f"✓ Successfully exported '{model_name}' to OpenVINO format")
            print(f"  Location: {output_path}")
            return True

        except FileNotFoundError:
            print("✗ Error: optimum-cli not found. Install with: pip install optimum[openvino]")
            return False
        except Exception as e:
            print(f"✗ Error during export: {e}")
            return False

    def list_models(self) -> None:
        """List all cached embedding models."""
        if not self.cache_dir.exists():
            print("No cached models found.")
            return

        models = list(self.cache_dir.iterdir())

        if not models:
            print("No cached models found.")
            return

        print(f"Cached embedding models in {self.cache_dir}:\n")
        for model_path in sorted(models):
            if model_path.is_dir():
                # Convert safe name back to readable format
                display_name = model_path.name.replace("--", "/").replace("-", ".")
                size = self._get_dir_size(model_path)
                print(f"  • {display_name}")
                print(f"    Path: {model_path}")
                print(f"    Size: {self._format_size(size)}\n")

    def get_model_info(self, model_name: str) -> bool:
        """
        Display information about a cached model.

        Args:
            model_name: Model name in Ollama or HuggingFace format.

        Returns:
            True if model found, False otherwise.
        """
        hf_model = self.resolve_model_name(model_name)
        model_path = self.get_model_cache_path(hf_model)

        if not model_path.exists():
            print(f"✗ Model '{model_name}' not found in cache")
            print(f"  Expected path: {model_path}")
            return False

        print(f"Model Information: {model_name}")
        print(f"HuggingFace Name: {hf_model}")
        print(f"Cache Location: {model_path}")
        print(f"Size: {self._format_size(self._get_dir_size(model_path))}")

        # List files in cache
        files = list(model_path.rglob("*"))
        file_count = sum(1 for f in files if f.is_file())
        print(f"Files: {file_count}")

        # Try to read model config if it exists
        config_path = model_path / "model_config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    if "hidden_size" in config:
                        print(f"Embedding Dimension: {config['hidden_size']}")
                    if "max_position_embeddings" in config:
                        print(f"Max Sequence Length: {config['max_position_embeddings']}")
            except Exception as e:
                print(f"(Could not read model config: {e})")

        return True

    @staticmethod
    def _get_dir_size(path: Path) -> int:
        """Calculate total size of a directory."""
        total = 0
        try:
            for entry in path.rglob("*"):
                if entry.is_file():
                    total += entry.stat().st_size
        except Exception:
            pass
        return total

    @staticmethod
    def _format_size(size: int) -> str:
        """Format byte size to human-readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"


def main():
    """Main entry point for the CLI tool."""
    parser = argparse.ArgumentParser(
        description="Download and export embedding models to OpenVINO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download a model by Ollama name
  python download_model.py download bge-small

  # Download a model by HuggingFace name
  python download_model.py download BAAI/bge-small-en-v1.5

  # Force re-download
  python download_model.py download bge-small --force

  # List all cached models
  python download_model.py list

  # Show info about a model
  python download_model.py info bge-small
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download and export a model")
    download_parser.add_argument("model", help="Model name (Ollama or HuggingFace format)")
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model exists",
    )

    # List command
    subparsers.add_parser("list", help="List cached models")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show info about a model")
    info_parser.add_argument("model", help="Model name (Ollama or HuggingFace format)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    manager = ModelManager()

    try:
        if args.command == "download":
            success = manager.download_model(args.model, force=args.force)
            return 0 if success else 1

        elif args.command == "list":
            manager.list_models()
            return 0

        elif args.command == "info":
            success = manager.get_model_info(args.model)
            return 0 if success else 1

        else:
            print(f"Unknown command: {args.command}")
            return 1

    except KeyboardInterrupt:
        print("\n✗ Operation cancelled by user")
        return 130
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

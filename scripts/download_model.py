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
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    from huggingface_hub import HfApi
except ImportError:  # pragma: no cover - optional until download is invoked
    HfApi = None

try:
    from npu_proxy.models.converter import REQUIRED_OPENVINO_FILES
    from npu_proxy.models.downloader import DOWNLOAD_METADATA_FILE
    from npu_proxy.models.mapper import OLLAMA_TO_HUGGINGFACE, resolve_model_repo
except ImportError:
    REQUIRED_OPENVINO_FILES = ("openvino_model.xml", "openvino_model.bin")
    DOWNLOAD_METADATA_FILE = ".npu_proxy_download.json"
    resolve_model_repo = None

    # Minimal fallback if package APIs are unavailable. Normal execution uses npu_proxy.models.mapper.
    OLLAMA_TO_HUGGINGFACE = {
        "bge-small": "BAAI/bge-small-en-v1.5",
        "bge-base": "BAAI/bge-base-en-v1.5",
        "bge-large": "BAAI/bge-large-en-v1.5",
        "nomic-embed-text": "nomic-ai/nomic-embed-text-v1",
        "all-minilm": "sentence-transformers/all-MiniLM-L6-v2",
        "all-mpnet": "sentence-transformers/all-mpnet-base-v2",
    }


def is_openvino_model(path: Path) -> bool:
    """Return True when the export contains the required OpenVINO IR files."""
    model_path = Path(path)
    return model_path.is_dir() and all(
        (model_path / file_name).exists() for file_name in REQUIRED_OPENVINO_FILES
    )


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
        if resolve_model_repo is not None:
            resolved_repo = resolve_model_repo(model_name)
            if resolved_repo is not None:
                return resolved_repo[0]

        # If it already looks like a HuggingFace repo (contains /), return as-is
        if "/" in model_name:
            return model_name

        # Fallback to the package-derived alias mapping when available.
        if model_name in OLLAMA_TO_HUGGINGFACE:
            resolved = OLLAMA_TO_HUGGINGFACE[model_name]
            if isinstance(resolved, tuple):
                return resolved[0]
            return resolved

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
        try:
            from npu_proxy.inference.embedding_config import (
                is_known_embedding_model,
                resolve_embedding_model_config,
            )

            if is_known_embedding_model(model_name):
                return resolve_embedding_model_config(
                    model_name,
                    model_dir=self.cache_dir.parent,
                ).canonical_path
        except ImportError:
            pass

        safe_name = self.get_safe_model_name(model_name)
        return self.cache_dir / safe_name

    def download_model(
        self,
        model_name: str,
        force: bool = False,
        revision: Optional[str] = None,
        timeout_seconds: int = 3600,
    ) -> bool:
        """
        Download and export a model to OpenVINO format.

        Args:
            model_name: Model name in Ollama or HuggingFace format.
            force: Force re-download even if model exists.
            revision: Optional Hugging Face revision/branch/tag/commit to export.
            timeout_seconds: Maximum time to wait for optimum-cli before cleanup.

        Returns:
            True if successful, False otherwise.
        """
        # Resolve model name
        hf_model = self.resolve_model_name(model_name)
        output_path = self.get_model_cache_path(hf_model)

        # Check if already downloaded
        if output_path.exists() and not force:
            if is_openvino_model(output_path):
                print(f"✓ Model '{model_name}' already cached at {output_path}")
                return True
            missing = ", ".join(self._missing_required_files(output_path))
            print(
                f"→ Re-exporting '{model_name}' because cached files are incomplete: {missing}"
            )
            self._remove_existing_cache(output_path)

        if output_path.exists() and force:
            print(f"→ Removing existing cache for '{model_name}'...")
            self._remove_existing_cache(output_path)

        print(f"→ Downloading and exporting '{model_name}' (HuggingFace: {hf_model})...")
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Build optimum-cli command
            cmd = [
                "optimum-cli",
                "export",
                "openvino",
                "--model",
                hf_model,
                "--task",
                "feature-extraction",
            ]
            if revision:
                cmd.extend(["--revision", revision])
            cmd.append(str(output_path))

            print(f"→ Running: {' '.join(cmd)}")

            # Run the export command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )

            if result.returncode != 0:
                print(f"✗ Export failed with return code {result.returncode}")
                print(f"STDERR: {result.stderr}")
                if result.stdout:
                    print(f"STDOUT: {result.stdout}")
                self._cleanup_failed_export(output_path)
                return False

            if not is_openvino_model(output_path):
                missing = ", ".join(self._missing_required_files(output_path))
                print(
                    "✗ Export completed but missing required OpenVINO files: "
                    f"{missing or 'unknown'}"
                )
                self._cleanup_failed_export(output_path)
                return False

            commit_sha = self._resolve_commit_sha(hf_model, revision)
            self._write_download_metadata(
                output_path,
                source_repo=hf_model,
                requested_model=model_name,
                revision=revision,
                commit_sha=commit_sha,
            )

            print(f"✓ Successfully exported '{model_name}' to OpenVINO format")
            print(f"  Location: {output_path}")
            return True

        except subprocess.TimeoutExpired:
            print(
                f"✗ Export timed out after {timeout_seconds} seconds; removing partial output at {output_path}"
            )
            self._cleanup_failed_export(output_path)
            return False
        except FileNotFoundError:
            print("✗ Error: optimum-cli not found. Install with: pip install optimum[openvino]")
            self._cleanup_failed_export(output_path)
            return False
        except Exception as e:
            print(f"✗ Error during export: {e}")
            self._cleanup_failed_export(output_path)
            return False

    def list_models(self) -> None:
        """List all cached embedding models."""
        if not self.cache_dir.exists():
            print("No cached models found.")
            return

        models = list(self.cache_dir.iterdir())

        valid_models = [model_path for model_path in models if is_openvino_model(model_path)]

        if not valid_models:
            print("No cached models found.")
            return

        print(f"Cached embedding models in {self.cache_dir}:\n")
        for model_path in sorted(valid_models):
            display_name = model_path.name
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
        if not is_openvino_model(model_path):
            missing = ", ".join(self._missing_required_files(model_path))
            print(f"✗ Model '{model_name}' cache is incomplete")
            print(f"  Missing files: {missing}")
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

    @staticmethod
    def _missing_required_files(path: Path) -> list[str]:
        """Return any missing OpenVINO IR files for the cache directory."""
        return [
            file_name
            for file_name in REQUIRED_OPENVINO_FILES
            if not (Path(path) / file_name).exists()
        ]

    @staticmethod
    def _remove_existing_cache(path: Path) -> None:
        """Remove an existing cache directory before re-exporting."""
        shutil.rmtree(path)

    @staticmethod
    def _cleanup_failed_export(path: Path) -> None:
        """Remove partial export output after a failed or timed-out conversion."""
        if path.exists():
            shutil.rmtree(path)

    @staticmethod
    def _resolve_commit_sha(repo_id: str, revision: Optional[str]) -> Optional[str]:
        """Resolve a Hugging Face revision to a commit SHA when metadata is reachable."""
        if HfApi is None:
            print("⚠ huggingface_hub is unavailable; commit SHA was not recorded")
            return None
        try:
            info = HfApi().repo_info(
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
                timeout=10,
            )
        except Exception as exc:
            print(f"⚠ Could not resolve Hugging Face commit SHA: {exc}")
            return None
        sha = getattr(info, "sha", None)
        return sha if isinstance(sha, str) and sha else None

    @staticmethod
    def _write_download_metadata(
        path: Path,
        *,
        source_repo: str,
        requested_model: str,
        revision: Optional[str],
        commit_sha: Optional[str],
    ) -> None:
        """Persist source/revision metadata next to the exported model."""
        metadata = {
            "source_repo": source_repo,
            "requested_model": requested_model,
            "revision": revision or "main",
            "commit_sha": commit_sha,
        }
        (path / DOWNLOAD_METADATA_FILE).write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )


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
    download_parser.add_argument(
        "--revision",
        help="Hugging Face branch, tag, or commit SHA to export",
    )
    download_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=int(os.environ.get("NPU_PROXY_DOWNLOAD_TIMEOUT", "3600")),
        help="Maximum seconds to wait for optimum-cli export (default: 3600)",
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
            success = manager.download_model(
                args.model,
                force=args.force,
                revision=args.revision,
                timeout_seconds=args.timeout_seconds,
            )
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

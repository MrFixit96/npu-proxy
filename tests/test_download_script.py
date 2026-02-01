"""Tests for the download_model.py script."""

import subprocess
import os
import pytest
from pathlib import Path

from npu_proxy.inference.embedding_engine import get_embedding_model_path
from npu_proxy.models.mapper import OLLAMA_TO_HUGGINGFACE


class TestDownloadScript:
    """Tests for the download_model.py script."""

    def test_download_script_help(self):
        """Test that the download script help works correctly.
        
        Runs: python scripts/download_model.py --help
        Verifies:
        - Exit code is 0
        - Output contains "download", "list", and "info"
        """
        # Get the path to the script
        script_path = os.path.join(
            os.path.dirname(__file__), "..", "scripts", "download_model.py"
        )

        # Run the script with --help
        result = subprocess.run(
            ["python", script_path, "--help"],
            capture_output=True,
            text=True,
        )

        # Verify exit code is 0
        assert (
            result.returncode == 0
        ), f"Script failed with code {result.returncode}: {result.stderr}"

        # Verify output contains expected strings
        output = result.stdout + result.stderr
        assert "download" in output.lower(), "Output should contain 'download'"
        assert "list" in output.lower(), "Output should contain 'list'"
        assert "info" in output.lower(), "Output should contain 'info'"

    def test_model_path_generation(self):
        """Test that model paths are generated correctly.
        
        Verifies:
        - Path contains .cache, npu-proxy, and embeddings directories
        - Path ends with model name with slashes replaced by underscores
        """
        model_name = "BAAI/bge-small-en-v1.5"
        path = get_embedding_model_path(model_name)

        # Convert to string if Path object
        path_str = str(path)

        # Verify path contains required components
        assert ".cache" in path_str, "Path should contain .cache directory"
        assert "npu-proxy" in path_str, "Path should contain 'npu-proxy'"
        assert "embeddings" in path_str, "Path should contain 'embeddings'"

        # Verify path ends with model name with slashes replaced
        expected_model_dir = "BAAI_bge-small-en-v1.5"
        assert path_str.endswith(
            expected_model_dir
        ), f"Path should end with {expected_model_dir}, but got {path_str}"

    def test_model_name_resolution(self):
        """Test that model name mappings are valid.
        
        Verifies:
        - bge-small mapping exists and maps to a valid HuggingFace repo
        - all-minilm mapping exists and maps to a valid HuggingFace repo
        """
        # Verify the mapper has the expected keys
        assert (
            "bge-small" in OLLAMA_TO_HUGGINGFACE
        ), "OLLAMA_TO_HUGGINGFACE should contain 'bge-small' mapping"
        assert (
            "all-minilm" in OLLAMA_TO_HUGGINGFACE
        ), "OLLAMA_TO_HUGGINGFACE should contain 'all-minilm' mapping"

        # Verify the mapped values are valid - mapping returns (repo, type) tuple
        bge_small_mapping = OLLAMA_TO_HUGGINGFACE["bge-small"]
        all_minilm_mapping = OLLAMA_TO_HUGGINGFACE["all-minilm"]

        # Extract repo name (first element of tuple)
        bge_small_repo = bge_small_mapping[0] if isinstance(bge_small_mapping, tuple) else bge_small_mapping
        all_minilm_repo = all_minilm_mapping[0] if isinstance(all_minilm_mapping, tuple) else all_minilm_mapping

        # They should be non-empty strings
        assert (
            isinstance(bge_small_repo, str) and len(bge_small_repo) > 0
        ), "bge-small mapping should be a non-empty string"
        assert (
            isinstance(all_minilm_repo, str) and len(all_minilm_repo) > 0
        ), "all-minilm mapping should be a non-empty string"

        # They should look like HuggingFace repo names (contain forward slash)
        assert (
            "/" in bge_small_repo
        ), f"bge-small mapping '{bge_small_repo}' should be a HuggingFace repo name (org/model)"
        assert (
            "/" in all_minilm_repo
        ), f"all-minilm mapping '{all_minilm_repo}' should be a HuggingFace repo name (org/model)"

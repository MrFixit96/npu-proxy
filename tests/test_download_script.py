"""Tests for the download_model.py script."""

import subprocess
import os
from unittest.mock import MagicMock, patch
from pathlib import Path

from scripts.download_model import ModelManager
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

    def test_model_path_generation(self, tmp_path):
        """Known embedding models should use the runtime's canonical cache path."""
        manager = ModelManager()
        manager.cache_dir = tmp_path / "models" / "embeddings"

        path = manager.get_model_cache_path("BAAI/bge-small-en-v1.5")

        assert path == manager.cache_dir / "bge-small"

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

    def test_download_manager_unwraps_tuple_alias_mapping(self):
        """Tuple-based alias mappings should resolve to the repo ID string."""
        manager = ModelManager()

        assert manager.resolve_model_name("all-minilm") == "sentence-transformers/all-MiniLM-L6-v2"

    def test_download_manager_uses_runtime_embedding_cache_path(self):
        """Known embedding aliases should export into the runtime's canonical path."""
        manager = ModelManager()

        path = manager.get_model_cache_path("sentence-transformers/all-MiniLM-L6-v2")

        assert path == Path.home() / ".cache" / "npu-proxy" / "models" / "embeddings" / "all-minilm-l6-v2"

    def test_list_models_preserves_canonical_directory_names(self, tmp_path, capsys):
        """The list command should print canonical cache directory names verbatim."""
        manager = ModelManager()
        manager.cache_dir = tmp_path / "embeddings"
        model_dir = manager.cache_dir / "all-minilm-l6-v2"
        model_dir.mkdir(parents=True)
        (model_dir / "openvino_model.xml").touch()
        (model_dir / "openvino_model.bin").touch()

        manager.list_models()
        output = capsys.readouterr().out

        assert "all-minilm-l6-v2" in output
        assert "all.minilm.l6.v2" not in output

    @patch("scripts.download_model.subprocess.run")
    def test_download_manager_uses_optimum_model_flag(self, mock_run, tmp_path):
        """Exports should use the Optimum CLI's documented --model flag."""
        manager = ModelManager()
        manager.cache_dir = tmp_path / "embeddings"
        output_dir = manager.cache_dir / "all-minilm-l6-v2"

        def fake_run(cmd, **kwargs):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "openvino_model.xml").touch()
            (output_dir / "openvino_model.bin").touch()
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = fake_run

        assert manager.download_model("all-minilm") is True

        command = mock_run.call_args.args[0]
        assert "--model" in command
        assert "--model_name_or_path" not in command

    @patch("scripts.download_model.subprocess.run")
    def test_download_manager_rejects_incomplete_export_output(self, mock_run, tmp_path, capsys):
        """Exports should fail closed when either required OpenVINO file is missing."""
        manager = ModelManager()
        manager.cache_dir = tmp_path / "embeddings"
        output_dir = manager.cache_dir / "all-minilm-l6-v2"

        def fake_run(cmd, **kwargs):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "openvino_model.xml").touch()
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = fake_run

        assert manager.download_model("all-minilm") is False

        output = capsys.readouterr().out
        assert "missing required openvino files" in output.lower()

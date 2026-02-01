"""Tests for the converter module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from io import StringIO

from npu_proxy.models.converter import (
    is_openvino_model,
    convert_to_openvino,
    auto_download_and_convert,
    get_conversion_progress,
)


class TestIsOpenVINOModel:
    """Tests for is_openvino_model function."""

    def test_is_openvino_model_with_valid_model(self, tmp_path: Path):
        """Valid OpenVINO model returns True."""
        model_dir = tmp_path / "valid_model"
        model_dir.mkdir()
        (model_dir / "openvino_model.xml").touch()
        (model_dir / "openvino_model.bin").touch()

        assert is_openvino_model(model_dir) is True

    def test_is_openvino_model_missing_xml(self, tmp_path: Path):
        """Model missing .xml file returns False."""
        model_dir = tmp_path / "invalid_model"
        model_dir.mkdir()
        (model_dir / "openvino_model.bin").touch()

        assert is_openvino_model(model_dir) is False

    def test_is_openvino_model_missing_bin(self, tmp_path: Path):
        """Model missing .bin file returns False."""
        model_dir = tmp_path / "invalid_model"
        model_dir.mkdir()
        (model_dir / "openvino_model.xml").touch()

        assert is_openvino_model(model_dir) is False

    def test_is_openvino_model_nonexistent_directory(self, tmp_path: Path):
        """Non-existent directory returns False."""
        nonexistent = tmp_path / "does_not_exist"
        assert is_openvino_model(nonexistent) is False

    def test_is_openvino_model_with_string_path(self, tmp_path: Path):
        """Function accepts string paths."""
        model_dir = tmp_path / "valid_model"
        model_dir.mkdir()
        (model_dir / "openvino_model.xml").touch()
        (model_dir / "openvino_model.bin").touch()

        assert is_openvino_model(str(model_dir)) is True

    def test_is_openvino_model_with_file_path(self, tmp_path: Path):
        """File path (not directory) returns False."""
        file_path = tmp_path / "some_file.txt"
        file_path.touch()

        assert is_openvino_model(file_path) is False


class TestConvertToOpenVINO:
    """Tests for convert_to_openvino function."""

    def test_convert_to_openvino_invalid_task(self, tmp_path: Path):
        """Invalid task type returns error."""
        result = convert_to_openvino(
            "gpt2",
            tmp_path,
            task="invalid-task",
        )

        assert "error" in result
        assert "invalid" in result["error"].lower()

    @patch("npu_proxy.models.converter.subprocess.run")
    def test_convert_to_openvino_optimum_not_found(self, mock_run, tmp_path: Path):
        """Missing optimum-cli returns helpful error."""
        mock_run.side_effect = FileNotFoundError()

        result = convert_to_openvino(
            "gpt2",
            tmp_path,
        )

        assert "error" in result
        assert "optimum-cli not found" in result["error"]
        assert "pip install optimum-intel" in result["error"]

    @patch("npu_proxy.models.converter.subprocess.run")
    def test_convert_to_openvino_conversion_failed(self, mock_run, tmp_path: Path):
        """Failed subprocess returns error."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="Model not found on HuggingFace",
            stdout="",
        )

        result = convert_to_openvino(
            "nonexistent-model",
            tmp_path,
        )

        assert "error" in result
        assert "Model not found on HuggingFace" in result["error"]

    @patch("npu_proxy.models.converter.subprocess.run")
    def test_convert_to_openvino_success(self, mock_run, tmp_path: Path):
        """Successful conversion returns success dict."""
        # Create the expected output files
        (tmp_path / "openvino_model.xml").touch()
        (tmp_path / "openvino_model.bin").touch()

        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

        result = convert_to_openvino("gpt2", tmp_path)

        assert result["status"] == "success"
        assert str(tmp_path) in result["path"]
        assert "gpt2" in result["model"]

    @patch("npu_proxy.models.converter.subprocess.run")
    def test_convert_to_openvino_success_dict_keys(self, mock_run, tmp_path: Path):
        """Success dict has required keys."""
        (tmp_path / "openvino_model.xml").touch()
        (tmp_path / "openvino_model.bin").touch()

        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

        result = convert_to_openvino("gpt2", tmp_path)

        assert "status" in result
        assert "path" in result
        assert "model" in result

    @patch("npu_proxy.models.converter.subprocess.run")
    def test_convert_to_openvino_progress_callback(self, mock_run, tmp_path: Path):
        """Progress callback is called."""
        (tmp_path / "openvino_model.xml").touch()
        (tmp_path / "openvino_model.bin").touch()

        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

        callback = MagicMock()
        result = convert_to_openvino("gpt2", tmp_path, progress_callback=callback)

        assert result["status"] == "success"
        assert callback.call_count > 0

    @patch("npu_proxy.models.converter.subprocess.run")
    def test_convert_to_openvino_timeout(self, mock_run, tmp_path: Path):
        """Timeout error is handled."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 3600)

        result = convert_to_openvino("gpt2", tmp_path)

        assert "error" in result
        assert "timed out" in result["error"].lower()

    @patch("npu_proxy.models.converter.subprocess.run")
    def test_convert_to_openvino_missing_output_files(self, mock_run, tmp_path: Path):
        """Subprocess succeeds but output files missing returns error."""
        # Don't create the expected output files
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

        result = convert_to_openvino("gpt2", tmp_path)

        assert "error" in result
        assert "output files not found" in result["error"]

    @patch("npu_proxy.models.converter.subprocess.run")
    def test_convert_to_openvino_with_string_output_dir(self, mock_run, tmp_path: Path):
        """Function accepts string output_dir."""
        (tmp_path / "openvino_model.xml").touch()
        (tmp_path / "openvino_model.bin").touch()

        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

        result = convert_to_openvino("gpt2", str(tmp_path))

        assert result["status"] == "success"

    @patch("npu_proxy.models.converter.subprocess.run")
    def test_convert_to_openvino_feature_extraction_task(self, mock_run, tmp_path: Path):
        """Feature extraction task is supported."""
        (tmp_path / "openvino_model.xml").touch()
        (tmp_path / "openvino_model.bin").touch()

        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

        result = convert_to_openvino(
            "sentence-transformers/all-MiniLM-L6-v2",
            tmp_path,
            task="feature-extraction",
        )

        assert result["status"] == "success"
        # Verify the correct task was passed to subprocess
        call_args = mock_run.call_args
        assert "--task" in call_args[0][0]
        assert "feature-extraction" in call_args[0][0]


class TestAutoDownloadAndConvert:
    """Tests for auto_download_and_convert function."""

    @patch("npu_proxy.models.converter.resolve_model_repo")
    def test_auto_download_unknown_model(self, mock_resolve):
        """Unknown model returns error."""
        mock_resolve.return_value = None

        result = auto_download_and_convert("unknown-model")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @patch("npu_proxy.models.converter.resolve_model_repo")
    def test_auto_download_skips_if_exists(self, mock_resolve, tmp_path: Path):
        """Skip conversion if OpenVINO model already in cache."""
        model_dir = tmp_path / "tinyllama"
        model_dir.mkdir(parents=True)
        (model_dir / "openvino_model.xml").touch()
        (model_dir / "openvino_model.bin").touch()

        mock_resolve.return_value = ("TinyLlama/TinyLlama-1.1B", "tinyllama")

        result = auto_download_and_convert("tinyllama", cache_dir=tmp_path)

        assert result["status"] == "success"
        assert result["source"] == "cache"
        assert str(model_dir) in result["path"]

    @patch("npu_proxy.models.converter.resolve_model_repo")
    @patch("npu_proxy.models.converter.convert_to_openvino")
    def test_auto_download_converts_if_not_exists(
        self, mock_convert, mock_resolve, tmp_path: Path
    ):
        """Convert model if not in cache."""
        model_dir = tmp_path / "tinyllama"

        mock_resolve.return_value = ("TinyLlama/TinyLlama-1.1B", "tinyllama")
        mock_convert.return_value = {
            "status": "success",
            "path": str(model_dir),
            "model": "TinyLlama/TinyLlama-1.1B",
        }

        result = auto_download_and_convert("tinyllama", cache_dir=tmp_path)

        assert result["status"] == "success"
        assert result["source"] == "converted"
        mock_convert.assert_called_once()

    @patch("npu_proxy.models.converter.resolve_model_repo")
    @patch("npu_proxy.models.converter.convert_to_openvino")
    def test_auto_download_conversion_error(self, mock_convert, mock_resolve, tmp_path: Path):
        """Conversion error is propagated."""
        mock_resolve.return_value = ("TinyLlama/TinyLlama-1.1B", "tinyllama")
        mock_convert.return_value = {"error": "Conversion failed"}

        result = auto_download_and_convert("tinyllama", cache_dir=tmp_path)

        assert "error" in result

    @patch("npu_proxy.models.converter.resolve_model_repo")
    def test_auto_download_with_string_cache_dir(self, mock_resolve, tmp_path: Path):
        """Function accepts string cache_dir."""
        model_dir = tmp_path / "tinyllama"
        model_dir.mkdir(parents=True)
        (model_dir / "openvino_model.xml").touch()
        (model_dir / "openvino_model.bin").touch()

        mock_resolve.return_value = ("TinyLlama/TinyLlama-1.1B", "tinyllama")

        result = auto_download_and_convert("tinyllama", cache_dir=str(tmp_path))

        assert result["status"] == "success"

    @patch("npu_proxy.models.converter.resolve_model_repo")
    @patch("npu_proxy.models.converter.convert_to_openvino")
    def test_auto_download_with_feature_extraction_task(
        self, mock_convert, mock_resolve, tmp_path: Path
    ):
        """Feature extraction task is passed through."""
        model_dir = tmp_path / "embedding-model"

        mock_resolve.return_value = ("sentence-transformers/all-MiniLM-L6-v2", "embedding-model")
        mock_convert.return_value = {
            "status": "success",
            "path": str(model_dir),
            "model": "sentence-transformers/all-MiniLM-L6-v2",
        }

        result = auto_download_and_convert(
            "embedding-model",
            task="feature-extraction",
            cache_dir=tmp_path,
        )

        assert result["status"] == "success"
        # Verify task was passed to convert_to_openvino
        # call_args[0] is positional args, call_args[1] is keyword args
        call_args, call_kwargs = mock_convert.call_args
        # The call is: convert_to_openvino(hf_repo, output_dir, task)
        # task is the 3rd positional argument
        assert call_args[2] == "feature-extraction"


class TestGetConversionProgress:
    """Tests for get_conversion_progress generator."""

    def test_get_conversion_progress_invalid_task(self, tmp_path: Path):
        """Invalid task yields error."""
        gen = get_conversion_progress(
            "gpt2",
            tmp_path,
            task="invalid-task",
        )

        progress = list(gen)
        assert len(progress) > 0
        assert progress[0]["status"] == "error"
        assert "invalid" in progress[0]["message"].lower()

    @patch("npu_proxy.models.converter.subprocess.Popen")
    def test_get_conversion_progress_optimum_not_found(self, mock_popen, tmp_path: Path):
        """Missing optimum-cli yields error."""
        mock_popen.side_effect = FileNotFoundError()

        gen = get_conversion_progress("gpt2", tmp_path)
        progress = list(gen)

        assert len(progress) > 0
        error_messages = [p["message"] for p in progress if p["status"] == "error"]
        assert any("optimum-cli not found" in msg for msg in error_messages)

    @patch("npu_proxy.models.converter.subprocess.Popen")
    def test_get_conversion_progress_yields_updates(self, mock_popen, tmp_path: Path):
        """Progress updates are yielded."""
        # Create expected output files
        (tmp_path / "openvino_model.xml").touch()
        (tmp_path / "openvino_model.bin").touch()

        # Mock subprocess output
        mock_process = MagicMock()
        mock_process.stdout = StringIO("Converting model\nProgress: 50%\n")
        mock_process.wait.return_value = 0

        mock_popen.return_value = mock_process

        gen = get_conversion_progress("gpt2", tmp_path)
        progress = list(gen)

        # Check for expected statuses
        statuses = [p["status"] for p in progress]
        assert "starting" in statuses
        assert "running" in statuses
        assert "success" in statuses

    @patch("npu_proxy.models.converter.subprocess.Popen")
    def test_get_conversion_progress_nonzero_exit(self, mock_popen, tmp_path: Path):
        """Non-zero exit code yields error."""
        mock_process = MagicMock()
        mock_process.stdout = StringIO("")
        mock_process.wait.return_value = 1

        mock_popen.return_value = mock_process

        gen = get_conversion_progress("gpt2", tmp_path)
        progress = list(gen)

        error_statuses = [p for p in progress if p["status"] == "error"]
        assert len(error_statuses) > 0

    @patch("npu_proxy.models.converter.subprocess.Popen")
    def test_get_conversion_progress_timeout(self, mock_popen, tmp_path: Path):
        """Timeout is caught and yielded as error."""
        import subprocess

        mock_process = MagicMock()
        mock_process.wait.side_effect = subprocess.TimeoutExpired("cmd", 3600)

        mock_popen.return_value = mock_process

        gen = get_conversion_progress("gpt2", tmp_path)
        progress = list(gen)

        error_statuses = [p for p in progress if p["status"] == "error"]
        assert len(error_statuses) > 0
        assert any("timed out" in p["message"].lower() for p in error_statuses)

    @patch("npu_proxy.models.converter.subprocess.Popen")
    def test_get_conversion_progress_with_string_output_dir(self, mock_popen, tmp_path: Path):
        """Function accepts string output_dir."""
        (tmp_path / "openvino_model.xml").touch()
        (tmp_path / "openvino_model.bin").touch()

        mock_process = MagicMock()
        mock_process.stdout = StringIO("")
        mock_process.wait.return_value = 0

        mock_popen.return_value = mock_process

        gen = get_conversion_progress("gpt2", str(tmp_path))
        progress = list(gen)

        assert len(progress) > 0
        assert "success" in [p["status"] for p in progress]


class TestConversionIntegration:
    """Integration tests combining multiple functions."""

    @patch("npu_proxy.models.converter.resolve_model_repo")
    def test_full_conversion_flow_cached(self, mock_resolve, tmp_path: Path):
        """Full flow when model is cached."""
        # Setup cache
        model_dir = tmp_path / "gpt2"
        model_dir.mkdir(parents=True)
        (model_dir / "openvino_model.xml").touch()
        (model_dir / "openvino_model.bin").touch()

        mock_resolve.return_value = ("openai-community/gpt2", "gpt2")

        # First check
        assert is_openvino_model(model_dir) is True

        # Auto download should skip conversion
        result = auto_download_and_convert("gpt2", cache_dir=tmp_path)

        assert result["status"] == "success"
        assert result["source"] == "cache"

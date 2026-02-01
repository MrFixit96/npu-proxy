"""Tests for the downloader module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from npu_proxy.models.downloader import (
    download_model,
    get_download_progress,
    is_model_downloaded,
    get_downloaded_models,
)


class TestDownloadModel:
    """Tests for download_model function."""

    def test_download_model_unknown_returns_error(self, tmp_path: Path):
        """Unknown model should return error dict."""
        result = download_model("nonexistent-model-xyz", tmp_path)

        assert "error" in result
        assert isinstance(result["error"], str)
        assert "not found" in result["error"].lower()

    @patch("npu_proxy.models.downloader.snapshot_download")
    def test_download_model_success(self, mock_snapshot, tmp_path: Path):
        """Successful download returns success dict."""
        model_dir = tmp_path / "tinyllama"
        model_dir.mkdir(parents=True)
        (model_dir / "openvino_model.xml").touch()
        mock_snapshot.return_value = str(model_dir)

        result = download_model("tinyllama", tmp_path)

        assert result.get("status") == "success"
        mock_snapshot.assert_called_once()

    @patch("npu_proxy.models.downloader.snapshot_download")
    def test_download_success_dict_keys(self, mock_snapshot, tmp_path: Path):
        """Success dict has status, model, path, source keys."""
        model_dir = tmp_path / "tinyllama"
        model_dir.mkdir(parents=True)
        (model_dir / "openvino_model.xml").touch()
        mock_snapshot.return_value = str(model_dir)

        result = download_model("tinyllama", tmp_path)

        assert "status" in result
        assert "model" in result
        assert "path" in result
        assert "source" in result


class TestIsModelDownloaded:
    """Tests for is_model_downloaded function."""

    def test_is_model_downloaded_false(self, tmp_path: Path):
        """Missing model returns False."""
        result = is_model_downloaded("nonexistent-model", tmp_path)

        assert result is False

    def test_is_model_downloaded_true(self, tmp_path: Path):
        """Model with openvino_model.xml returns True."""
        model_dir = tmp_path / "tinyllama"
        model_dir.mkdir(parents=True)
        (model_dir / "openvino_model.xml").touch()

        result = is_model_downloaded("tinyllama", tmp_path)

        assert result is True


class TestGetDownloadedModels:
    """Tests for get_downloaded_models function."""

    def test_get_downloaded_models_list(self, tmp_path: Path):
        """Returns list of model names."""
        model1 = tmp_path / "model-one"
        model1.mkdir()
        (model1 / "openvino_model.xml").touch()

        model2 = tmp_path / "model-two"
        model2.mkdir()
        (model2 / "openvino_model.xml").touch()

        # Invalid directory (no openvino_model.xml)
        invalid = tmp_path / "not-a-model"
        invalid.mkdir()
        (invalid / "other_file.txt").touch()

        result = get_downloaded_models(tmp_path)

        assert isinstance(result, list)
        assert len(result) == 2
        assert "model-one" in result
        assert "model-two" in result
        assert "not-a-model" not in result


class TestGetDownloadProgress:
    """Tests for get_download_progress function."""

    @patch("npu_proxy.models.downloader.HfApi")
    @patch("npu_proxy.models.downloader.hf_hub_download")
    def test_download_progress_yields_dicts(
        self, mock_download, mock_api_class, tmp_path: Path
    ):
        """Generator yields status dicts."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.repo_info.return_value = MagicMock(siblings=[])

        progress_gen = get_download_progress("test-repo/model", tmp_path)
        statuses = list(progress_gen)

        assert len(statuses) >= 1
        assert all(isinstance(s, dict) for s in statuses)
        assert statuses[0].get("status") == "pulling manifest"

    @patch("npu_proxy.models.downloader.HfApi")
    @patch("npu_proxy.models.downloader.hf_hub_download")
    def test_download_progress_success(
        self, mock_download, mock_api_class, tmp_path: Path
    ):
        """Includes 'success' status when model xml exists."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.repo_info.return_value = MagicMock(siblings=[])

        # Create the xml file so verification succeeds
        (tmp_path / "openvino_model.xml").touch()

        progress_gen = get_download_progress("test-repo/model", tmp_path)
        statuses = list(progress_gen)

        status_values = [s.get("status") for s in statuses]
        assert "success" in status_values

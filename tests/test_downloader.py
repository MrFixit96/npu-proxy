"""Tests for the downloader module."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from npu_proxy.models.downloader import (
    DOWNLOAD_METADATA_FILE,
    download_model,
    get_download_progress,
    is_model_downloaded,
    get_downloaded_models,
    resolve_download_target,
    _validate_download_manifest,
)


class TestDownloadManifest:
    def test_full_snapshot_allows_nested_safe_paths(self):
        repo_info = MagicMock(
            siblings=[
                MagicMock(rfilename="openvino_model.xml", size=1),
                MagicMock(rfilename="nested/config.json", size=1),
            ]
        )

        assert _validate_download_manifest(
            repo_info, ("openvino_model.xml",), full_snapshot=True
        ) is None


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
        model_dir = tmp_path / "tinyllama-1.1b-chat-int4-ov"
        model_dir.mkdir(parents=True)
        (model_dir / "openvino_model.xml").write_text("<model/>", encoding="utf-8")
        (model_dir / "openvino_model.bin").write_bytes(b"weights")
        mock_snapshot.return_value = str(model_dir)

        result = download_model("tinyllama", tmp_path)

        assert result.get("status") == "success"
        mock_snapshot.assert_called_once()

    @patch("npu_proxy.models.downloader.HfApi")
    @patch("npu_proxy.models.downloader.snapshot_download")
    def test_download_model_disables_ambient_token_by_default(
        self,
        mock_snapshot,
        mock_api_class,
        tmp_path: Path,
    ):
        """Anonymous pulls must disable ambient Hugging Face credentials."""
        model_dir = tmp_path / "tinyllama-1.1b-chat-int4-ov"
        model_dir.mkdir(parents=True)
        (model_dir / "openvino_model.xml").write_text("<model/>", encoding="utf-8")
        (model_dir / "openvino_model.bin").write_bytes(b"weights")
        mock_snapshot.return_value = str(model_dir)
        mock_api = MagicMock()
        mock_api.repo_info.return_value = MagicMock(private=False)
        mock_api_class.return_value = mock_api

        result = download_model("tinyllama", tmp_path)

        assert result.get("status") == "success"
        mock_api_class.assert_called_once_with(token=False)
        assert mock_snapshot.call_args.kwargs["token"] is False
        metadata = json.loads((model_dir / DOWNLOAD_METADATA_FILE).read_text(encoding="utf-8"))
        assert metadata["requires_token"] is False

    @patch("npu_proxy.models.downloader.HfApi")
    @patch("npu_proxy.models.downloader.snapshot_download")
    def test_download_model_uses_explicit_token_for_private_pull(
        self,
        mock_snapshot,
        mock_api_class,
        tmp_path: Path,
    ):
        """Private pulls should only use the caller-provided token."""
        model_dir = tmp_path / "tinyllama-1.1b-chat-int4-ov"
        model_dir.mkdir(parents=True)
        (model_dir / "openvino_model.xml").write_text("<model/>", encoding="utf-8")
        (model_dir / "openvino_model.bin").write_bytes(b"weights")
        mock_snapshot.return_value = str(model_dir)
        mock_api = MagicMock()
        mock_api.repo_info.return_value = MagicMock(private=True)
        mock_api_class.return_value = mock_api

        result = download_model("tinyllama", tmp_path, token="hf_secret")

        assert result.get("status") == "success"
        mock_api_class.assert_called_once_with(token="hf_secret")
        assert mock_snapshot.call_args.kwargs["token"] == "hf_secret"
        metadata = json.loads((model_dir / DOWNLOAD_METADATA_FILE).read_text(encoding="utf-8"))
        assert metadata["requires_token"] is True
        assert metadata["private"] is True

    @patch("npu_proxy.models.downloader.HfApi")
    @patch("npu_proxy.models.downloader.snapshot_download")
    def test_download_model_public_repo_with_token_does_not_gate_cache(
        self,
        mock_snapshot,
        mock_api_class,
        tmp_path: Path,
    ):
        """Public repos pulled with a token should remain anonymously reusable."""
        model_dir = tmp_path / "tinyllama-1.1b-chat-int4-ov"
        model_dir.mkdir(parents=True)
        (model_dir / "openvino_model.xml").write_text("<model/>", encoding="utf-8")
        (model_dir / "openvino_model.bin").write_bytes(b"weights")
        mock_snapshot.return_value = str(model_dir)
        mock_api = MagicMock()
        mock_api.repo_info.return_value = MagicMock(private=False)
        mock_api_class.return_value = mock_api

        result = download_model("tinyllama", tmp_path, token="hf_optional_token")

        assert result.get("status") == "success"
        metadata = json.loads((model_dir / DOWNLOAD_METADATA_FILE).read_text(encoding="utf-8"))
        assert metadata["requires_token"] is False
        assert metadata["private"] is False

    @patch("npu_proxy.models.downloader.HfApi")
    @patch("npu_proxy.models.downloader.snapshot_download")
    def test_download_model_gated_repo_marks_cache_token_gated(
        self,
        mock_snapshot,
        mock_api_class,
        tmp_path: Path,
    ):
        """Gated repos should remain token-gated even when not marked private."""
        model_dir = tmp_path / "tinyllama-1.1b-chat-int4-ov"
        model_dir.mkdir(parents=True)
        (model_dir / "openvino_model.xml").write_text("<model/>", encoding="utf-8")
        (model_dir / "openvino_model.bin").write_bytes(b"weights")
        mock_snapshot.return_value = str(model_dir)
        mock_api = MagicMock()
        mock_api.repo_info.return_value = MagicMock(private=False, gated=True)
        mock_api_class.return_value = mock_api

        result = download_model("tinyllama", tmp_path, token="hf_secret")

        assert result.get("status") == "success"
        metadata = json.loads((model_dir / DOWNLOAD_METADATA_FILE).read_text(encoding="utf-8"))
        assert metadata["requires_token"] is True
        assert metadata["private"] is False

    @patch("npu_proxy.models.downloader.snapshot_download")
    def test_download_success_dict_keys(self, mock_snapshot, tmp_path: Path):
        """Success dict has status, model, path, source keys."""
        model_dir = tmp_path / "tinyllama-1.1b-chat-int4-ov"
        model_dir.mkdir(parents=True)
        (model_dir / "openvino_model.xml").write_text("<model/>", encoding="utf-8")
        (model_dir / "openvino_model.bin").write_bytes(b"weights")
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
        """Trusted catalog model with matching metadata returns True."""
        model_dir = tmp_path / "tinyllama-1.1b-chat-int4-ov"
        model_dir.mkdir(parents=True)
        (model_dir / "openvino_model.xml").write_text("<model/>", encoding="utf-8")
        (model_dir / "openvino_model.bin").write_bytes(b"weights")
        (model_dir / DOWNLOAD_METADATA_FILE).write_text(
            json.dumps(
                {
                    "source_repo": "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov",
                    "requires_token": False,
                }
            ),
            encoding="utf-8",
        )

        result = is_model_downloaded("tinyllama", tmp_path)

        assert result is True

    def test_is_model_downloaded_catalog_cache_without_metadata_fails_closed(self, tmp_path: Path):
        """Catalog cache dirs without provenance metadata should not be trusted."""
        model_dir = tmp_path / "tinyllama-1.1b-chat-int4-ov"
        model_dir.mkdir(parents=True)
        (model_dir / "openvino_model.xml").write_text("<model/>", encoding="utf-8")
        (model_dir / "openvino_model.bin").write_bytes(b"weights")

        assert is_model_downloaded("tinyllama", tmp_path) is False

    def test_is_model_downloaded_requires_token_for_token_gated_cache(self, tmp_path: Path):
        """Token-gated cached models should not appear downloaded anonymously."""
        model_dir = tmp_path / "tinyllama-1.1b-chat-int4-ov"
        model_dir.mkdir(parents=True)
        (model_dir / "openvino_model.xml").write_text("<model/>", encoding="utf-8")
        (model_dir / "openvino_model.bin").write_bytes(b"weights")
        (model_dir / DOWNLOAD_METADATA_FILE).write_text(
            json.dumps(
                {
                    "source_repo": "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov",
                    "requires_token": True,
                }
            ),
            encoding="utf-8",
        )

        assert is_model_downloaded("tinyllama", tmp_path) is False

    def test_is_model_downloaded_direct_repo_without_metadata_fails_closed(self, tmp_path: Path):
        """Direct repo cache entries without metadata should not be anonymously trusted."""
        model_dir = tmp_path / "phi-2-int4-ov"
        model_dir.mkdir(parents=True)
        (model_dir / "openvino_model.xml").write_text("<model/>", encoding="utf-8")
        (model_dir / "openvino_model.bin").write_bytes(b"weights")

        assert is_model_downloaded("OpenVINO/phi-2-int4-ov", tmp_path) is False

    @patch("npu_proxy.models.downloader.HfApi")
    def test_is_model_downloaded_validates_repo_access_for_token_gated_cache(
        self,
        mock_api_class,
        tmp_path: Path,
    ):
        """A token-gated cache entry only counts when the supplied token can still access the repo."""
        model_dir = tmp_path / "tinyllama-1.1b-chat-int4-ov"
        model_dir.mkdir(parents=True)
        (model_dir / "openvino_model.xml").write_text("<model/>", encoding="utf-8")
        (model_dir / "openvino_model.bin").write_bytes(b"weights")
        (model_dir / DOWNLOAD_METADATA_FILE).write_text(
            json.dumps(
                {
                    "source_repo": "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov",
                    "requires_token": True,
                }
            ),
            encoding="utf-8",
        )
        mock_api = MagicMock()
        mock_api.repo_info.return_value = MagicMock()
        mock_api_class.return_value = mock_api

        assert is_model_downloaded("tinyllama", tmp_path, token="hf_secret") is True
        mock_api_class.assert_called_once_with(token="hf_secret")

    def test_catalog_alias_registry_and_repo_share_download_target(self, tmp_path: Path):
        """Trusted aliases and repo IDs should land in one canonical cache directory."""
        alias_target = resolve_download_target("tinyllama", tmp_path)
        registry_target = resolve_download_target("tinyllama-1.1b-chat-int4-ov", tmp_path)
        repo_target = resolve_download_target("OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov", tmp_path)

        assert alias_target is not None
        assert registry_target is not None
        assert repo_target is not None
        assert alias_target[2] == registry_target[2] == repo_target[2]
        assert alias_target[2].name == "tinyllama-1.1b-chat-int4-ov"

    def test_unknown_repo_basename_cannot_collide_with_trusted_cache_name(self, tmp_path: Path):
        """Uncatalogued repos should use a fully-qualified storage key."""
        trusted_target = resolve_download_target("tinyllama", tmp_path)
        attacker_target = resolve_download_target("attacker/tinyllama", tmp_path)

        assert trusted_target is not None
        assert attacker_target is not None
        assert trusted_target[2] != attacker_target[2]
        assert attacker_target[2].name == "attacker%2Ftinyllama"

    def test_secondary_alias_shares_download_target_with_primary_alias(self, tmp_path: Path):
        """aliases[] names should reuse the primary alias repo and cache path."""
        primary_target = resolve_download_target("tinyllama:fp16", tmp_path)
        secondary_target = resolve_download_target("tinyllama-fp16", tmp_path)

        assert primary_target is not None
        assert secondary_target is not None
        assert primary_target[0] == secondary_target[0] == "OpenVINO/TinyLlama-1.1B-Chat-v1.0-fp16-ov"
        assert primary_target[2] == secondary_target[2]

    def test_is_model_downloaded_accepts_secondary_alias_for_primary_cache(self, tmp_path: Path):
        """Secondary aliases should trust the same canonical downloaded cache entry."""
        target = resolve_download_target("tinyllama:fp16", tmp_path)
        assert target is not None

        repo_id, _, local_dir, _ = target
        local_dir.mkdir(parents=True)
        (local_dir / "openvino_model.xml").write_text("<model/>", encoding="utf-8")
        (local_dir / "openvino_model.bin").write_bytes(b"weights")
        (local_dir / DOWNLOAD_METADATA_FILE).write_text(
            json.dumps({"source_repo": repo_id, "requires_token": False}),
            encoding="utf-8",
        )

        assert is_model_downloaded("tinyllama:fp16", tmp_path) is True
        assert is_model_downloaded("tinyllama-fp16", tmp_path) is True

    def test_is_model_downloaded_false_for_xml_only_dir(self, tmp_path: Path):
        """Xml-only model directories are incomplete and must not look downloaded."""
        target = resolve_download_target("tinyllama", tmp_path)
        assert target is not None

        _, _, local_dir, _ = target
        local_dir.mkdir(parents=True)
        (local_dir / "openvino_model.xml").write_text("<model/>", encoding="utf-8")
        (local_dir / DOWNLOAD_METADATA_FILE).write_text(
            json.dumps(
                {
                    "source_repo": "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov",
                    "requires_token": False,
                }
            ),
            encoding="utf-8",
        )

        assert is_model_downloaded("tinyllama", tmp_path) is False


class TestGetDownloadedModels:
    """Tests for get_downloaded_models function."""

    def test_get_downloaded_models_list(self, tmp_path: Path):
        """Returns list of model names."""
        model1 = tmp_path / "model-one"
        model1.mkdir()
        (model1 / "openvino_model.xml").write_text("<model/>", encoding="utf-8")
        (model1 / "openvino_model.bin").write_bytes(b"weights")

        model2 = tmp_path / "model-two"
        model2.mkdir()
        (model2 / "openvino_model.xml").write_text("<model/>", encoding="utf-8")
        (model2 / "openvino_model.bin").write_bytes(b"weights")

        # Invalid directories (missing one or both required files)
        invalid = tmp_path / "not-a-model"
        invalid.mkdir()
        (invalid / "other_file.txt").touch()
        missing_bin = tmp_path / "missing-bin"
        missing_bin.mkdir()
        (missing_bin / "openvino_model.xml").write_text("<model/>", encoding="utf-8")

        result = get_downloaded_models(tmp_path)

        assert isinstance(result, list)
        assert len(result) == 2
        assert "model-one" in result
        assert "model-two" in result
        assert "not-a-model" not in result
        assert "missing-bin" not in result


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

        # Create the required files so verification succeeds
        (tmp_path / "openvino_model.xml").write_text("<model/>", encoding="utf-8")
        (tmp_path / "openvino_model.bin").write_bytes(b"weights")

        progress_gen = get_download_progress("test-repo/model", tmp_path)
        statuses = list(progress_gen)

        status_values = [s.get("status") for s in statuses]
        assert "success" in status_values

    @patch("npu_proxy.models.downloader.HfApi")
    @patch("npu_proxy.models.downloader.hf_hub_download")
    def test_download_progress_disables_ambient_token_by_default(
        self, mock_download, mock_api_class, tmp_path: Path
    ):
        """Streaming pulls should also disable ambient Hugging Face credentials."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.repo_info.return_value = MagicMock(siblings=[])

        def create_required_files(*args, **kwargs):
            filename = kwargs["filename"]
            target = tmp_path / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"data")
            return str(target)

        mock_download.side_effect = create_required_files

        statuses = list(get_download_progress("test-repo/model", tmp_path))

        assert statuses[-1]["status"] == "success"
        mock_api_class.assert_called_once_with(token=False)
        assert all(call.kwargs["token"] is False for call in mock_download.call_args_list)

    @patch("npu_proxy.models.downloader.HfApi")
    @patch("npu_proxy.models.downloader.hf_hub_download")
    def test_download_progress_falls_back_when_repo_info_fails(
        self, mock_download, mock_api_class, tmp_path: Path
    ):
        """Generator should still complete when repo metadata lookup fails."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.repo_info.side_effect = RuntimeError("metadata lookup failed")

        def create_required_files(*args, **kwargs):
            filename = kwargs["filename"]
            target = tmp_path / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"data")
            return str(target)

        mock_download.side_effect = create_required_files

        statuses = list(get_download_progress("test-repo/model", tmp_path))

        assert statuses[0]["status"] == "pulling manifest"
        assert statuses[-1]["status"] == "success"

    @patch("npu_proxy.models.downloader.HfApi")
    @patch("npu_proxy.models.downloader.hf_hub_download")
    def test_download_progress_uses_target_specific_required_files_for_downloads(
        self, mock_download, mock_api_class, tmp_path: Path
    ):
        """Streaming downloads should use the caller's target-specific required files."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.repo_info.side_effect = RuntimeError("metadata lookup failed")

        required_files = ("embedding.xml", "embedding.bin")

        def create_required_files(*args, **kwargs):
            filename = kwargs["filename"]
            target = tmp_path / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"data")
            return str(target)

        mock_download.side_effect = create_required_files

        statuses = list(
            get_download_progress(
                "test-repo/model",
                tmp_path,
                required_files=required_files,
            )
        )

        assert statuses[-1]["status"] == "success"
        assert [call.kwargs["filename"] for call in mock_download.call_args_list] == list(required_files)

    @patch("npu_proxy.models.downloader.HfApi")
    @patch("npu_proxy.models.downloader.hf_hub_download")
    def test_download_progress_verifies_target_specific_required_files(
        self, mock_download, mock_api_class, tmp_path: Path
    ):
        """Streaming verification should fail if caller-specific required files are missing."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.repo_info.side_effect = RuntimeError("metadata lookup failed")

        (tmp_path / "openvino_model.xml").write_text("<model/>", encoding="utf-8")
        (tmp_path / "openvino_model.bin").write_bytes(b"weights")
        mock_download.return_value = str(tmp_path / "unused")

        statuses = list(
            get_download_progress(
                "test-repo/model",
                tmp_path,
                required_files=("special.required",),
            )
        )

        assert statuses[-1]["status"] == "error: missing required files special.required"

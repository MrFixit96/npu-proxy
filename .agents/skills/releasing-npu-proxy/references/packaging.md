# Building distribution artifacts

## Contents

- Windows executable (PyInstaller)
- Debian package
- winget manifest
- Where artifacts land

## Windows executable (PyInstaller)

`scripts/build_windows.ps1` builds a standalone `npu-proxy.exe` bundling the
OpenVINO runtime via the spec `npu_proxy.pyinstaller.spec`.

```powershell
.\scripts\build_windows.ps1
```

Notes:
- It prefers `.\.venv\Scripts\python.exe`, else the active `$VIRTUAL_ENV`.
- It installs a **pinned** `pyinstaller==6.11.1` into that environment.
- It fails fast (`$ErrorActionPreference = "Stop"`) if no venv exists.

## Debian package

Assets live in `packaging/debian/` (`control`, `rules`, `changelog`,
`postinst`/`prerm`/`postrm`, `copyright`, `pip-constraints.txt`, `build.sh`) plus
the systemd unit `packaging/npu-proxy.service` and `packaging/npu-proxy.environment`.

Build via `packaging/debian/build.sh` on a Debian toolchain. Keep
`packaging/debian/changelog` in sync with the top-level `CHANGELOG.md` version.

## winget manifest

`packaging/winget/` holds the three-file manifest:
`MrFixit96.NPUProxy.yaml`, `...installer.yaml`, `...locale.en-US.yaml`. Update the
version and installer hash to match the published release artifact before
submitting.

## Where artifacts land

- PyInstaller output goes under `dist/` (per the spec).
- Keep the version in `pyproject.toml`, `npu_proxy/__init__.py`,
  `packaging/debian/changelog`, and the winget manifest consistent with the
  release tag.

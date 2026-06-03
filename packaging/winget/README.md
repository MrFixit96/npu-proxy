# WinGet manifest notes

This directory contains the WinGet manifests that currently exist in-tree for version `0.2.0`.

## Files

- `MrFixit96.NPUProxy.yaml`
- `MrFixit96.NPUProxy.locale.en-US.yaml`
- `MrFixit96.NPUProxy.installer.yaml`

## Current build artifact truth

`.\scripts\build_windows.ps1` builds:

```text
dist\npu-proxy.exe
```

The current installer manifest points at a release asset named:

```text
npu-proxy-0.2.0-win64.exe
```

So if you are publishing the current manifest set, you need to stage a release asset with that filename.

These manifests target the default OpenVINO runtime packaging path. The optional alpha `llama.cpp` GGUF experiment path is source-install only and is not represented in the WinGet package.

After installing the packaged app, the validated NPU embedding path is still an explicit opt-in: export `all-minilm`, set `NPU_PROXY_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`, and set `NPU_PROXY_EMBEDDING_DEVICE=NPU`.

## Typical workflow for the current manifests

```powershell
.\scripts\build_windows.ps1
Copy-Item dist\npu-proxy.exe dist\npu-proxy-0.2.0-win64.exe
Get-FileHash dist\npu-proxy-0.2.0-win64.exe -Algorithm SHA256
```

Then update `MrFixit96.NPUProxy.installer.yaml` if the hash changed and publish the release asset referenced by its `InstallerUrl`.

Example release command for the current manifest version:

```powershell
gh release create v0.2.0 dist\npu-proxy-0.2.0-win64.exe --title "v0.2.0" --notes "0.2.0 release"
```

## Local validation

```powershell
winget validate .\packaging\winget\
winget install --manifest .\packaging\winget\
```

## References

- https://learn.microsoft.com/windows/package-manager/package/manifest
- https://github.com/microsoft/winget-pkgs

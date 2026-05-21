# WinGet Package Submission Guide

This directory contains the WinGet manifest files for submitting npu-proxy to the
Windows Package Manager Community Repository.

## Files

- `MrFixit96.NPUProxy.yaml` - Version manifest
- `MrFixit96.NPUProxy.locale.en-US.yaml` - Localization manifest
- `MrFixit96.NPUProxy.installer.yaml` - Installer manifest

## Prerequisites

Before submitting:

1. **Create a GitHub Release** with the Windows executable:
   ```powershell
   # Build the executable
   cd <repo-root>
   .\scripts\build_windows.ps1
   
   # Create release on GitHub with the .exe
   gh release create v0.2.0 dist/npu-proxy-0.2.0-win64.exe dist/SHA256SUMS.txt --title "v0.2.0" --notes "0.2.0 release"
   ```

2. **Calculate SHA256** of the executable:
   ```powershell
   Get-FileHash dist\npu-proxy-0.2.0-win64.exe -Algorithm SHA256
   ```

3. **Update the installer manifest** with the SHA256 hash.

## Submission Process

1. Fork https://github.com/microsoft/winget-pkgs

2. Create the package directory:
   ```
   manifests/m/MrFixit96/NPUProxy/0.2.0/
   ```

3. Copy all three YAML files to that directory

4. Validate the manifests:
   ```powershell
   winget validate manifests/m/MrFixit96/NPUProxy/0.2.0/
   ```

5. Submit a pull request to microsoft/winget-pkgs

## Testing Locally

```powershell
# Install from local manifest
winget install --manifest .\packaging\winget\

# Verify installation
npu-proxy --version
```

## References

- [WinGet Manifest Schema](https://github.com/microsoft/winget-cli/blob/master/doc/ManifestSpecv1.6.md)
- [WinGet-pkgs Contributing](https://github.com/microsoft/winget-pkgs/blob/master/CONTRIBUTING.md)

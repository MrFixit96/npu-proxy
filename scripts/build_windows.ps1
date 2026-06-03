#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Build NPU Proxy as a Windows executable.

.DESCRIPTION
    Uses PyInstaller to create a standalone npu-proxy.exe that includes
    all dependencies including OpenVINO runtime. The build runs through the
    project virtual environment when available and installs a pinned
    PyInstaller into that environment.

.EXAMPLE
    .\scripts\build_windows.ps1
#>

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Push-Location $ProjectRoot

try {
    Write-Host "Building NPU Proxy..." -ForegroundColor Cyan

    $venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        $Python = $venvPython
        Write-Host "Using project virtual environment: $venvPython" -ForegroundColor Gray
    } elseif ($env:VIRTUAL_ENV) {
        $Python = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
        if (-not (Test-Path $Python)) {
            throw "Active virtual environment does not contain Scripts\python.exe: $env:VIRTUAL_ENV"
        }
        Write-Host "Using active virtual environment: $env:VIRTUAL_ENV" -ForegroundColor Gray
    } else {
        throw "No virtual environment found. Create one with 'python -m venv .venv' before building."
    }

    $PyInstallerVersion = "6.11.1"
    Write-Host "Installing pinned build dependency: pyinstaller==$PyInstallerVersion" -ForegroundColor Yellow
    & $Python -m pip install "pyinstaller==$PyInstallerVersion"

    # Clean previous builds
    if (Test-Path "dist") {
        Write-Host "Cleaning previous build..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force "dist"
    }
    if (Test-Path "build") {
        Remove-Item -Recurse -Force "build"
    }

    # Run PyInstaller
    Write-Host "Running PyInstaller..." -ForegroundColor Cyan
    & $Python -m PyInstaller npu_proxy.pyinstaller.spec --noconfirm

    if (Test-Path "dist\npu-proxy.exe") {
        $size = (Get-Item "dist\npu-proxy.exe").Length / 1MB
        Write-Host "`nBuild successful!" -ForegroundColor Green
        Write-Host "Executable: dist\npu-proxy.exe" -ForegroundColor Green
        Write-Host "Size: $([math]::Round($size, 2)) MB" -ForegroundColor Green

        # Test the executable
        Write-Host "`nTesting executable..." -ForegroundColor Cyan
        & "dist\npu-proxy.exe" --version
        exit $LASTEXITCODE
    } else {
        Write-Host "Build failed - executable not found" -ForegroundColor Red
        exit 1
    }
} finally {
    Pop-Location
}

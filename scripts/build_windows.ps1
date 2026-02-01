#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Build NPU Proxy as a Windows executable.

.DESCRIPTION
    Uses PyInstaller to create a standalone npu-proxy.exe that includes
    all dependencies including OpenVINO runtime.

.EXAMPLE
    .\scripts\build_windows.ps1
#>

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Push-Location $ProjectRoot

try {
    Write-Host "Building NPU Proxy..." -ForegroundColor Cyan
    
    # Check for PyInstaller
    if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
        Write-Host "Installing PyInstaller..." -ForegroundColor Yellow
        pip install pyinstaller
    }
    
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
    pyinstaller npu_proxy.spec --noconfirm
    
    if (Test-Path "dist/npu-proxy.exe") {
        $size = (Get-Item "dist/npu-proxy.exe").Length / 1MB
        Write-Host "`nBuild successful!" -ForegroundColor Green
        Write-Host "Executable: dist/npu-proxy.exe" -ForegroundColor Green
        Write-Host "Size: $([math]::Round($size, 2)) MB" -ForegroundColor Green
        
        # Test the executable
        Write-Host "`nTesting executable..." -ForegroundColor Cyan
        & "dist/npu-proxy.exe" --version
    } else {
        Write-Host "Build failed - executable not found" -ForegroundColor Red
        exit 1
    }
} finally {
    Pop-Location
}

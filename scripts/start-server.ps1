# NPU Proxy Server Startup Script
# Starts the FastAPI server for Intel NPU inference

[CmdletBinding()]
param(
    [switch]$ListenAll
)

$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "NPU Proxy Server"

function Test-Truthy($Value) {
    return $Value -match '^(1|true|yes|on)$'
}

$listenAllFromEnv = Test-Truthy $env:NPU_PROXY_LISTEN_ALL
if (-not $env:NPU_PROXY_HOST) {
    $env:NPU_PROXY_HOST = if ($ListenAll -or $listenAllFromEnv) { "0.0.0.0" } else { "127.0.0.1" }
}
$env:NPU_PROXY_PORT = if ($env:NPU_PROXY_PORT) { $env:NPU_PROXY_PORT } else { "11435" }

$port = 0
if (-not [int]::TryParse($env:NPU_PROXY_PORT, [ref]$port) -or $port -lt 1 -or $port -gt 65535) {
    Write-Error "NPU_PROXY_PORT must be an integer between 1 and 65535 (got '$env:NPU_PROXY_PORT')."
    exit 1
}

$loopbackHosts = @("127.0.0.1", "localhost", "::1")
if ($loopbackHosts -notcontains $env:NPU_PROXY_HOST) {
    Write-Warning "NPU Proxy is binding to '$env:NPU_PROXY_HOST', which may expose the API beyond this machine. Use -ListenAll or NPU_PROXY_LISTEN_ALL=true only on trusted networks."
}

Write-Host "Starting NPU Proxy Server..." -ForegroundColor Cyan
Write-Host "  Host: $env:NPU_PROXY_HOST" -ForegroundColor Gray
Write-Host "  Port: $env:NPU_PROXY_PORT" -ForegroundColor Gray
Write-Host ""

& python -m uvicorn npu_proxy.main:app --host $env:NPU_PROXY_HOST --port $port
exit $LASTEXITCODE

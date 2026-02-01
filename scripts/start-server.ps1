# NPU Proxy Server Startup Script
# Starts the FastAPI server for Intel NPU inference

$Host.UI.RawUI.WindowTitle = "NPU Proxy Server"

$env:NPU_PROXY_HOST = if ($env:NPU_PROXY_HOST) { $env:NPU_PROXY_HOST } else { "0.0.0.0" }
$env:NPU_PROXY_PORT = if ($env:NPU_PROXY_PORT) { $env:NPU_PROXY_PORT } else { "11435" }

Write-Host "Starting NPU Proxy Server..." -ForegroundColor Cyan
Write-Host "  Host: $env:NPU_PROXY_HOST" -ForegroundColor Gray
Write-Host "  Port: $env:NPU_PROXY_PORT" -ForegroundColor Gray
Write-Host ""

python -m uvicorn npu_proxy.main:app --host $env:NPU_PROXY_HOST --port $env:NPU_PROXY_PORT

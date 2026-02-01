# Launch Claude CLI with NPU Proxy as Ollama backend
# Sets OLLAMA_HOST to point to the local NPU proxy server

$env:OLLAMA_HOST = "http://localhost:11435"

Write-Host "Launching Claude CLI with NPU Proxy backend" -ForegroundColor Cyan
Write-Host "  OLLAMA_HOST=$env:OLLAMA_HOST" -ForegroundColor Gray
Write-Host ""

# Check if proxy is running
try {
    $health = Invoke-RestMethod -Uri "$env:OLLAMA_HOST/health" -Method Get -TimeoutSec 2
    Write-Host "NPU Proxy Status: $($health.status)" -ForegroundColor Green
    Write-Host "  NPU Available: $($health.npu_available)" -ForegroundColor Gray
    Write-Host ""
} catch {
    Write-Host "WARNING: NPU Proxy not responding at $env:OLLAMA_HOST" -ForegroundColor Yellow
    Write-Host "  Start the server with: .\scripts\start-server.ps1" -ForegroundColor Yellow
    Write-Host ""
}

# Launch Claude CLI
claude --provider ollama

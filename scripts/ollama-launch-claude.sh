#!/bin/bash
# Launch Claude CLI with NPU Proxy as Ollama backend
# Sets OLLAMA_HOST to point to the Windows host NPU proxy server

# Get Windows host IP (for WSL2)
if grep -qi microsoft /proc/version 2>/dev/null; then
    # Running in WSL2 - get Windows host IP
    WINDOWS_HOST=$(ip route show | grep -i default | awk '{ print $3 }')
    export OLLAMA_HOST="http://${WINDOWS_HOST}:11435"
else
    # Native Linux - use localhost
    export OLLAMA_HOST="http://localhost:11435"
fi

echo "Launching Claude CLI with NPU Proxy backend"
echo "  OLLAMA_HOST=$OLLAMA_HOST"
echo ""

# Check if proxy is running
if curl -s --connect-timeout 2 "$OLLAMA_HOST/health" > /dev/null 2>&1; then
    health=$(curl -s "$OLLAMA_HOST/health")
    echo "NPU Proxy Status: $(echo $health | jq -r '.status')"
    echo "  NPU Available: $(echo $health | jq -r '.npu_available')"
    echo ""
else
    echo "WARNING: NPU Proxy not responding at $OLLAMA_HOST"
    echo "  Start the server on Windows with: .\\scripts\\start-server.ps1"
    echo ""
fi

# Launch Claude CLI
claude --provider ollama

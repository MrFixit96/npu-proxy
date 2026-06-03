#!/bin/bash
# Launch Claude CLI with NPU Proxy as Ollama backend
# Sets OLLAMA_HOST to point to the Windows host NPU proxy server when unset
set -euo pipefail

require_tool() {
    local tool="$1"
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "Required tool not found: ${tool}" >&2
        exit 1
    fi
}

require_tool curl
require_tool jq
require_tool claude

if [[ -z "${OLLAMA_HOST:-}" ]]; then
    if grep -qi microsoft /proc/version 2>/dev/null; then
        require_tool ip
        WINDOWS_HOST="$(ip route show default | awk '{ print $3; exit }')"
        if [[ -z "${WINDOWS_HOST}" ]]; then
            echo "Could not determine Windows host IP from default route." >&2
            exit 1
        fi
        export OLLAMA_HOST="http://${WINDOWS_HOST}:11435"
    else
        export OLLAMA_HOST="http://localhost:11435"
    fi
fi

echo "Launching Claude CLI with NPU Proxy backend"
echo "  OLLAMA_HOST=${OLLAMA_HOST}"
echo ""

# Check if proxy is running
if health="$(curl -fsS --connect-timeout 2 "${OLLAMA_HOST}/health" 2>/dev/null)"; then
    echo "NPU Proxy Status: $(printf '%s' "${health}" | jq -r '.status')"
    echo "  NPU Available: $(printf '%s' "${health}" | jq -r '.npu_available')"
    echo ""
else
    echo "WARNING: NPU Proxy not responding at ${OLLAMA_HOST}"
    echo "  Start the server on Windows with: .\scripts\start-server.ps1"
    echo ""
fi

# Launch Claude CLI
exec claude --provider ollama

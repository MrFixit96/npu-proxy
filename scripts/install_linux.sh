#!/bin/bash
#
# NPU Proxy Linux Installation Script
#
# Usage: sudo ./install_linux.sh
#
set -euo pipefail

# Colors for output
RED='[0;31m'
GREEN='[0;32m'
YELLOW='[1;33m'
NC='[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVICE_SOURCE="${PROJECT_ROOT}/packaging/npu-proxy.service"
ENV_SOURCE="${PROJECT_ROOT}/packaging/npu-proxy.environment"

require_file() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo -e "${RED}Required file not found: ${path}${NC}" >&2
        exit 1
    fi
}

require_file "${SERVICE_SOURCE}"
require_file "${ENV_SOURCE}"

if [[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
    echo -e "${RED}Project root is missing pyproject.toml: ${PROJECT_ROOT}${NC}" >&2
    exit 1
fi

echo -e "${GREEN}Installing NPU Proxy from ${PROJECT_ROOT}...${NC}"

# Check for root
if [[ "${EUID}" -ne 0 ]]; then
    echo -e "${RED}Please run as root (sudo)${NC}"
    exit 1
fi

# Create user and group if they don't exist
if ! id "npu-proxy" &>/dev/null; then
    echo "Creating npu-proxy user..."
    useradd --system --no-create-home --shell /usr/sbin/nologin npu-proxy
fi

# Add user to render group for Intel NPU access
if getent group render > /dev/null; then
    usermod -aG render npu-proxy
    echo "Added npu-proxy to render group"
fi

# Create directories
echo "Creating directories..."
install -d -m 0755 /etc/npu-proxy
install -d -m 0755 -o npu-proxy -g npu-proxy /var/lib/npu-proxy
install -d -m 0755 -o npu-proxy -g npu-proxy /var/log/npu-proxy

# Install Python package
echo "Installing Python package..."
python3 -m pip install "${PROJECT_ROOT}"

# Install systemd service
echo "Installing systemd service..."
install -m 0644 "${SERVICE_SOURCE}" /etc/systemd/system/npu-proxy.service

# Install default environment file if not exists
if [[ ! -f /etc/npu-proxy/environment ]]; then
    install -m 0640 -o root -g npu-proxy "${ENV_SOURCE}" /etc/npu-proxy/environment
else
    echo -e "${YELLOW}Existing /etc/npu-proxy/environment preserved.${NC}"
fi

# Reload systemd
systemctl daemon-reload

echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Edit configuration: sudo nano /etc/npu-proxy/environment"
echo "  2. Start the service:  sudo systemctl start npu-proxy"
echo "  3. Enable at boot:     sudo systemctl enable npu-proxy"
echo "  4. Check status:       sudo systemctl status npu-proxy"
echo "  5. View logs:          sudo journalctl -u npu-proxy -f"

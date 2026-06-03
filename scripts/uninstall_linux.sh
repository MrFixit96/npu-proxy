#!/bin/bash
#
# NPU Proxy Linux Uninstall Script
#
set -euo pipefail

RED='[0;31m'
GREEN='[0;32m'
YELLOW='[1;33m'
NC='[0m'

if [[ "${EUID}" -ne 0 ]]; then
    echo -e "${RED}Please run as root (sudo)${NC}"
    exit 1
fi

echo "Stopping NPU Proxy..."
systemctl stop npu-proxy 2>/dev/null || true
systemctl disable npu-proxy 2>/dev/null || true

echo "Removing systemd service..."
rm -f /etc/systemd/system/npu-proxy.service
systemctl daemon-reload

echo "Removing Python package..."
python3 -m pip uninstall -y npu-proxy 2>/dev/null || true

echo "Removing directories..."
rm -rf /var/lib/npu-proxy
rm -rf /var/log/npu-proxy

# Keep /etc/npu-proxy for config preservation
echo -e "${YELLOW}Note: Configuration in /etc/npu-proxy preserved. Remove manually if desired.${NC}"

echo -e "${GREEN}Uninstall complete!${NC}"

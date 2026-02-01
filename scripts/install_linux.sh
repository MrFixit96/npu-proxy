#!/bin/bash
#
# NPU Proxy Linux Installation Script
#
# Usage: sudo ./install_linux.sh
#
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Installing NPU Proxy...${NC}"

# Check for root
if [ "$EUID" -ne 0 ]; then
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
mkdir -p /etc/npu-proxy
mkdir -p /var/lib/npu-proxy
mkdir -p /var/log/npu-proxy

# Set ownership
chown npu-proxy:npu-proxy /var/lib/npu-proxy
chown npu-proxy:npu-proxy /var/log/npu-proxy

# Install Python package
echo "Installing Python package..."
pip3 install --system .

# Install systemd service
echo "Installing systemd service..."
cp packaging/npu-proxy.service /etc/systemd/system/
chmod 644 /etc/systemd/system/npu-proxy.service

# Install default environment file if not exists
if [ ! -f /etc/npu-proxy/environment ]; then
    cp packaging/npu-proxy.environment /etc/npu-proxy/environment
    chmod 640 /etc/npu-proxy/environment
    chown root:npu-proxy /etc/npu-proxy/environment
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

# Debian Package Build Guide

This directory contains the Debian packaging files for npu-proxy.

## Files

- `control` - Package metadata and dependencies
- `copyright` - License information
- `changelog` - Version history
- `rules` - Build rules (Makefile)
- `compat` - Debhelper compatibility level
- `postinst` - Post-installation script
- `prerm` - Pre-removal script
- `postrm` - Post-removal script
- `build.sh` - Build script

## Prerequisites

Install build dependencies:

```bash
sudo apt-get install build-essential debhelper devscripts python3-all dh-python
```

## Building the Package

From the repository root:

```bash
chmod +x packaging/debian/build.sh
./packaging/debian/build.sh
```

The .deb file will be created in `dist/`.

## Installing the Package

```bash
sudo dpkg -i dist/npu-proxy_0.1.0-1_amd64.deb

# Install dependencies if needed
sudo apt-get install -f
```

## Package Management

```bash
# Start service
sudo systemctl start npu-proxy

# Enable at boot
sudo systemctl enable npu-proxy

# Check status
sudo systemctl status npu-proxy

# View logs
sudo journalctl -u npu-proxy -f

# Remove package
sudo apt-get remove npu-proxy

# Purge (remove config and data)
sudo apt-get purge npu-proxy
```

## Configuration

Edit `/etc/default/npu-proxy` to configure environment variables:

```bash
NPU_PROXY_HOST=0.0.0.0
NPU_PROXY_PORT=11435
NPU_PROXY_DEVICE=NPU
NPU_PROXY_REAL_INFERENCE=1
```

Then restart:

```bash
sudo systemctl restart npu-proxy
```

## Directory Structure

After installation:

```
/opt/npu-proxy/venv/     # Python virtual environment
/var/lib/npu-proxy/      # Model cache and data
/var/log/npu-proxy/      # Log files
/etc/default/npu-proxy   # Configuration
/lib/systemd/system/npu-proxy.service  # Systemd unit
```

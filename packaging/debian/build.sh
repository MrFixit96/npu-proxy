#!/bin/bash
# Build Debian package for npu-proxy
# Run from repository root: ./packaging/debian/build.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VERSION=$(grep -oP 'version = "\K[^"]+' "$REPO_ROOT/pyproject.toml" | head -1)

echo "Building npu-proxy ${VERSION} Debian package..."

# Create build directory
BUILD_DIR=$(mktemp -d)
PACKAGE_DIR="$BUILD_DIR/npu-proxy-${VERSION}"

echo "Build directory: $BUILD_DIR"

# Copy source
mkdir -p "$PACKAGE_DIR"
tar \
    --exclude='.git' \
    --exclude='dist' \
    --exclude='build' \
    --exclude='.pytest_cache' \
    --exclude='__pycache__' \
    -C "$REPO_ROOT" -cf - . | tar -C "$PACKAGE_DIR" -xf -

# Copy debian directory to source root (required by dpkg-buildpackage)
cp -r "$REPO_ROOT/packaging/debian" "$PACKAGE_DIR/"
chmod 755 "$PACKAGE_DIR/debian/rules" "$PACKAGE_DIR/debian/postinst" "$PACKAGE_DIR/debian/prerm" "$PACKAGE_DIR/debian/postrm"

# Build package
cd "$PACKAGE_DIR"
dpkg-buildpackage -us -uc -b

# Copy results
mkdir -p "$REPO_ROOT/dist"
cp "$BUILD_DIR"/*.deb "$REPO_ROOT/dist/" 2>/dev/null || true

echo ""
echo "Build complete! Package(s) in dist/:"
ls -la "$REPO_ROOT/dist/"*.deb 2>/dev/null || echo "No .deb files found"

# Cleanup
rm -rf "$BUILD_DIR"

# Debian packaging notes

This directory contains the Debian packaging assets currently present in the repository.

## Files

- `build.sh`
- `control`
- `rules`
- `changelog`
- `postinst`
- `prerm`
- `postrm`
- `pip-constraints.txt`

## Build prerequisites

```bash
sudo apt-get install build-essential debhelper devscripts python3-all dh-python python3-build python3-setuptools python3-wheel python3-venv
```

## Building

Run from a native Linux filesystem (for example your WSL home directory, not `/mnt/c/...`):

```bash
chmod +x packaging/debian/build.sh
./packaging/debian/build.sh
```

Current `build.sh` behavior:

- creates a temporary build tree
- builds the package with `dpkg-buildpackage`
- copies resulting `.deb` files into `dist/`

## Installing the current package

The current Debian changelog version is `0.2.0-2`, so the built artifact will typically look like:

```bash
sudo dpkg -i dist/npu-proxy_0.2.0-2_amd64.deb
sudo apt-get install -f
```

If you prefer not to hardcode the version:

```bash
sudo dpkg -i dist/npu-proxy_*_amd64.deb
sudo apt-get install -f
```

## What the package installs

Current packaging rules install:

- app wheel → `/usr/share/npu-proxy/wheels/`
- pinned Python constraints → `/usr/share/npu-proxy/constraints.txt`
- systemd unit → `/lib/systemd/system/npu-proxy.service`
- environment file → `/etc/default/npu-proxy`

At install time, `postinst` also creates:

- runtime venv → `/opt/npu-proxy/venv`
- state dir → `/var/lib/npu-proxy`
- log dir → `/var/log/npu-proxy`

## Service management

```bash
sudo systemctl start npu-proxy
sudo systemctl enable npu-proxy
sudo systemctl status npu-proxy
sudo journalctl -u npu-proxy -f
```

## Current packaged configuration defaults

Edit:

```text
/etc/default/npu-proxy
```

The packaged environment template currently defaults to:

```bash
NPU_PROXY_HOST=127.0.0.1
NPU_PROXY_PORT=8080
NPU_PROXY_DEVICE=AUTO
NPU_PROXY_TOKEN_LIMIT=1800
NPU_PROXY_REAL_INFERENCE=1
NPU_PROXY_EMBEDDING_DEVICE=CPU
NPU_PROXY_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
NPU_PROXY_INFERENCE_TIMEOUT=180
NPU_PROXY_LOAD_TIMEOUT=300
```

These packaged defaults describe the standard OpenVINO runtime path. The optional alpha `llama.cpp` GGUF path is not packaged here as a first-class install target; using it requires a source install plus manual `llama-cpp-python` dependency management.

If you want the validated NPU embedding path after installation, keep the packaged defaults as-is, then place an `all-minilm` OpenVINO export under the packaged service-user cache path:

```text
/var/lib/npu-proxy/.cache/npu-proxy/models/embeddings/all-minilm-l6-v2
```

This can come from a source checkout, a manual `optimum-cli export openvino ...` run, or another trusted prepared export, but it must end up owned by the `npu-proxy` service user. Then set:

```bash
NPU_PROXY_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
NPU_PROXY_EMBEDDING_DEVICE=NPU
```

See `docs/guides/MODEL_DOWNLOAD.md` for the supported export flow and runtime cache paths.

Restart after changes:

```bash
sudo systemctl restart npu-proxy
```

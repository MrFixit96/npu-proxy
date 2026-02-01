# Contributing to NPU Proxy

Thank you for your interest in NPU Proxy! This is a personal project, and I appreciate any contributions.

## Project Status

This is a **personal/hobby project**. I check for issues and pull requests occasionally, so please be patient with response times.

## How to Contribute

### Found a Bug?

1. Check if the issue already exists in [GitHub Issues](https://github.com/MrFixit96/npu-proxy/issues)
2. If not, open a new issue with:
   - Your environment (OS, Python version, OpenVINO version)
   - Steps to reproduce
   - Expected vs actual behavior

### Want to Add a Feature?

1. Open an issue first to discuss the idea
2. Fork the repository
3. Create a feature branch
4. Write tests for your changes
5. Submit a pull request

### Forking Encouraged

If you need changes faster than I can review them, or want to take the project in a different direction, **please fork!** That's what open source is for.

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/npu-proxy.git
cd npu-proxy

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # Linux/macOS

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest
```

## Code Style

- Follow existing code patterns
- Add type hints to new functions
- Include docstrings (Google style)
- Run tests before submitting PRs

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

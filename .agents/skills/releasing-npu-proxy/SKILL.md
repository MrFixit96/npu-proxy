---
name: releasing-npu-proxy
description: >-
  Cuts a versioned NPU Proxy release and builds its distribution artifacts. Use
  when bumping the version, stamping the CHANGELOG, tagging, creating a GitHub
  release, or building the PyInstaller exe, Debian package, or winget manifest.
  Also use when the user mentions release, version bump, changelog, git tag,
  packaging, or shipping a new version.
license: Apache-2.0
compatibility: >-
  Requires the npu-proxy repo, its .venv, git, and the gh CLI. PowerShell has no
  heredocs; commit/tag via a temp file with `git -F`. Windows builds need
  PyInstaller; Debian builds need a Debian toolchain.
metadata:
  author: npu-proxy
  version: "1.0"
---

# Releasing NPU Proxy

A release is a coordinated version bump + changelog stamp + tag + GitHub release,
optionally followed by building distribution artifacts. The steps below are
**low-freedom**: follow them exactly, because the version lives in two places and
the editable install can silently report a stale version.

## Version sources (BOTH must change together)

1. `pyproject.toml` → `version = "X.Y.Z"`
2. `npu_proxy/__init__.py` → the `_detect_version()` fallback `return "X.Y.Z"`

`__version__` resolves from installed package metadata first, then the fallback.
After bumping, **reinstall editable** so the runtime and `/health` report the new
version, otherwise tests and the live server keep showing the old one:

```bash
$env:VIRTUAL_ENV = "$PWD\.venv"; uv pip install -e . --no-deps -q
.\.venv\Scripts\python.exe -c "import npu_proxy; print(npu_proxy.__version__)"
```

> Gotcha: on OneDrive, a locked `dist-info\licenses` dir can make the reinstall
> leave the editable `.pth` missing — then `import npu_proxy` only works from the
> repo root. Remove the orphaned `dist-info` and reinstall until the `.pth` and
> `__editable__*finder.py` reappear.

## Release checklist

```
Release vX.Y.Z:
- [ ] 1. Bump version in pyproject.toml AND __init__.py fallback
- [ ] 2. Reinstall editable; confirm __version__ == X.Y.Z
- [ ] 3. Stamp CHANGELOG: [Unreleased] -> [X.Y.Z] - <date>; refresh compare links
- [ ] 4. Fast suite green
- [ ] 5. Commit (temp-file -F, with Co-authored-by trailer); push
- [ ] 6. Merge the release PR into master (--merge)
- [ ] 7. Annotated tag vX.Y.Z on master; push the tag
- [ ] 8. gh release create vX.Y.Z (notes from CHANGELOG)
```

Commands:

```bash
.\.venv\Scripts\python.exe -m pytest -q -m "not slow and not e2e"
gh pr merge <N> --merge
git tag -a vX.Y.Z -F .git\TAG_MSG.txt   # then: git push origin vX.Y.Z
gh release create vX.Y.Z --title "vX.Y.Z" --notes-file <notes>
```

## CHANGELOG conventions (Keep a Changelog)

- Promote `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`; keep `Added/Changed/
  Fixed/Security` subsections.
- Update the footer compare links: add `[X.Y.Z]: .../compare/v<prev>...vX.Y.Z`
  and repoint `[Unreleased]` to `compare/vX.Y.Z...HEAD`.

## Conventions that bite

- **Annotated tags** for `.0` releases (`git tag -a`), matching v0.1.0/v0.2.0.
- **`--merge`** strategy (the repo history uses merge commits, not squash).
- **PowerShell has no heredocs**: write the commit/tag/notes body to a temp file,
  use `git commit -F` / `git tag -F` / `gh ... --notes-file`, then delete it.
- Always `git --no-pager`.
- Commit trailer: `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`.

## Reference

- **Building artifacts (PyInstaller exe, Debian, winget)**:
  [references/packaging.md](references/packaging.md)

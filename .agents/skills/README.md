# Agent Skills

This directory contains [Agent Skills](https://agentskills.io) for the
`npu-proxy` repository — packaged, version-controlled procedural knowledge that
AI coding agents load on demand to work safely in this codebase.

## Why `.agents/skills/`?

`.agents/skills/` is the cross-client discovery convention. Skills-compatible
agents (Claude Code, Cursor, GitHub Copilot, Gemini CLI, and others) scan it
automatically at session start, so **cloning the repo is all that's required** —
no install step. Each agent preloads only the `name` + `description` of every
skill, then reads a full `SKILL.md` (and its `references/`) only when the task is
relevant (progressive disclosure).

## Available skills

| Skill | Use when you are… |
| --- | --- |
| `certifying-device-routing` | changing device selection/fallback, headers/metrics, or running `scripts/certify_npu.py` |
| `converting-models-for-npu` | adding/converting models to OpenVINO IR, debugging missing `openvino_model.*` |
| `serving-npu-embeddings` | changing embedding models/devices/fallback or the embedding endpoints |
| `maintaining-api-compat` | adding/changing OpenAI or Ollama endpoints, schemas, or streaming |
| `operating-npu-proxy` | running the server, setting `NPU_PROXY_*` env vars, choosing test markers |
| `releasing-npu-proxy` | bumping the version, stamping the CHANGELOG, tagging, packaging |

## Authoring conventions

Each skill is a directory whose name matches its `name` field and contains a
`SKILL.md` (YAML frontmatter + Markdown body) plus an optional `references/`
folder. Follow the [Agent Skills specification](https://agentskills.io/specification)
and keep `SKILL.md` bodies lean (under ~500 lines), with reference files one
level deep.

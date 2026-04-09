# AGENTS.md

Rules for AI agents working in this repository.

## No Backward Compatibility

Do not implement backward-compatible shims, wrappers, or migration layers.
When replacing a tool, library, or pattern, remove the old code entirely.
Do not keep deprecated code paths "just in case."

## Fallbacks Require Explicit Approval

Never silently implement a fallback mechanism (e.g. "try X, if unavailable fall
back to Y"). If a fallback seems appropriate, stop and ask before implementing
it. The answer may be to fail loudly instead.

## Standalone-First Architecture

The `src/modelling/` package must have zero ZenML imports. ZenML orchestration
lives exclusively in `src/pipelines/`, which is a separate workspace member.
Any code in `src/modelling/` must work without ZenML installed.

## Environment Variables Over Hardcoded Config

Use environment variables (loaded via direnv / `.env`) for runtime
configuration such as tracking URIs and API tokens. Do not hardcode paths or
secrets. Document new env vars in `.env-default`.

## Use UV For Python Workflows

Use `uv` for Python package management and command execution in this repository.
Prefer `uv sync`, `uv add`, and `uv run ...` over direct `pip` or bare `python`
invocations when working with workspace packages and scripts.

## Clean Removal

When removing a dependency or integration (e.g. Neptune), delete all related
code, configs, env vars, and gitignore entries in one pass. Do not leave stubs
or "TODO: remove later" comments.

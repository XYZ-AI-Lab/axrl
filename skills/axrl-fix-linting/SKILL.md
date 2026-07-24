---
name: axrl-fix-linting
description: Fix lint, formatting, pyright, and mypy failures in this repository by running scripts/run-precommit-check.sh, making the smallest non-semantic changes, and stopping if a fix would change logits or training semantics.
argument-hint: Describe the failing check, files, or error messages to fix
user-invocable: true
---

# axrl-fix-linting

Use this skill to fix lint, formatting, pyright, or mypy failures in this repository with the standard validation command.

## Defaults

- Use `bash scripts/run-precommit-check.sh` as the source of truth.
- Run the lint/type-check workflow on the local CPU node in the shared workspace, not on GPU nodes. GPU nodes may lack outbound network access for pre-commit hook initialization, and the local CPU node shares the same files and Docker image.
- Keep fixes minimal and non-semantic by default.
- Do not change logits, training semantics, reward logic, sampling behavior, or public behavior unless the user explicitly approves it.
- Use the existing Python environment if present; do not create a new virtual environment unless the user asks.

The repo check command is:

```bash
bash scripts/run-precommit-check.sh
```

It runs `pre-commit`, `pyright`, and `mypy`.

## Workflow

1. Run the repo check script locally on the CPU node and fix the first failing stage.
2. Prefer the smallest local fix:
   - formatter or lint failure: style-only fix
   - pyright or mypy failure: fix the owning type contract or shared source
3. After the first edit, run the narrowest relevant check, then rerun `bash scripts/run-precommit-check.sh`.
4. If a fix would change behavior or logits, stop and ask before proceeding.
5. If several failures share one root cause, fix the shared source once instead of scattering one-off patches.

## Done When

- `bash scripts/run-precommit-check.sh` passes, or
- the remaining blocker is identified clearly, including whether it is style-only, type-only, or semantic.

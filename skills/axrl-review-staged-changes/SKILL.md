---
name: axrl-review-staged-changes
description: Review git-staged changes (a specific file if given, otherwise all staged files) for correctness bugs, coding-style and codebase-consistency issues, and suggestions for modularization, maintainability, efficiency, and readability. Report findings without applying fixes unless the user asks.
argument-hint: Optional file path to narrow the review; default is all staged files
user-invocable: true
---

# axrl-review-staged-changes

Use this skill to produce a focused review of the code the user is about to commit.

## Defaults

- Review `git diff --cached` only. Do not review unstaged edits, committed history, or the working tree.
- If the user provides a file path, scope the diff to that path: `git diff --cached -- <path>`.
- Do not edit code. Report findings and let the user decide. Only apply a fix when the user explicitly asks.
- Read the full file around each changed hunk before flagging an issue; a line can look wrong in isolation and be correct in context.
- Match the reviewed file's existing conventions. Don't invent house style; infer it from the surrounding code.
- Keep the report short; group findings by severity so the user can skim.

## Workflow

1. Run `git diff --cached` (scoped to `<path>` if provided). If the diff is empty, say so and stop.
2. For each changed file, read the current file state (not just the diff hunks) to understand surrounding code and naming conventions.
3. Trace impact: for each new/renamed/changed symbol, find its call sites and consumers before assuming the change is self-contained.
4. Evaluate each hunk through the following lenses, in this order:
   1. **Correctness** — does the code do what it claims, for every input and branch the diff can reach?
   2. **Consistency** — does it match the patterns, idioms, and style already used in the same file and nearby modules?
   3. **Modularization** — is logic at the right level of abstraction? Would a helper clarify intent, or is an introduced helper unused?
   4. **Maintainability** — will a future reader understand it without this PR's context? Are magic values, hidden coupling, or dead code present?
   5. **Efficiency** — are there obvious per-iteration costs, redundant work, or allocations that matter at the scale this code runs at?
   6. **Readability** — naming, comment quality, line length, comprehensions, dead syntax (`as _`, trailing `continue`, redundant `else`).
5. Produce a grouped report:
   - **Blocking** (bugs that crash, corrupt state, or change behavior unexpectedly)
   - **Should-fix** (correctness-adjacent issues or style inconsistencies worth doing in this PR)
   - **Nit** (cosmetic, optional)
6. For each finding, cite `file_path:line_number`, quote only what's essential, and give a concrete suggestion — show the shape of the fix, don't just describe the problem.
7. End with one verdict line: either `Looks good, safe to commit` or the single highest-priority blocker.

## Report format

```
Review: <file or "all staged files">

🔴 Blocking
- <path:line> — <one-line summary>
  <concrete suggestion>

🟡 Should-fix
- <path:line> — <one-line summary>

⚪ Nit
- <path:line> — <one-line summary>

<verdict line>
```

Use plain text markers if the user disallows emoji.

## Don't

- Don't rewrite code unless asked. Reviews are read-only.
- Don't re-review committed history — scope is strictly staged changes.
- Don't speculate about intent when the diff is ambiguous; ask the user.
- Don't flag a style point that the surrounding file already violates — match the file, not your preference.
- Don't mix review output with a commit. The user runs `git commit` themselves.

## Done When

- The staged diff (or the provided path subset) has been read in full.
- Each finding is tied to a specific line and a concrete suggestion.
- The final verdict line states whether the change is safe to commit or names the top blocker.

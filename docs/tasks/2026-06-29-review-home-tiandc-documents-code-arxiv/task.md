# Codex Task: 2026-06-29-review-home-tiandc-documents-code-arxiv

task_id: 2026-06-29-review-home-tiandc-documents-code-arxiv
target_project: /home/tiandc/Documents/code/arxiv-daily
task_kind: implementation
mode: semi-auto
sandbox: read-only
provider: bnu
artifact_policy: keep-report-only
source: claude-code-prompt

## Goal

全面 review /home/tiandc/Documents/code/arxiv-daily 项目，检查代码质量、架构设计、潜在问题和改进建议。重点关注：
1. arxiv_daily.py 的整体架构和逻辑
2. plugin/ 目录下的 Obsidian 插件代码
3. extensions/ 目录下的 VS Code 扩展代码
4. 安全性和可维护性问题

## Scope

Allowed:

- Make the focused changes needed to satisfy the goal.

Out of scope:

- Unrelated refactors.
- `git add`.
- `git commit`.

## Constraints

- Do not run `git add`.
- Do not run `git commit`.
- Do not write temporary files outside `.codex-runs/2026-06-29-review-home-tiandc-documents-code-arxiv/`.
- Preserve unrelated user changes.

## Verification

Commands:

- Run the project's existing tests or build when applicable.

Expected result:

- The goal is complete and the final report explains verification.

## Report

Write report to:

```text
docs/tasks/2026-06-29-review-home-tiandc-documents-code-arxiv/codex-report.md
```

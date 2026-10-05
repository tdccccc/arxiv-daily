# P3 — Direction review in the workbench

goal_ref: ../goal.md
created: 2026-10-05T12:59:01+08:00
updated: 2026-10-05T12:59:01+08:00
revision: 1

## Outcome

用户在DSH工作台按0.5.0流程检查候选方向及代表论文，编辑/移动/删除/改名/预览并显式接受到普通研究主题，查看文献库分析概览。

## Assumptions

- Existing core proposal editing, acceptance receipts, preview classifier and topic settings remain authoritative.
- User-triggered generation and preview may call configured models in normal use; tests use fixtures only.
- No standalone profile store is restored. Existing settings remain the editor for accepted directions.

## Approach

Expose the existing pure proposal operations through LibraryWorkflow under revision guards. Add typed workbench review snapshots/actions with host lock and config transaction. Long model actions use the existing cancellable run tray. Add proposed directions and library overview tabs in the right pane, leaving reading navigation intact.

## Chunks

### Chunk 1 — Core workflow review operations

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red: library-workflow tests require rename/move/remove/preview service methods, stale revision rejection and zero-write preview
- Green/regression: library-workflow, proposal-review, direction-preview and acceptance tests; core typecheck
- [ ] accepted

### Chunk 2 — Structured HTTP review and model jobs

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red: missing snapshot/action HTTP endpoints; stale config/proposal must reject; acceptance must persist once
- Green/regression: temporary fixture service/HTTP tests, same-process mutation serialization, cancellation and run-state handling; CLI library/settings tests
- [ ] accepted

### Chunk 3 — Review and overview UI

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red: DOM cannot inspect/edit/select/accept candidates or distinguish processed/thin evidence/coverage
- Green/regression: DOM fixtures exercise editing, representative selection, rename/move/remove, preview, partial acceptance, conflict preservation, bilingual output and back/forward; CLI tests/typecheck/build and isolated DSH Host
- [ ] accepted

## Phase verification

Build one new local DSH package and update user-facing docs after integration. No production installation, push or main merge without later instruction. Record exact limitations rather than claiming complete visual parity. Independent core and UI chunks may be delegated; main owner controls Helm and commits.

## Abort / reshape triggers

If edits require replacing shared domain semantics, stop and adapt shared application service instead. Historical proposal coverage is not current coverage when authoritative directions changed. Never auto-run models when opening a view or auto-accept candidates.

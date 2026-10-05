# P3 — Direction review in the workbench

goal_ref: ../goal.md
created: 2026-10-05T12:59:01+08:00
updated: 2026-10-05T13:14:56+08:00
revision: 2

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
- [x] accepted

### Chunk 2 — Structured HTTP review and model jobs

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red: missing snapshot/action HTTP endpoints; stale config/proposal must reject; acceptance must persist once
- Green/regression: temporary fixture service/HTTP tests, same-process mutation serialization, cancellation and run-state handling; CLI library/settings tests
- [x] accepted

### Chunk 3 — Review and overview UI

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red: DOM cannot inspect/edit/select/accept candidates or distinguish processed/thin evidence/coverage
- Green/regression: DOM fixtures exercise editing, representative selection, rename/move/remove, preview, partial acceptance, conflict preservation, bilingual output and back/forward; CLI tests/typecheck/build and isolated DSH Host
- [x] accepted

## Phase verification

Build one new local DSH package and update user-facing docs after integration. No production installation, push or main merge without later instruction. Record exact limitations rather than claiming complete visual parity. Independent core and UI chunks may be delegated; main owner controls Helm and commits.

## Abort / reshape triggers

If edits require replacing shared domain semantics, stop and adapt shared application service instead. Historical proposal coverage is not current coverage when authoritative directions changed. Never auto-run models when opening a view or auto-accept candidates.

## Acceptance evidence

- Core operations: observed missing-method and processed-candidate Red, then102 focused Green; Obsidian review/authorization regression137 Green. Core commit df00de4.
- Structured services/HTTP: actual temporary-store proposal generation, preview, acceptance and configuration reload passed. Independent authorization endpoint observed404 Red then16 HTTP Green. Delayed request body demonstrated search bypassing the busy gate (200 instead of409); fixed and17 HTTP Green. Held-write-lock reads observed timeout Red then catalog/review/PDF Green without weakening mutation locks. Service commit20d90e6.
- UI:20 review component tests passed after observed behavior/lifecycle failures. Navigation observed missing entry and lost completed preview;7 navigation tests passed. CSS bundle assertions observed missing styles and then passed. Sidebar heading scope corrected by source inspection; no visual walkthrough claim. UI commit7b5e32c.
- Full repository run:3950 passed,2 explicitly opt-in real-corpus skips, exit0. Includes CLI422 and Obsidian1007. Workspace typechecks, boundaries, product inventory and root build passed. Lint0 errors/22 existing permitted warnings.
- Final DSH build and20 actual isolated Host tests passed. Claude copied-package/workbench tests7 passed.34 relative documentation links checked. No real models, emails, user corpus writes, Computer Use or Electron walkthrough.
- Local linux/x64 package: extensions/dsh-arxiv-daily/dist/dsh-arxiv-daily-0.1.17.tgz; sha256 7daeeecadb2802534f746bf187b17f531b9d884cea5fc3231657c7ce4838f294. Manifest, README and CLI bytes checked against built package. Documentation/version commit41a4857.

## Local adjustments and limits

Read-only views consume atomic committed store snapshots while model work continues; only search/index/mutations require the workflow lease. Direction generation can renew disclosed authorization independently of indexing. No model work starts merely by opening the view. Accepted directions remain ordinary topic settings and candidate receipts prevent resurrection.

PDF serving is bounded to25MiB and uses the browser/host PDF viewer. Workbench reading and review are implemented; Markdown editing and richer P4 run management remain outside this phase. Changes stay on the feature worktree; no main merge, push, publication or production installation in this phase.

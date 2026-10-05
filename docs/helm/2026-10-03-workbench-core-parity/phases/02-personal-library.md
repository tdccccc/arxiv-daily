# P2 — Personal library workflows

goal_ref: ../goal.md
created: 2026-10-03T12:38:19+08:00
updated: 2026-10-05T12:49:49+08:00
revision: 2

## Outcome

用户可在 DSH 工作台连接自己的文献目录，审核处理披露并授权，运行准备、扫描、索引及检索。

## Assumptions

- 复用 runCliLibrary 与 connect/authorize/revokeCliLibrary，不复制 LibraryWorkflow。
- 文献库文件保持原目录；授权沿用已定义的深度、端点和fingerprint。

## Approach

连接/授权/索引设置已在P5/P14完成，保留现有入口。以0.5.0原生流程为准补网页主区域浏览、检索和PDF打开；通过共享CLI应用上下文调用core，不复制索引或检索业务。P3随后接入同一工作区的方向审核。

## Chunks

### Chunk 1 — Library application adapter

- change kind: behavior change; existing context extraction is behavior-preserving
- strategy: Green CLI library baseline for context export; strict Red-Green for structured workbench service
- Red signal: new workbench-library tests require missing browse/search/PDF service; then observable catalog/filter and unsafe-path failures
- Green check: temporary-library fixtures verify catalog pagination, title/abstract retrieval, root identity, PDF paper-key-only access and current config checks
- regression checks: cli-library, cli-library-config, CLI typecheck
- [ ] implementation and tests accepted

### Chunk 2 — HTTP and workbench library view

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red signal: HTTP catalog/search/PDF endpoints absent; DOM cannot navigate to personal library
- Green check: actual temporary HTTP server and DOM contracts for browse/search, empty/error/loading states, stale responses, PDF links, settings and task navigation
- regression checks: workbench navigation/UI/settings/run tests, all CLI tests and typecheck, build and isolated DSH Host
- [ ] implementation and tests accepted

## Phase verification

No real user corpus/model/email; temporary fixtures and stub parser/embedding only. Reuse current scoped source, config revisions and authorization. In-app library navigation participates in existing back/forward history. English and Chinese UI supported. Independent chunks may be delegated; owner alone edits Helm and commits.

## Abort / reshape triggers

If existing app service cannot return structured values safely, factor a shared host adapter under Green CLI baseline; do not parse terminal logs. If model access scope changes, use existing authorization disclosure. P3 remains pending until P2 accepted.

# P16 — Rebase onto main

goal_ref: ../goal.md
created: 2026-10-05T01:53:35+08:00
updated: 2026-10-05T02:15:06+08:00
revision: 2

## Outcome

Rebase当前隔离分支到origin/main 7e1774b，保留主线文献库/设置/可靠性改进及DSH共享设置/P15状态，验证后提供重新构建的DSH包。

## Assumptions

- 用户明确授权rebase。备份backup/claude-code-research-plugin-pre-rebase指向eafe972。
- 仅修改当前worktree；不改main、不push、不安装生产插件、不用Computer Use。
- 行为保持型整合：Green基线→冲突语义整合→Green；整合暴露的实际缺陷用已有失败回归或新增Red→Green。

## Chunks

1. [x] 保存备份并确认基线：CLI配置/业务设置/日历20、Obsidian设置/onboarding54、core源/状态30，104项已通过。
2. [x] 逐提交rebase；检查重叠设置、源适配器、宿主服务和测试，不能用全局ours/theirs替代整合。
3. [x] 回归各端设置/状态/文献库，typecheck、boundaries、inventory、构建和DSH隔离Host；检查main是当前HEAD祖先与工作区干净。

## Abort / reshape triggers

- 不确定行为取舍时先查双方提交/测试，不丢弃功能；若出现目标冲突则记录并报告。
- 备份保留；不使用skip绕过未解决的业务冲突。

## Acceptance

- Initial target7e1774b rebased successfully; origin/main advanced during execution with release-budget PR55, so a second clean replay incorporated c2de350. Final origin/main is an ancestor of this branch, with zero upstream-only commits. Backup remains eafe972 at backup/claude-code-research-plugin-pre-rebase.
- Range-diff against the original77 commits shows67 unchanged patches and10 adapted patches, none dropped. Conflict resolution retained main's topic/direction schema, atomic topic acceptance, native settings layout, download progress/cancellation and bounded PDF parsing while preserving core extraction and P15 outcomes.
- Main ADR0012/13/14 retired standalone interest profiles and moved library understanding to titles/abstracts. The CLI now accepts proposed directions into normal topic settings with atomic TOML receipts; old profile commands return migration guidance. Local library indexing never probes or sends PDFs to document sidecars. DSH preserves all direction IDs/origins and supports maxDailyPapers; config reads derive stable IDs for legacy directions.
- Observed pre-rebase Green104 tests. Integration failures exposed removed APIs, missing extractor provenance, stale settings fixtures/metadata and sidecar selector mismatch. Focused behavior Red→Green covered settings roundtrip/cap, atomic acceptance persistence, stable config IDs and no sidecar requests. Historical Saturday cap fixtures moved to Monday to continue testing ranking under P15; packaged arXiv fixture updated to the main direction filter response contract.
- Green: CLI339 tests/38 files; Obsidian1007/50; core focused277 plus main daily-cap6 and settings/topics/filter187, workflow/proposal/acceptance95. Core/CLI/plugin typechecks, boundaries, product inventory and builds pass. DSH20 (zero skipped, actual isolated configured/first-run Host), copied Claude package7 and updated native-workflow5 pass.
- Produced extensions/dsh-arxiv-daily/dist/dsh-arxiv-daily-0.1.14.tgz, linux/x64, sha1 0fe441c8a72d692e46d7c1f91e33afb2f6ab0b9c. Rebuilt the Claude CLI bundle too. No production installation, main checkout, user vault or remote branch was modified; no paid model/email/Computer Use.
- Checks not run: entire core suite, native cross-platform matrix, Electron visual acceptance. These are not claimed by the focused/DOM/actual Host checks above.

Post-replay integration commit2dd61aa; command docs61d1c7a; packaged acceptance22430c6; local releaseb57e7f6. P2/P3/P4 remain pending outside this rebase request.

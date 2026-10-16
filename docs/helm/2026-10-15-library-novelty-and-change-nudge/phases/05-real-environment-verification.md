# P5 — real-environment-verification

goal_ref: ../goal.md
created: 2026-10-15T23:30:00+08:00
updated: 2026-10-15T23:30:00+08:00
revision: 1

## Outcome

在隔离的真实 Obsidian、真实 CLI 与真实模型上确认：变化提醒与只扫新增可用、结果合理；旧日报可读；固定验收全过。

## Chunks

### Chunk 1 — 真实环境检查（子代理，/tmp/p5/report.md）
- change: 固定验收（Obsidian 8、workbench 12、工具测试 420）；新增独立脚本 scripts/acceptance/obsidian-library-change.mjs 驱动真实 Obsidian 走完提醒→弹窗→确认→只发新论文→再索引无提醒→2 篇无提醒但弹窗可更新；旧日报（v1/v2 标记+命中方向）在 Dashboard 正常、无警告；真实模型 2 次调用评估只扫新增。
- kind: non-behavioral（验证）+ 验收工具
- first signal: n/a
- checks: 见报告；截图在 /tmp/p5/lib-change-6/。
- [x] accepted on observed evidence

### Chunk 2 — 发现 A：只点「索引」看不到新加入的文件
- change: 用户选 A（2026-10-15）：插件「索引」与 CLI `library index` 在建索引前做一次本地文件清单比对（不联网），文件夹中可识别文件集合与上次扫描不同（新增/删除/改名）时先重新扫描，且只对新文件做联网元数据查询；文件夹未变时行为与现在完全一致（不联网）。
- kind: behavior
- first signal: 插件 library-build-index-scan-first 与 CLI cli-library 新用例：已扫描过的库新增 PDF 后只调用索引，新文件被收录且提醒出现（当前 Red）；文件夹未变时不触发扫描与联网。
- checks: 同文件 Green；fulltext-index-lifecycle、library 相关 core 测试；typecheck；真实 Obsidian 独立脚本改为「放入 PDF→只点索引」仍得到提醒。
- [x] accepted on observed evidence

### Chunk 3 — 观察 B：更新结果中显示被既有方向覆盖的数量
- change: 只扫新增的提案结果页（插件弹窗与工作台）显示一句「其中 N 篇已被现有方向覆盖」，N=0 不显示。
- kind: behavior
- first signal: 插件弹窗与工作台 UI 测试新用例（当前 Red）。
- checks: 同文件 Green；typecheck；lint。
- [x] accepted on observed evidence

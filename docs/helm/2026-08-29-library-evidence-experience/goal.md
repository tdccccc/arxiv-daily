# 文献证据结果可用（library evidence experience）

status: active
updated: 2026-08-29
owner: claude-code-main-session

## Intent

让个人文献库的搜索和相似论文结果真正能用：点开能到 PDF 对应页，证据片段读得像论文而不是噪声，界面不跟日常阅读列表抢注意力。

## Success criteria

- [ ] 在 Vault 内的个人文献库上，Dashboard「Library matches」和 Similar papers 的「Open PDF at page N」能打开嵌入式 PDF 到该页；换一页会落到另一页。
- [ ] 打开失败时，通知里能看到具体原因，而不是只写 `Open PDF at page N failed`。
- [ ] 证据结果能一眼看到论文标题、可用的章节、一段可读正文和页码；版权页、目录点线、参考文献列表不再作为默认主证据。
- [ ] Dashboard 与 Similar papers 共用同一套证据呈现；不改变混合检索的论文排序。

## Non-goals

- 真按篇增量重建 generation 索引，或继续抠构建吞吐。
- PDF 坐标高亮、改解析器/sidecar、改 BM25/RRF 论文排序。
- 重做整个 Dashboard 或日报阅读列表。
- 问答或 Agent。

## Constraints

- 不扩大 `ScopedLibrarySource` 权限面；打开前仍须确认该 PDF 属于当前索引。
- Core 保持 host-neutral。
- 不修改并行 active Helm（发现闭环、email relay v2）的状态。
- 每个行为块有 Red/Green 证据；Vault 内打开以真实 Obsidian 隔离会话验收。

## Phases

1. P1 — Vault 内证据 PDF 能打开到对应页，失败原因可见 — status: active
2. P2 — 证据卡片可读：正文、章节、页码和打开动作层次清楚 — status: pending
3. P3 — 默认主证据不再是版权页、目录点线或参考文献列表 — status: pending

## Open questions

- P3 是只改展示（藏噪声 hit），还是也改每篇保留哪些 hit。计划 P3 时用真实结果决定，不在目标层锁死。

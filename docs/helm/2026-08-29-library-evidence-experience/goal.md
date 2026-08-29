# 文献检索结果可用（library evidence experience）

status: active
updated: 2026-08-29
owner: claude-code-main-session

## Intent

让个人文献库搜索和相似论文给出能用的论文列表：按相关性排序，一点就能打开对应 PDF。段落摘录和页码跳转在没有评测之前不作为产品承诺。

## Success criteria

- [x] 在 Vault 内的个人文献库上，Dashboard「Library matches」和 Similar papers 能打开对应 PDF；打开失败时通知带具体原因。
- [x] Dashboard 与 Similar papers 共用同一套结果呈现；不改变混合检索的论文排序。
- [ ] 结果只展示相关论文（标题、文件名、打开 PDF）；不展示段落、章节或页码。
- [ ] 打开动作打开整份 PDF，不声称跳到某一页。

## Non-goals

- 真按篇增量重建 generation 索引，或继续抠构建吞吐。
- 段落级证据质量、版权页/目录/参考文献过滤、PDF 坐标高亮、改解析器/sidecar、改 BM25/RRF 论文排序。
- 重做整个 Dashboard 或日报阅读列表。
- 问答或 Agent。

## Constraints

- 不扩大 `ScopedLibrarySource` 权限面；打开前仍须确认该 PDF 属于当前索引。
- Core 保持 host-neutral；检索内部仍可保留 hit，产品面不展示。
- 不修改并行 active Helm（发现闭环、email relay v2）的状态。
- 每个行为块有 Red/Green 证据。

## Phases

1. P1 — Vault 内证据 PDF 能打开到对应页，失败原因可见 — status: done
2. P2 — 证据卡片可读：正文、章节、页码和打开动作层次清楚 — status: done
3. P3 — 默认主证据不再是版权页、目录点线或参考文献列表 — status: superseded
4. P3b — 产品面只列出相关论文并打开整份 PDF — status: active

## Open questions

- （已关闭）P3 噪声 hit 选择：用户选择不在没有评测时把半成品段落端出去，改为不展示段落。

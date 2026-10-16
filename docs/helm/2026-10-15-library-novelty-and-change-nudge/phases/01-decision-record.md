# P1 — decision-record

goal_ref: ../goal.md
created: 2026-10-15T10:18:36+08:00
updated: 2026-10-15T10:18:36+08:00
revision: 1

## Outcome

新 ADR 0015 记录两项决定，ADR 0004 / 0007 / 0011 / 0012 的相关段落标明被取代或收口，`CONTEXT.md` 术语与之一致。

## Assumptions

- 目录里最大的 ADR 编号是 0014（0009 有两份重号，不影响新编号）。
- ADR 正文为英文，沿用现有 ADR 的结构（Status / Related / Context / Decision / Consequences）。

## Approach

写一份 ADR 0015 覆盖两项决定；对旧 ADR 只改 Status 行或加一句指向 0015 的说明，不重写正文；CONTEXT.md 改 Personal novelty、Direction suggestion、Research signal 三条。

## Chunks

### Chunk 1 — ADR 0015 与旧 ADR 状态

- change: 新增 `docs/adr/0015-*.md`：(a) novelty 对比基准=每篇日报论文按标题+摘要按需检索的最相似库论文（上限 5 篇），覆盖全部日报论文，需文献库处理授权，模型可判定无可比论文则不写，不设相似度下限；(b) 取代 ADR 0007：建索引后本地计数提醒（不调模型、无需授权），研究者手动触发只扫新增论文的重扫，沿用 ADR 0014 的覆盖/新增划分与复审；不做 split/merge/lock，自动触发不恢复。0007 Status 标 Superseded；0012 Consequences 的 novelty 未决项与 0004 §4 加指向 0015 的注；0011 中对 0007 增量建议的引用加注。
- kind: non-behavioral
- first signal: n/a
- checks: 人工通读；`grep -n "0007\|0015" docs/adr/*.md` 交叉引用一致
- [x] accepted on observed evidence

### Chunk 2 — CONTEXT.md 术语

- change: Personal novelty 写明基准来源；Direction suggestion 改为描述「库变化提醒 + 只扫新增」；Research signal 删除「representative sets are authoritative signals」的过时说法。
- kind: non-behavioral
- first signal: n/a
- checks: 人工通读；与 ADR 0015 用词一致
- [x] accepted on observed evidence

## Phase verification

- ADR 0015 与 CONTEXT.md 对同一概念用同一术语；旧 ADR 不再声称已退休的机制仍有效。

## Abort / reshape triggers

- 若写 ADR 时发现现有代码与决定冲突（例如重扫无法只送新增论文），记入 P3 规划而不是改决定；若冲突动摇决定本身，L3 回到用户。

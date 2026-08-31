# 文献库方向驱动日报（library directions drive the daily）

status: active
created: 2026-08-31T22:05:18+08:00
updated: 2026-08-31T23:04:00+08:00
revision: 4
owner: claude-code-main-session

## Intent

让从个人文献库总结出的研究方向真正驱动 arXiv 日报推送：确认一次就能开始推，此后画像跟着库变化自动更新、有变动就提醒复核。这条链的每一段都已存在，但从未端到端跑通过一次。

## Success criteria

- [ ] 方向候选在给研究者看之前已跨簇综合：讲同一件事的不再各占一条，测试库上 22 条收敛到个位数（ADR 0009）。
- [ ] 方向候选可以一次成组接受/排除并一次落盘，不必逐个确认；证据单薄的候选默认不勾。确认后 confirmed interest profile 非空。
- [ ] 手写主题为 0、已确认方向 ≥ 1 时，日报正常运行；两者皆无时仍拒绝且说清原因（ADR 0010）。
- [ ] 待复核的条目数量与入口出现在 Dashboard 上，不必记得去开命令面板。
- [ ] 在测试库上产出一篇真实日报，其中含由文献库方向选出的论文并标明 discovery source。

## Non-goals

- 自动确认方向 / 默认放行。沿用 ADR 0007 §3：机器建议进复审队列，永不自动生效。用户 2026-08-31 明确选择「等确认完才推」。
- 一键接受后再提醒研究者回头细看。用户 2026-08-31 决定：接受了就一视同仁，只有库变化产生新建议时才再问。
- 综合步骤丢弃候选。综合只合并、只标注，不删除（ADR 0009 §2）。
- 取消「方向」这层中间产物、改成直接拿库做相似度推送（2026-08-31 评估后未采纳）。
- 改日报生成格式。库方向不成为日报章节；只有一个 `Library-guided discoveries` 平铺章节，来源写在每篇条目里（ADR 0010 §3）。
- 重做增量更新的触发机制——ADR 0007 已定「建索引完成且 indexed > 0 时触发」且已接线，本 goal 只让它的产出被看见。
- 改检索排序、段落证据。
- 把 reading candidates 加回来（ADR 0011）。

## Constraints

- ADR 0004 / 0005 / 0007 / 0008 不改。授权模型、consent 分割、指纹语义一律沿用；综合步骤走既有的 `personal-library-direction-generation` 授权与取消范围，不新增授权面。
- CLI 产品没有文献库，前置检查放宽后 CLI 行为必须一字不变。
- 端到端验证需要 LLM 端点可达；当前 `100.124.147.116:8081` 从本机不可达，属环境阻塞，需用户先弄通。综合步骤的实现与单测不受阻，但要看到收敛效果必须重跑一次方向生成。
- **Dashboard 未进过桌面验收，单测里视图也不实例化**。P4 的改动「测试绿」不构成交付证据，必须由用户在真实 Obsidian 里看过。
- 工作区为 worktree `chore/drop-reading-candidates`（PR #43 未合）。不提交、不推送、不开 PR，除非用户显式指令。
- 不修改并行 active helm：`2026-08-11-email-relay-v2-production-cutover`、`2026-08-27-obsidian-desktop-acceptance-harness`、`2026-08-13-discovery-loop-and-library-insight`。

## Phases

<!-- Single source of truth for phase status. PN ↔ filename NN. -->
1. P1 — 方向候选生成时跨簇综合，同义方向不再各占一条 — status: done
2. P2 — 候选可一次成组确认，证据单薄的默认不勾，确认后画像非空 — status: pending
3. P3 — 已确认方向可独立满足日报的前置检查 — status: pending
4. P4 — 待复核的条目数量与入口出现在 Dashboard 上 — status: pending
5. P5 — 端到端跑出一篇带文献库来源的真实日报 — status: pending

## Open questions

- P5 依赖 LLM 端点可达，排期由用户决定。
- 日报是否按方向分章：现决定不分（ADR 0010 §3），待 P5 跑出真实报告、看过实际长度与可读性后再定是否重开。

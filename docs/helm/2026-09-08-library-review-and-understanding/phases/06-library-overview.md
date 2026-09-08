# P6 — library-overview

goal_ref: ../goal.md
created: 2026-09-08T08:22:37+08:00
updated: 2026-09-08T08:27:00+08:00
revision: 2

## Outcome

用户能在文献库概览持续查看本次分析的主题、方向、覆盖、未覆盖论文并打开证据；零新增依然有可读结构。

## Assumptions

- 基于当前提案与设置投影概览，不恢复已退休画像；清楚显示分析日期及已知证据边界。
- 主题的一句总结由其方向组成，不新增未验证模型结论。
- 未覆盖列表按需展开，默认保持一行数量。
- 打开论文复用现有 scoped PDF opener，保留路径/库身份校验。

## Approach

复审窗口增加 Overview 页签，按有效已有覆盖与候选归属组织主题段落；已接受后改写/删除的方向不展示为当前事实。展示已分析/当前索引/未纳入数量、分析日期，以及每方向论文列表。统一证据标题按钮用于候选、覆盖及未覆盖；页签切换保留草稿并支持键盘。

## Chunks

### Chunk 1 — 概览与证据浏览

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- files: plugin review modal/main/styles; DOM/controller tests。
- Red signal: 全覆盖零新增仍看到主题、方向和论文；未覆盖可展开；点击证据进入现有PDF opener；页签切换保留草稿。
- Green check: proposal-acceptance-ui、interest-profile-modal、library-generation-evidence-guard tests。
- [x] implementation and tests accepted — 2概览DOM Red + PDF controller Red；plugin88项通过，四包typecheck通过。

## Phase verification

- plugin/core回归、typecheck；P7真实Obsidian宽窄布局、点击及渲染核验。

## Abort / reshape triggers

- 无证据文字被称作库事实或删除方向被复活：拒绝，显示历史/变更状态。
- 证据打开绕过现有本地范围校验：拒绝。

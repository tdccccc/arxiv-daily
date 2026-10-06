# P6 — Settings navigation

goal_ref: ../goal.md
created: 2026-10-03T16:37:51+08:00
updated: 2026-10-03T16:40:24+08:00
revision: 2

## Outcome

保留已验收的Obsidian条目与控件，左侧导航点击跳转，右侧完整滚动设置，分组视觉清晰。

## Assumptions

- 导航不改为隐藏面板切换，不重建表单或丢弃草稿。
- 窄屏导航改为顶部横向导航，宽屏左栏与正文分别滚动。

## Chunks

- behavior change；strict Red→Green：DOM测试导航标签、点击滚动、当前区域高亮、不提交、不丢草稿；回归设置源码对照和保存/授权测试。
- CSS presentation：比例验证采用结构与构建检查；遵守用户不用Computer Use，无视觉自动化。

## Verification

- settings navigation/UI/setup tests、CLI typecheck、build/pack和DSH回归。

## Abort / reshape triggers

- 如果改变原条目顺序、控件或保存行为，收回到仅导航层。

## Acceptance

- a9edb6d: navigation no-op tests observed two behavioral Red failures, then Green. Moving existing controls retains values/listeners, and jumping does not save or hide sections.
- 34 UI tests across 5 files pass, including Obsidian source parity and settings persistence. 12 DSH component/registry tests pass. Typecheck, boundaries, product inventory, diff check and build/pack pass.
- CSS uses a fixed-height dialog, independently scrolling nav/content, title backgrounds and section borders, fixed save actions, and a top navigation at narrow widths. No Computer Use or live visual walkthrough, full CLI/Host rerun not required for this isolated presentation/navigation change.
- Packaged linux/x64 DSH 0.1.6; installation/restart left to user.

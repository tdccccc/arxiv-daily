# P12 — Stable setup guide during model fetch

goal_ref: ../goal.md
created: 2026-10-04T02:21:14+08:00
updated: 2026-10-04T02:22:20+08:00
revision: 2

## Outcome

Get models保存当前配置但不重新创建Getting started，不改变设置页滚动位置。首次引导显式步骤仍可刷新进度。

## Strategy and evidence

Bug fix / strict Red→Green：先复现已有引导DOM被替换、原先完成引导再次插入两条失败，再把静默保存与引导刷新分开。后台完成通知只刷新已可见的引导，重复完成通知去重。

## Verification

- 20 focused tests覆盖settings/guide/model/nav，typecheck通过；继续DSH组件、build/pack。
- 不改变设置字段或模型请求内容；无Computer Use。

## Abort / reshape triggers

- 修复不能禁用真正的首次引导或阻止模型配置保存。

## Acceptance

- Both reported guide replacement/reappearance tests observed Red then Green. Existing guide generation and model control tests pass: 20 focused tests across4files.
- 12 DSH component/registry checks, typecheck, boundaries, inventory, diff check, build/pack pass. Backend unchanged, no full Host rerun necessary.
- Installed0.1.10 to existing Web profile and verified matching bundle hash. User session left running; restart/refresh needed. No Computer Use or live provider requests.

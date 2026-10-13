# Exploration safety and coverage

status: active
created: 2026-10-13T10:05:00+08:00
updated: 2026-10-13T10:05:00+08:00
revision: 1
owner: main-session-3a9f449b

## Intent

补上模型驱动发版探索的安全漏洞，并把探索覆盖扩展到尚未被模型操作过的功能，尤其是个人文献库，使发版前的探索报告既守住"只能安全操作本插件界面"的边界，又能在更多真实交互上发现问题。

## Success criteria

- [ ] 两端的键盘按键动作和点击一样受危险操作拦截：焦点落在外部链接、调度开关、清除运行状态、密钥显示等受限或敏感控件上，或焦点不在本产品界面内时，按键不发送；固定测试覆盖这些情形。
- [ ] Obsidian 端截图与 workbench 一样遮住密钥字段；固定测试覆盖。
- [ ] Dashboard "More" 菜单的 "Retry failed dates" 与 "Run pending" 各有带程序判定的目标任务，并有"未操作时判定不通过"的测试。
- [ ] 目标任务依赖的界面文字在产品中找不到时，结果为 blocked 且原因明确写出"任务过期"及缺失的文字，不再表现为步数耗尽；固定测试覆盖。
- [ ] 个人文献库在 Plugin product 和 Reading workbench 上各有一组带程序判定的目标任务（覆盖范围在对应阶段确定），起点由夹具确定性准备，不使用系统文件选择框，不读写真实文献库。
- [ ] 至少一次真实模型完整运行取得证据（运行前征得用户同意费用）；疑点逐条复核处置；发版执行说明更新任务数量与成本。

## Non-goals

- 调整用户角色的步数预算，或新增程序侧的疑点自动发现机制。
- 让模型能使用 Obsidian 命令面板；仅能从命令面板触达的命令不在本次范围。
- 为 `scripts/` 新增常驻 lint / typecheck。
- 作为自动发版门槛；普通固定验收和 CI 仍不调用模型 API。
- 修改其他 Helm 计划的状态。

## Constraints

- 分支 `fix/exploration-safety-coverage`，worktree `/home/tiandc/Documents/code/arxiv-daily-exploration-coverage`。
- 复用 `scripts/acceptance/exploration*.mjs`、`exploration-tasks/*`、`scripts/desktop-acceptance/*` 现有的循环、动作校验、受限控件分类、夹具与重置；不复制平行实现。
- 普通固定验收和 CI 不调用模型 API、不读取真实密钥；文献库夹具只在临时目录中构造。
- 真实模型运行会产生费用，每次运行前先征得用户同意。
- 实现块尽量交给 subagent；owner 验收后按独立意图提交；不推送，推送与 PR 需用户逐次同意。
- 起因与证据：2026-10-13 审查报告（harness / task library / offline run），要点记录在 journal。

## Phases

<!-- Single source of truth for phase status. PN ↔ filename NN. Integer IDs are permanent and never reused; replacement phases take the next unused number. Outcomes only — no steps. The active line is the current focus. -->
1. P1 — 两端按键动作受与点击相同的受限/敏感/范围拦截，Obsidian 截图遮住密钥字段 — status: active
2. P2 — "Retry failed dates" 与 "Run pending" 有目标任务；界面文字缺失时任务结果明确为"任务过期" — status: pending
3. P3 — Plugin product 上的个人文献库目标任务就绪，起点由夹具确定性准备 — status: pending
4. P4 — Reading workbench 上的个人文献库目标任务就绪 — status: pending
5. P5 — 真实模型完整运行、疑点复核与处置、发版说明更新 — status: pending

## Open questions

- 审查中"只要有一个非等待动作即可通过"的门槛是否可能借前一任务残留状态误判通过（未找到具体实例）；P2 或 P3 补任务时顺带确认，确认无实例则在 journal 记录后关闭。

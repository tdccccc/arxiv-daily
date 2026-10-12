# Model-driven release exploration

status: done
created: 2026-10-10T22:27:32+08:00
updated: 2026-10-12T23:30:00+08:00
revision: 5
owner: main-session-3a9f449b

## Intent

每次发版前，用一条命令让模型 API 扮演用户，在真实 Plugin product（Obsidian）和 Reading workbench 中执行一组任务并自由探索，产出可复查的报告，用来发现固定场景想不到的交互问题。

## Success criteria

- [x] 一条命令运行完整探索任务库：Obsidian 10–12 个目标任务、workbench 新增 2–3 个目标任务、2–3 个用户角色自由探索；支持只运行指定任务或角色。
- [x] 每个目标任务由独立程序检查判定 passed / failed / blocked，模型自称完成不算通过；单个任务失败不阻止其余任务，任务之间恢复到确定起点。
- [x] 自由探索只产出“待复核”疑点及证据（步骤、截图、观察），不产生通过结论；报告与目标任务结果分开展示。
- [x] 模型只能观察和操作本插件自己的界面；发送邮件、清除运行状态、调度开关、系统文件对话框、外部链接、密钥字段均不可操作；外部请求全部走固定响应，不读写用户 Vault 或资料。
- [x] 每任务及整次运行均有步数、API 调用和时长上限；报告记录实际 token、调用数和耗时。
- [x] 至少一次真实模型完整运行取得证据；疑点逐条复核，确认的问题转为修复或固定回归；发版执行说明（命令、看报告、成本）可照文档复现。

## Non-goals

- 作为自动发版门槛（只出报告）。
- 评价论文筛选相关性或摘要质量。
- 替代固定验收场景或既有单元/集成测试。
- Linux 以外平台、多个 Obsidian 版本。
- 修改 `2026-08-27-obsidian-desktop-acceptance-harness` 或其他 Helm 计划的状态。

## Constraints

- 分支 `test/model-exploration`，worktree `/home/tiandc/Documents/code/arxiv-daily-model-exploration`。
- 复用 `scripts/acceptance/exploration*.mjs`、`exploration-model.ts` 的循环、动作校验、预算、脱敏，以及 `scripts/desktop-acceptance/*` 的启动、隔离、诊断和截图；不复制平行实现。
- 普通固定验收和 CI 不调用模型 API、不读取真实密钥。
- 实现块尽量交给 subagent；owner 验收后按独立意图提交；不推送，推送与 PR 需用户逐次同意。

## Phases

<!-- Single source of truth for phase status. PN ↔ filename NN. Integer IDs are permanent and never reused; replacement phases take the next unused number. Outcomes only — no steps. The active line is the current focus. -->
1. P1 — 探索框架能连续运行一组相互隔离的目标任务和自由探索角色，可按任务选择并按任务计预算（先在 workbench 上用受控模型验证） — status: done
2. P2 — 模型可在真实 Obsidian 中只观察和操作本插件界面，危险操作被拦截，受控与真实模型各跑通简单任务 — status: done
3. P3 — Obsidian 与 workbench 任务库及用户角色就绪，每个目标任务有程序判定 — status: done
4. P4 — 真实模型完整运行、疑点复核与处置、预算校准和发版执行说明完成 — status: done

# P5 — Obsidian settings parity

goal_ref: ../goal.md
created: 2026-10-03T12:53:53+08:00
updated: 2026-10-03T12:53:53+08:00
revision: 1

## Outcome

工作台设置按 Obsidian 1.13+ definitions/declarative-rows 主路径呈现全部设置，控件保存后有真实业务效果。

## Assumptions

- 用户要求现有 Obsidian 条目为准，采用当前主路径；legacy差异不混入。
- 额外 vault root 仅作为独立宿主上下文，不改分组。
- 不回显旧密钥；Show仅显示新输入。

## Approach

保留P1的首次启动、安全写入和热更新。扩展缺少的detail/sidecar/应用内schedule配置，复用shared scheduler和library应用服务。UI按主路径顺序呈现，读真实定义做结构对照测试。

## Chunks

1. 配置映射和保存：behavior change；strict Red/Green；新增round-trip tests先证实缺字段，然后通过并回归CLI config/settings。
2. UI条目与控件：behavior change；strict Red/Green；对照Obsidian主路径的heading/row/select options，真实保存回读；回归所有workbench UI。
3. 宿主操作：behavior change；strict Red/Green；模型列表、library连接/授权/build/revoke、邮件显式按钮和分钟级调度；模拟外部服务，不发送真实邮件。

## Phase verification

- DOM与原定义对照、设置round-trip、模拟HTTP与调度，完整CLI及DSH Host回归、build/pack。
- 不将无法工作的控件宣称可用；不更改真实用户定时/文献数据。

## Abort / reshape triggers

- 外部cron语义不能替代Obsidian应用内检查间隔。
- 若需要另造library业务逻辑，改为共享服务适配。

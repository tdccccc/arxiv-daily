# P5 — Obsidian settings parity

goal_ref: ../goal.md
created: 2026-10-03T12:53:53+08:00
updated: 2026-10-03T13:24:49+08:00
revision: 2

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

## Acceptance evidence

- Configuration chunk accepted in 5c197c6: missing-field/custom-profile Red → complete round trips; preserved cron/custom thresholds/secrets and invalidated consent after endpoint expansion.
- Host chunk accepted in 4691720: missing action HTTP Red, missing scheduled command Red, timer no-op Red, late cancellation Red → Green. Shared core retains library, email, weekday/window/completion behavior.
- UI/package chunk accepted in cf1cd8e: source-definition parity Red, control/revision/busy-state Red and old 0.1.4 Host persistence Red → Green. All conditional groups and declaration dropdown options match the Obsidian 1.13+ definitions; callback controls separately tested.
- Final CLI suite: 30 files, 261 tests pass. DSH suite: 20 tests pass, none skipped; real isolated Host covers existing and new configs, extended settings persistence, no secret projection, mock daily generation and lifecycle.
- Typecheck, boundaries, product inventory, diff check, build and pack pass. Browser build needed a Markdown text loader when consuming shared core exports; observed build failure corrected and rebuilt.
- DSH linux/x64 package: extensions/dsh-arxiv-daily/dist/dsh-arxiv-daily-0.1.5.tgz.
- Host adaptations are explicit: extra root folder outside original sections; Choose folder uses a path dialog in Web; saved secrets remain write-only, Show reveals only newly entered values; settings use explicit Save; scheduler runs only while workbench process lives. No system cron installed.
- No real provider/email sends, no Computer Use, and no Electron visual walkthrough performed. Real library parsing/indexing remains covered by existing CLI workflow tests plus mocked settings-adapter integration, not a new live user-library run.

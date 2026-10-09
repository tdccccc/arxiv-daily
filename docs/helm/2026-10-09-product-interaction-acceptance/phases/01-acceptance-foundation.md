# P1 — acceptance-foundation

goal_ref: ../goal.md
created: 2026-10-09T21:39:29+08:00
updated: 2026-10-09T22:20:17+08:00
revision: 2

## Outcome

两端共享可控外部接口和隔离数据，真实界面场景与模型探索能被统一运行并产出诚实的证据报告。

## Assumptions

- 已有 desktop session/CDP 与已打包 CLI 足以复用，无需增加产品测试开关。
- 固定 arXiv/LLM/embedding 响应可贯穿真实 scheduler、pipeline、取消与文件存储。
- 业务模型夹具和探索模型 API 分开配置；模型只能操作被测本地界面。
- 本机 Chrome、Obsidian、Xvfb 可用于真实验证；缺失时报告 blocked。

## Approach

父会话维护隔离数据、fixture server、统一报告和入口；工作台、Obsidian、模型探索分别作为同一阶段的独立模块。使用当前产品构建和真实用户入口，固定外部传输行为。未来 P2 只保留结果目标，此阶段不预写其详细任务。

## Chunks

### Chunk 1 — shared fixtures and evidence runner

- change kind: behavior change (test infrastructure)
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 新 Node contract tests 对尚不存在的 fixture/报告公共契约失败；建立可导入最小表面后观察正常/取消/未知请求/空结果判定的行为 Red。
- Green check: node --test scripts/tests/acceptance-fixtures.test.mjs scripts/tests/acceptance-report.test.mjs
- regression checks: 原有 desktop harness 与 root runner tests；check:boundaries、check:product-units、git diff --check。
- [ ] implementation and tests accepted

### Chunk 2 — workbench browser journeys

- change kind: behavior-preserving test coverage; harness behavior change
- strategy: existing Green baseline for product, Red-Green for harness contracts; real UI assertions as compensating verification
- Red / baseline signal: 运行已有工作台设置、导航、运行 HTTP 测试；新 harness 的启动/错误分类契约先 Red，既有产品预期首次通过不制造 Red。
- Green check: 独立 fixture 下构建 CLI，运行工作台真实浏览器场景并检查实际 TOML、Markdown、索引与进程状态。
- regression checks: 相关 apps/cli Vitest suites；新增 harness Node tests。
- exception: 新增既有行为的真实 UI 覆盖采用 Green characterization，不篡改产品制造失败；对验收判据做负向对照。
- [ ] implementation and tests accepted

### Chunk 3 — Obsidian browser journeys

- change kind: behavior-preserving test coverage; harness behavior change
- strategy: existing desktop harness Green baseline; Red-Green for new session/fixture contracts
- Red / baseline signal: 既有 desktop tests Green；新 workflow runner 的缺失环境/未完成操作/错误报告契约 Red。
- Green check: 自有临时 vault 中启动真实 Obsidian，取得主流程、取消和失败恢复及截图/存储证据。
- regression checks: scripts/tests/desktop-acceptance-*.test.mjs；相关 Plugin tests。
- exception: 既有产品行为测试按 Green characterization；模型返回与原生目录选择仅在外部边界控制。
- [ ] implementation and tests accepted

### Chunk 4 — bounded model API exploration

- change kind: behavior change (test infrastructure)
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 本地伪模型服务验证观察/动作循环、非法动作、预算、模型错误、无证据完成声明均不能假绿。
- Green check: Node contract tests plus actual browser exploration with controlled model; configured real API smoke when configuration is available.
- regression checks: fixture/report contracts and browser scenario runner.
- [ ] implementation and tests accepted

### Chunk 5 — explicit retry after a permanent failure

- change kind: bug fix exposed by a real user journey
- strategy: reproduce at the workbench HTTP/CLI boundary, then Red-Green-Refactor
- Red / baseline signal: a date marked failed_permanent is skipped after clicking the workbench retry action even when the model configuration has been corrected.
- Green check: explicit user retry executes the failed date again and can commit a complete report; ordinary completed-date runs remain idempotent.
- regression checks: workbench actions/server tests, real browser authentication-failure recovery, existing scheduler and CLI invocation contracts.
- constraint: change the explicit manual retry path; preserve automatic scheduling's permanent-failure stop semantics.
- [ ] implementation and tests accepted

## Phase verification

- 统一入口能列出场景、执行所选套件并在失败或阻塞时非零退出。
- 所有宿主仅访问自有临时数据；普通业务模型/arXiv 请求可在 fixture server 中观察。
- 模型探索的单步记录与截图可对应同一次真实浏览器运行。
- Checkpoint 记录每个实际命令与结果；环境阻塞或没有执行的检查明确记录。

## Abort / reshape triggers

- 如果任何方案需要真实用户目录、真实邮件或改系统配置，先换成隔离夹具方案。
- 如果测试通过依赖于把 scheduler/pipeline 替换为成功返回，重做验证边界。
- 如果不同宿主无法共享低层 transport，保留共同协议与报告，拆分宿主接线。

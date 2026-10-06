# P14 — Shared business settings

goal_ref: ../goal.md
created: 2026-10-04T18:24:00+08:00
updated: 2026-10-04T19:18:06+08:00
revision: 2

## Outcome

模型、研究主题、邮件等业务设置从core共享字段描述、编辑规则和操作服务；Obsidian与工作台消费同一来源，保留各自存储/控件，不自动同步值。

## Assumptions

- 编辑阶段允许不完整草稿，执行阶段仍使用core就绪校验。宿主的CAS、事务回滚、输出存储切换和处理授权不能丢失。
- 以Obsidian当前主路径字段语义为基准，保留已有兼容字段与宿主能力差异。

## Chunks

1. Core setting schema: feature Red→Green，共享group/field/control/options/conditional metadata，对照原定义；host-neutral无Obsidian/DOM依赖。
2. Candidate edit normalization: feature Red→Green，模型/主题/邮件字段统一规范化与结构校验，保留草稿；两个宿主改用同一函数，回归事务/密钥/CAS/授权。
3. Shared settings operations: refactor Green baseline，核对模型/邮件既有core服务，消除测试邮件与验证组装重复；保持host展示格式和取消行为。
4. Host schema render adapters: refactor Green baseline后迁移，对照两端可见字段/选项/条件，保留特殊控件与独立UI外观组。

## Verification

- core共享规则测试、Obsidian settings/change-service/email测试、CLI配置与界面回归、跨宿主合同测试、typechecks、boundaries、build。
- 不触及真实Obsidian库和用户模型/邮件，不自动同步密钥。

## Abort / reshape triggers

- 如果共用实现需要宿主对象进入core，改为端口/数据描述。
- 如果迁移削弱授权/修订检查或事务持久性，先修复再验收。

## Acceptance

- fef7915: shared business schema/order/options/conditional descriptions, topic subfields/time menus, pure normalization and explicit email operations. New contract tests observed Red→Green; shared operations extracted under existing Green delivery baselines.
- 16c4007: current Obsidian definitions and workbench render adapters consume the schema; legacy Obsidian and CLI init reuse option catalogs. Both real edit adapters produce identical model/topic/email drafts and reject invalid endpoints without persistence. Existing custom reasoning retention observed Red→Green; option value escaping and accessible enable state verified.
- Final CLI36files299tests, Obsidian46files771tests, focusedcore34tests and DSH20tests pass. Core/CLI/Plugin typechecks, boundaries, inventory, both builds and packaging pass. No actual provider/email calls or Computer Use.
- DSH0.1.12 installed in existing Web profile; installed bundle hash matched dist. Existing sessions preserved pending user restart. Obsidian main.js built only in the worktree; live vault/plugin unchanged.
- Host boundaries remain explicit: TOML mapping/secret placeholder semantics/revision locks in CLI, native controls/transaction rollback/runtime-store swap in Obsidian. CLI verification stays on the official endpoint; Obsidian retains its configured endpoint override. No automatic value synchronization.
- Obsidian legacy complex editors mutate live topic drafts before persistCurrent; this path now validates known topic/detail fields but does not claim rollback to a pre-edit snapshot. Current explicit transactions retain existing rollback/CAS guarantees.

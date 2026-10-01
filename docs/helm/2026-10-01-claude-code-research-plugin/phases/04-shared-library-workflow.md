# P4 — Shared library workflow

goal_ref: ../goal.md
created: 2026-10-01T20:08:00+08:00
updated: 2026-10-01T20:08:00+08:00
revision: 1

## Outcome

独立 CLI 能调用既有文献库识别、索引、方向提议/确认、增量更新与个性化发现核心，作为基础日报的可选增强；没有有效文献库时保留手动主题路径。

## Assumptions

- 不复制聚类、检索、方向/profile 格式或新意判断算法；CLI 只补宿主适配和任务编排。
- 源文献保持只读；模型处理范围、远程 embedding 的全文传输和端点变化沿用已有授权语义。
- 基础 P3 已接受；新增库能力不能使未配置库的日报/详报失效。
- 本阶段先调查 PDF/embedding 和连接状态的实际宿主依赖，再选择可复用的应用服务边界；不得通过另建 Markdown profile 规避它们。

## Chunks

### Chunk 1 — Verify the reusable service and host boundaries

- change kind: non-behavioral investigation
- strategy: proportionate source/API inspection
- baseline: core 的 catalog/index/proposer/review/eligibility 与 Obsidian orchestration 现状；明确 Node 缺失的 PDF、embedding、连接与授权适配。
- check: 将可直接复用、需抽取、需新增适配的边界及测试命令记录在本阶段。
- [ ] boundary accepted

### Chunk 2 — Provide the Node library runtime and commands

- change kind: behavior change; any shared extraction first uses a behavior-preserving baseline
- strategy: shared Green characterization followed by strict Red-Green-Refactor for the CLI entry
- Red: CLI 库命令缺失/无法建立现有 catalog 与索引、方向 proposal/profile 契约；使用真实隔离 PDF fixture 和受控 HTTP/embedding ports。
- Green: 同一 core 处理识别/索引/方向提议/确认；持久化文档能通过现有 decoder/store 读取；源目录字节保持不变。
- regression: 原 Obsidian catalog/profile/fulltext 测试、Node adapter、CLI P3 全流程、typecheck/boundaries。
- [ ] implementation and tests accepted

### Chunk 3 — Feed confirmed directions into the existing daily workflow

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red: 有效方向未进入独立日报过滤或新意解释；无效/未授权方向不得进入模型请求。
- Green: 手动主题与有效方向组合；同一索引/profile/input 校验和证据约束生效；未配置库仍通过 P3 流程。
- regression: core personalized filter/novelty、incremental/review、现有每日恢复、CLI/Obsidian 测试。
- [ ] integration accepted

## Phase verification

- 真实本地文档经过 Node parser；不能用 agent 的抽样阅读冒充索引。
- 模型是有界业务步骤，agent 会话可缺席；方向确认和阅读处理的权威状态由既有 core 管理。
- 分别报告本地/远程模型实际可用性、fixture 证据与未运行项，不隐式改变用户全局设置或生产数据。

## Abort / reshape triggers

- 出现第二套 catalog/profile/index 或复制业务算法时回到共享服务边界。
- 宿主依赖暴露未决的模型/授权契约时先解决契约，不用 silently fallback 或虚假成功跨过去。

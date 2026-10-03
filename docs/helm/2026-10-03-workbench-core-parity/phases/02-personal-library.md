# P2 — Personal library workflows

goal_ref: ../goal.md
created: 2026-10-03T12:38:19+08:00
updated: 2026-10-03T12:38:19+08:00
revision: 1

## Outcome

用户可在 DSH 工作台连接自己的文献目录，审核处理披露并授权，运行准备、扫描、索引及检索。

## Assumptions

- 复用 runCliLibrary 与 connect/authorize/revokeCliLibrary，不复制 LibraryWorkflow。
- 文献库文件保持原目录；授权沿用已定义的深度、端点和fingerprint。

## Approach

为既有CLI业务增加工作台适配层。短配置操作有revision与任务互斥，长任务接入可取消的运行展示，结果提供结构化论文列表。设置页增加文献库与嵌入配置分组，连接本身不隐式授权或开始处理。

## Chunks

### Chunk 1 — 文献库 HTTP 适配

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red signal: 临时目录测试中 library status/connect/authorize/revoke API 缺失。
- Green check: 连接、当前披露授权、过期fingerprint拒绝、撤销及config revision刷新。
- regression checks: cli-library-config、settings、workbench HTTP。
- [ ] implementation and tests accepted

### Chunk 2 — 长任务与检索

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red signal: scan/index/prepare/search不可从工作台执行；忙状态无法互斥。
- Green check: 模拟模型与解析器、临时文献目录，验证扫描/索引进度与结构化检索结果、取消及未授权拒绝。
- regression checks: cli-library-workflow 与 runs/settings 互斥。
- [ ] implementation and tests accepted

### Chunk 3 — 文献库界面

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red signal: DOM无连接/披露/授权与结果入口。
- Green check: DOM→真实适配层契约，包括取消授权无副作用与错误输入保留。
- regression checks: 全量workbench UI、CLI typecheck、实际DSH隔离Host。
- [ ] implementation and tests accepted

## Phase verification

- 无网络读取用户实际文献，使用临时资料与模拟模型/解析器验证完整流程。
- 连接、授权、scan/index及search在同一工作台持续可用，设置修订不会使后续请求永久失败。

## Abort / reshape triggers

- 如CLI入口无法传递结构化结果/取消，提取共享应用服务后接入，不能另写业务副本。
- 如嵌入配置更改扩大披露范围，必须由既有授权协议重新审核，不自动授权。

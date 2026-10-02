# P1 — Settings and onboarding

goal_ref: ../goal.md
created: 2026-10-03T00:22:59+08:00
updated: 2026-10-03T00:22:59+08:00
revision: 1

## Outcome

新用户无需终端 init 即可打开并配置工作台，已有用户可安全修改设置并直接生成日报。

## Assumptions

- 设置继续存入 CLI TOML，模型 API 与宿主 Agent 独立。
- 首次流程先覆盖日报必要字段，邮件、embedding、schedule 可分组扩展。

## Approach

共享配置服务管理安全投影、校验与原子保存；未配置状态启动同一工作台，保存后重建读取服务并更新生成配置。

## Chunks

### Chunk 1 — 配置服务

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red signal: 新设置服务契约测试缺少实现；再验证修订冲突、保留密钥、首次创建。
- Green check: workbench-settings tests
- regression checks: cli-config、cli-library-config tests
- [ ] implementation and tests accepted

### Chunk 2 — 首次启动与 HTTP 接入

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red signal: 无配置 ui 启动失败；settings HTTP 不存在。
- Green check: cli-workbench、workbench-server/onboarding tests
- regression checks: embedding、marks、generation HTTP guards
- [ ] implementation and tests accepted

### Chunk 3 — 设置界面

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red signal: DOM 中无法编辑保存设置，首次访问无引导。
- Green check: workbench settings UI tests
- regression checks: workbench UI tests、CLI typecheck、DSH build/Host tests
- [ ] implementation and tests accepted

## Phase verification

- 临时配置目录内创建配置、保存、重新读取，并启动第一份模拟日报；不触及用户配置。
- 现有 DSH 嵌入与浏览界面不回归。

## Abort / reshape triggers

- 如果需要 DSH 专属业务副本，改回共享工作台边界。
- 配置损坏不可静默当作首次运行覆盖。

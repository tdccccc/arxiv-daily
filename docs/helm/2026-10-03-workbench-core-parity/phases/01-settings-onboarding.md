# P1 — Settings and onboarding

goal_ref: ../goal.md
created: 2026-10-03T00:22:59+08:00
updated: 2026-10-03T12:38:19+08:00
revision: 2

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
- [x] implementation and tests accepted

### Chunk 2 — 首次启动与 HTTP 接入

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red signal: 无配置 ui 启动失败；settings HTTP 不存在。
- Green check: cli-workbench、workbench-server/onboarding tests
- regression checks: embedding、marks、generation HTTP guards
- [x] implementation and tests accepted

### Chunk 3 — 设置界面

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red signal: DOM 中无法编辑保存设置，首次访问无引导。
- Green check: workbench settings UI tests
- regression checks: workbench UI tests、CLI typecheck、DSH build/Host tests
- [x] implementation and tests accepted

## Phase verification

- 临时配置目录内创建配置、保存、重新读取，并启动第一份模拟日报；不触及用户配置。
- 现有 DSH 嵌入与浏览界面不回归。

## Abort / reshape triggers

- 如果需要 DSH 专属业务副本，改回共享工作台边界。
- 配置损坏不可静默当作首次运行覆盖。

## Observed evidence

- 配置契约缺模块 Red；typed stub 后 11 行为 Red；实现与补充校验后 14 设置测试 Green，含真实并发创建冲突、symlink、未知配置保留。
- CLI 无配置启动 Red（退出2而非0）；HTTP 无配置 Red（构造文档服务异常）；修复后首次设置、生成、运行中保存拒绝、换目录与外部来源防护 Green。
- UI 缺少表单 Red，分组导航 Red，真实保存发现主题id丢失 Red；修复后 DOM→真实配置回读与导航均 Green。
- 全量 CLI：26文件230测试通过；DSH：20测试通过、无skip，含两个真实隔离 Host 场景（已有配置/无配置→保存→模拟日报→阅读标记→停用/启用）。
- CLI typecheck、boundaries、product inventory、diff check通过；linux/x64 DSH 0.1.4包已生成。
- 不运行真实模型、邮件或desktop视觉自动化。设置范围为研究记录、模型API、每日发现；邮件/embedding/schedule字段保留但暂未开放编辑。
- 已接受实现：91a800f。此前名称修改独立保存于b4c3c8b。

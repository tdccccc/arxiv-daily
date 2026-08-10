# Settings Change Transaction

status: done
updated: 2026-08-10
owner: zcode-main-session

## Intent

让 declarative 与 legacy Settings 通过同一事务式变更入口，使持久化配置、live settings、state/history stores、logger 和 scheduler 始终同步。

## Success criteria

- [x] 普通 declarative scalar 不再直接修改 live settings 后裸保存，失败时配置与运行时均保持旧值。
- [x] output directories 同时校验冲突、预加载候选 stores，并只在持久化成功后安装；活跃运行时拒绝切换根目录。
- [x] log level、timezone、tick interval 和 schedule enabled 的 runtime effect 只在持久化成功后发生。
- [x] 时区以 draft 提交并严格验证，无效持久化时区安全回退，不再触发 Dashboard/Logger `RangeError`。
- [x] Obsidian 1.4 legacy 与 1.13+ declarative 行为一致，定向及完整门禁通过并同步技术报告。

## Non-goals

- 不整体重写 topic/category 编辑器或设置页面视觉设计。
- 不迁移 CLI 配置格式，也不引入 Obsidian 1.13-only 的运行时依赖。
- 不在目录切换时自动搬迁旧数据。

## Constraints

- 保留现有 settings 对象及嵌套对象 identity，避免旧 renderer 闭包失效。
- 变更请求必须串行；candidate 准备阶段不能暴露给运行中服务。
- 失败必须可诊断且不得依赖第二次持久化才能回滚。

## Phases

1. P1 — 两套 Settings renderer 通过统一事务保持配置与运行时一致 — status: done

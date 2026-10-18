# 面板"启用/禁用调度器"菜单吞掉保存错误；`setLlmBaseUrl` 是绕过设置事务的死代码

> 发现日期：2026-10-18 · 来源：main `21fe2ce` 深度代码审查 · 编号 P8
> 范围：插件面板、插件设置 · 严重程度：低 · 状态：已修复（分支 `fix/2026-10-18-review`，提交 11de11a） · 置信度：代码推断

## 问题

1. 面板"More"菜单里的调度器开关在异步 `onClick` 中 `await this.plugin.setScheduleEnabled(!enabled)`，没有 try/catch。data.json 写不进去（或调度器运行时效果在提交前失败）时，`setScheduleEnabled` 以 `SettingsChangeError` 拒绝，变成未处理的 Promise 拒绝：没有提示、没有日志；菜单文字基于构建菜单时捕获的旧 `enabled`，用户看不出什么都没变。设置页和引导页对同一调用都有 try/catch 并会报告。
2. `ArxivDailyPlugin.setLlmBaseUrl` 直接修改 `this.settings.llm.baseUrl` 并在 `SettingsChangeService` 之外持久化。生产代码从不调用它（设置走 `changeValue('llm.baseUrl')`），只有测试调用，于是这些测试锁定的是一条并不发布的路径（没有 `normalizeSettingsEdits`、不与其他事务串行），如果以后有人重新接上它就是个陷阱。

## 原因

- `plugin/src/dashboard/view.ts:1856-1866`：`.onClick(async () => {...})` 缺少兄弟菜单项使用的 `runDetached` / try-catch。
- `plugin/main.ts:722-743` `setLlmBaseUrl` 是 `SettingsChangeService` 成为事务边界之前的遗留；测试在 `plugin/tests/settings-lifecycle.test.ts:209-253`。

## 可能的解决方案

1. 用 try/catch（或 `this.runDetached`）包裹，失败时提示并刷新；点击时读取 `this.plugin.settings.schedule.enabled`，而不是用构建菜单时的值。
2. 删除 `setLlmBaseUrl`，三个测试改走 `settingsChanges.changeValue('llm.baseUrl', ...)`，覆盖真实路径（包括端点变更时取消操作）。

## 修复后应补的测试

- `setScheduleEnabled` 保存失败时出现提示，没有未处理的拒绝。
- base URL 相关测试改为通过 `changeValue` 进行。

## 修复情况

面板调度器开关改走 `runDetached`（失败有提示），点击时读取当前设置；删除 `setLlmBaseUrl`，三个测试改用 `changeValue('llm.baseUrl')`。

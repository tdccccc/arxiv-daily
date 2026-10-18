# `workbench-ui.json` 损坏后 `/api/preferences` 读写都返回 500，无法在界面里修复

> 发现日期：2026-10-18 · 来源：main `21fe2ce` 深度代码审查 · 编号 CLI-10
> 范围：CLI 阅读工作台界面偏好 · 严重程度：低 · 状态：已修复（分支 `fix/2026-10-18-review`，提交 ef0a714） · 置信度：代码推断

## 问题

`readPreferences` 只把 ENOENT 映射为默认值；其他错误（JSON 解析失败、校验不通过）都会重新抛出。手动编辑、同步工具的冲突文件、零长度文件都很常见。此时 `GET /api/preferences` 每次加载都返回 500 "read failed"；`POST /api/preferences` 也失败，因为 `savePreferences` 在锁内同样调用 `readPreferences`。用户无法在界面里修复，只能手动删文件。

## 原因

- `apps/cli/src/workbench/preferences.ts:12-16`：`catch` 对非 ENOENT 错误重新抛出。
- `preferences.ts:25` `savePreferences` 通过同一函数读取现有值。
- `apps/cli/src/workbench/server.ts:144` 把异常映射为 500。

## 可能的解决方案

推荐：无法解析或校验失败的内容视为默认值（记录一条警告），`savePreferences` 随后可正常覆盖。只丢失外观类偏好，代价很小。

## 修复后应补的测试

- 文件内容为非法 JSON / 零长度 / 校验失败时，读取返回默认值，保存成功并写出合法文件。

## 修复情况

损坏的 `workbench-ui.json` 返回默认值并在 stderr 警告；部分有效的文件保留有效字段；下一次保存写出合法文件。EISDIR 等真实 I/O 错误仍会报告。

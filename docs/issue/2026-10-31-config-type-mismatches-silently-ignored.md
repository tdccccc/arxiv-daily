# config.toml 里类型写错的值被静默忽略，功能悄悄回到默认

状态：未修复

> 来源：2026-10-31 main `e0498c94` 代码审查（会话内直接审查，静态推断，未运行）
> 严重程度：低 · 置信度：代码推断

## 问题与依据

`apps/cli/src/config.ts`：

- 433-441 行 `stringField` 遇到非字符串直接返回默认值；
- 185-187 行 `thinking_mode` 非布尔忽略；253 行 `email.enabled` 非布尔忽略；387-389 行 `interval_hours` 非数字忽略；
- 297 行 `request_delay_ms` 小于 3000 时静默抬到 3000。

例子：`enabled = "true"`（带引号）让邮件永远不发，`email status` 也不会指出；`thinking_mode = "true"` 让思考模式没开；`reasoning_effort = 1` 回到默认档。相反 `max_daily_papers` 类型错会报错（229-233 行），行为不一致。

## 原因

映射函数把"缺失"和"类型不对"当成同一种情况。

## 建议和验收

- 已知键类型不符时报 `CliConfigError`（与 `max_daily_papers` 一致），信息指出键名和期望类型；或至少在 `status` 和启动时给出警告。
- `request_delay_ms` 被抬高时提示一次。

## 修复后应补的测试

config 测试覆盖每个受影响键的类型错误：带引号的布尔、数字写成字符串、字符串写成数字。

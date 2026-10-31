# "加载模型列表"对 Anthropic 原生端点永远失败：请求头用了 Bearer 而不是 x-api-key

状态：未修复

> 来源：2026-10-31 main `e0498c94` 代码审查（会话内直接审查，静态推断，未运行）
> 严重程度：中 · 置信度：代码推断

## 问题与依据

`packages/core/src/llm/client.ts:285-320` 的 `fetchModels` 对所有候选 URL 固定发送 `Authorization: Bearer <key>`，不带 `anthropic-version`；而聊天请求的 `chatHeaders()`（558-570 行）对 `usesNativeAnthropic` 正确使用 `x-api-key` + `anthropic-version`。Anthropic 预设的 base URL 就是 `https://api.anthropic.com/v1`（`packages/core/src/settings/providers.ts:35-46`），所以选了这个预设就一定走原生端点。

三处调用都受影响：

- Obsidian 插件设置页的 "Get models" 按钮：`plugin/src/settings/declarative-rows.ts:217-260`
- CLI `init` 第 5 步 "Test connection and load model list"：`apps/cli/src/init.ts:427-470,890-907`
- 工作台设置页的加载模型：`apps/cli/src/workbench/settings-actions.ts:33`

Anthropic 对缺少 `x-api-key` 的请求返回 401，用户看到 "Failed to fetch models from any endpoint: …"；init 里显示 "Could not list models"，只能手打模型 id。聊天本身是正常的，所以这是一个只在设置阶段出现、看起来像"密钥不对"的假故障。

## 原因

`fetchModels` 早于 Anthropic 原生支持编写，没有复用 `chatHeaders()`。

## 建议和验收

- `fetchModels` 对原生 Anthropic 使用 `chatHeaders()`（`x-api-key`、`anthropic-version`），候选 URL 用 `{base}/models`；Anthropic 返回 `{data:[{id,…}]}`，现有 `parseModelList` 已能解析。
- OpenAI 兼容路径保持原样。

## 修复后应补的测试

- `packages/core/tests/llm/`：provider anthropic + api.anthropic.com 时，模型列表请求头含 `x-api-key` 与 `anthropic-version`、不含 `Authorization`；OpenAI 兼容配置的请求头不变。
- 解析 Anthropic 形状的模型列表响应。

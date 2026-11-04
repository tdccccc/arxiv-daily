# init 对所有服务商默认开启思考模式，而连接测试只请求 /models，参数不兼容到第一次日报才以永久失败暴露

状态：已修复（分支 `fix/2026-10-31-low-findings`）

> 来源：2026-10-31 main `e0498c94` 代码审查（会话内直接审查，静态推断，未运行）
> 严重程度：低 · 置信度：代码推断（对第三方网关行为的假设）

## 问题与依据

- `apps/cli/src/init.ts:240-241`：`thinkingMode` 默认 true、effort high，与所选预设无关（预设本身有 `thinkingMode` 字段，`packages/core/src/settings/providers.ts`）。
- 向导第 5 步只调用 `fetchModels`（`init.ts:427-470`），不发聊天请求。
- `packages/core/src/llm/client.ts:688-713`：对 openai / custom 预设开着思考模式时发送 `reasoning_effort`；严格的 OpenAI 兼容网关会以 400 拒绝未知参数，`isPermanentLlmError`（727-736 行）把 400 当永久错误，日报记为 `failed_permanent: paper filter LLM failed: …`。

用户看到的是：向导里 "Connected — N model(s) listed"，第一次 `run --today` 却永久失败，错误是服务商原话，和"思考模式"看不出关系。

## 原因

默认值不跟随预设；连接测试没有覆盖真正的聊天参数。

## 建议和验收

- 思考模式默认值跟随 `preset.thinkingMode`（custom 默认关）。
- 向导的测试改为（或增加）一次 `testConnection()` 式的最小聊天请求，并带上选定的思考/推理参数；失败时提示"可关闭思考模式再试"。

## 修复后应补的测试

init 测试：custom 预设默认 `thinking_mode = false`；连接测试发出聊天请求且携带所选参数；聊天测试失败时的提示文案。

## 修复情况

两条建议都做了：思考模式默认值随 `preset.thinkingMode`（custom 默认关），提示语也显示当前默认；在列出模型之后、写入配置之前，向导新增一次真实的最小聊天请求（`LlmClient.testConnection` 改为按选定的 provider/model/思考与推理参数发请求，而不是只发 `max_tokens:5`），失败时提示"可回去关掉思考模式再试"，选择"不保留"会退回思考模式步骤重选并重测。测试：`apps/cli/tests/cli-init.test.ts` 新增两个用例（custom 预设写入 `thinking_mode = false` 且提示 `(default: No)`；首次聊天被拒 → 返回关掉思考 → 第二次聊天成功后继续），既有的 provider 内容审查与模型列表用例不变通过。

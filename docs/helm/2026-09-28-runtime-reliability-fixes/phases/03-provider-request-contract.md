# P3 — provider-request-contract

goal_ref: ../goal.md
created: 2026-09-28T22:44:52+08:00
updated: 2026-09-28T22:56:01+08:00
revision: 2

## Outcome

LlmClient 发送服务商支持的顶层推理参数，输出预算有效，旧请求语义的 checkpoints 不会复用；所选真实连接的验收结果被记录。

## Assumptions

- 显式 provider 选择决定请求契约；不依靠模型名字静默更改用户 provider。
- OpenAI/自定义兼容服务使用 reasoning_effort；DeepSeek 使用 thinking 与 reasoning_effort；Anthropic 兼容接口使用 thinking/budget_tokens；当前 GLM 预设使用 thinking。
- Anthropic 的手动思考预算必须低于显式输出上限；过小上限应明确拒绝，不能静默关闭用户要求的思考。
- 用户现有私有中转服务可能有自己的参数兼容规则，独立记录实际结果。
- 官方文档确认 Claude 4.7+ 拒绝手动预算；官方 Anthropic host 使用原生 Messages API 传递 adaptive thinking 与 output_config.effort，第三方 Anthropic 兼容地址保持其 Chat Completions 路径。

## Approach

在单一请求构建路径上按 provider 生成参数并保留现有流式兼容回退。增加传输契约版本到 checkpoint generation identity 与解码校验，保证不能误复用旧行为生成的中间结果。

## Chunks

### Chunk 1 — 推理参数、输出预算与 checkpoint 合约同步（F19）

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 新 LlmClient HTTP 边界测试断言各 provider 的实际 JSON 不含 extra_body，正确保留/排除 thinking 和 reasoning_effort；显式禁用对支持 thinking 的服务发 disabled；Anthropic 预算小于 token cap；OpenAI token cap 使用 max_completion_tokens。旧实现不满足这些断言。补充官方 Anthropic Messages 的认证、system 消息、SSE 文本/usage/错误和自适应参数测试；checkpoint endpoint identity 需跟随实际请求 URL。各 checkpoint 的 generation 缺传输合约版本时不得解码/复用。
- Green check: focused core LLM 与 checkpoint 测试。
- regression checks: 完整 core suite、root typecheck、CLI/plugin LLM 相关测试与构建；流式回退保留参数，异常不泄漏密钥。
- [ ] implementation and tests accepted

### Chunk 2 — 现有连接合成请求验收

- change kind: non-behavioral verification
- strategy: proportionate check
- baseline: 读取已配置 provider/model/endpoint，密钥仅在内存中用于同一 endpoint，不写入报告。
- Green check: 用户选定连接收到简短合成提示，返回预期内容且没有不支持参数错误；记录模型、协议与结果，避免记录密钥或研究内容。
- exception: 真实连接需要可访问的 endpoint 与用户选择；不可达或未授权则保留待办，不阻塞 P4/P5 的独立实现。
- [ ] live acceptance recorded

## Phase verification

- 观察 Red/Green、checkpoint 旧记录拒绝与完整 core 回归。
- 不把成功生成文本等同于供应商内部思考过程已被独立证明；核实可观察的请求/响应契约。

## Abort / reshape triggers

- 服务商官方契约与预设假设冲突时先修正模型/请求映射，不靠删掉全部推理参数“让请求通过”。
- gateway 不接受所选 provider 的契约时记录实际兼容需求，不改用户配置或切换 endpoint。

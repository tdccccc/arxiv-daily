# P2 — cdp-session-layer

goal_ref: ../goal.md
updated: 2026-08-27

## Outcome

harness 提供一个可复用的 CDP 会话对象：连接到 vault 渲染进程、求值表达式并把异常映射为可断言结果、全程收集 console error 与 pageerror、处理首次打开的信任对话框，并能断言运行中的插件版本等于被测版本。

## Assumptions

- Node 22 内置 WebSocket 足以驱动 CDP，无需引入 `puppeteer` 或 `ws`；探针已用它完成 `Runtime.evaluate` 与 `Runtime.consoleAPICalled`。
- 隔离配置下每轮都会出现「Do you trust the author of this vault?」对话框，其按钮文本为 `Trust author and enable plugins`；接受后 `app.plugins.plugins` 出现被测插件。
- `Runtime.evaluate` 配合 `returnByValue` 足以读取 `app` 状态；无法结构化克隆的值需要在表达式内自行投影。
- 诊断收集存在时间窗口问题：会话在页面已加载后才连接，启动期 console 错误可能已经错过。这条假设必须在 P2 内验证而不是假定成立。

## Approach

先把协议层与决策层分开：一个可注入 transport 的 CDP 客户端负责请求/响应配对与事件分发，其上是求值、诊断收集与对话框处理三个可独立测试的能力。真实连接只用来确认协议假设成立，不承担逻辑证据。启动期诊断的时间窗口先用一次真实观察定性，再决定是否需要重载页面重放。

## Test strategy

- change kind: behavior change（新增 harness 能力）
- strategy: 协议与能力层走 strict Red–Green–Refactor（注入假 transport）；真实连接走 Green characterization
- Red / baseline signal: `node --test scripts/tests/desktop-acceptance-cdp*.test.mjs` 在实现前因模块缺失或断言不满足而失败
- Green / regression checks: `node --test scripts/tests/desktop-acceptance-*.test.mjs` 全绿；`npm run lint`、`check:boundaries`、`check:product-units` 通过；`test:release-tools` 仅出现既有 flaky；`git diff --check` 干净
- exception: 真实渲染进程的连接与对话框形态无法在单元层制造有意义的 Red，改为一次可观察的端到端 Green，并以「插件版本断言为被测版本」作为补偿验证

## Tasks

- [ ] CDP 客户端：连接指定 target、请求/响应按 id 配对、事件分发、连接失败与关闭的可诊断错误；拒绝非 vault 页面 target。
- [ ] 求值能力：`Runtime.evaluate` 的返回值与抛出异常映射为可断言结果，异常携带原始描述而非静默返回 undefined。
- [ ] 诊断收集：收集 console error/warning 与 pageerror，提供「零错误」断言；并确定启动期时间窗口是否需要重载重放，把结论写进 journal。
- [ ] 信任对话框：检测并接受，无对话框时为 no-op；以插件出现在 `app.plugins.plugins` 作为等待条件，而非固定 sleep。
- [ ] 会话整合：`runDesktopSession` 的 body 收到已就绪的 session 对象，并在真实运行中断言插件版本等于被测版本。

## Verification

- 定向：`node --test scripts/tests/desktop-acceptance-*.test.mjs` 全绿，且每个任务的 Red 曾被观察到。
- 端到端：一次真实运行中 session 报告插件版本为当前分支构建版本（当前为 `0.4.3`，与 vault 自带的 `0.4.5` 不同，因此该断言能真正区分被测构建），且诊断收集器返回可读的错误列表。
- 安全边界：运行前后用户真实 Obsidian 会话存活；测试 vault 的受保护文件逐字节还原；无 harness 进程或沙箱目录残留。
- 门禁：`npm run lint`、`check:boundaries`、`check:product-units` 通过；`test:release-tools` 仅出现既有 flaky；`git diff --check` 干净。

## Abort / reshape triggers

- 如果启动期 console 错误必须靠重载页面才能捕获，而重载会改变被测行为（例如重跑插件 onload 的副作用），停止并 L2 reshape 诊断策略，不得以「大概没漏」代替证据。
- 如果信任对话框形态不稳定（文本变化、偶尔不出现、被其它 modal 遮挡）导致检测不可靠，停止并 reshape 为配置预置或显式等待条件。
- 如果 `Runtime.evaluate` 无法读取断言所需的插件状态（例如设置对象不可序列化），停止并重新设计断言的投影方式，而不是放宽断言。

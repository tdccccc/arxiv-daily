# P4 — integration-and-degradation

goal_ref: ../goal.md
updated: 2026-08-28

## Outcome

桌面验收可由单条命令复现执行；环境不具备时给出指名道姓的阻塞原因并非零退出，而不是伪装成通过或抛出栈追踪；harness 不进入默认测试与 plugin bundle，既有门禁维持通过。

## Assumptions

- 仓库既有门禁 `check:boundaries` 与 `check:product-units` 不覆盖 `scripts/`，但 plugin bundle 预算与 lint 覆盖 `plugin/`，因此 harness 只要留在 `scripts/` 下即天然隔离；这一点需由门禁实际运行确认而非假定。
- 阻塞形态至少包括：未设置测试 vault、vault 不存在或不是 vault、Obsidian 可执行文件缺失、虚拟显示不可用、分支未构建、vault 内无可用 PDF。
- 默认 `npm test` 经 `scripts/run-root-tests.mjs` 转发到 workspace 测试，不会拾取 `scripts/tests/`；`test:release-tools` 会拾取，因此 harness 的单元测试应当在那里保持全绿。

## Approach

先把「能否运行」与「运行结果」分开：一个 preflight 逐项检查环境并返回结构化阻塞清单，验收入口在 preflight 失败时打印可操作指引并以专用退出码结束。随后接入 npm script，并实际运行既有门禁确认隔离成立。最后把可复现的手工步骤写进 harness 自身的 README，使其在无法自动执行时仍是等价记录。

## Test strategy

- change kind: behavior change（新增 preflight 与集成入口）
- strategy: preflight 判定走 strict Red–Green–Refactor（注入假 fs 与假探测）；门禁隔离走实际运行的 Green 证据
- Red / baseline signal: `node --test scripts/tests/desktop-acceptance-preflight.test.mjs` 在实现前因模块缺失或断言不满足而失败
- Green / regression checks: `node --test scripts/tests/desktop-acceptance-*.test.mjs` 全绿；`npm test`、`npm run lint`、`check:boundaries`、`check:product-units`、`test:release-tools`（仅既有 flaky）、`npm run build` 通过；一次真实 `npm run test:desktop` 通过
- exception: 无

## Tasks

- [x] preflight：逐项检查 vault、Obsidian 可执行文件、虚拟显示、分支构建与 vault 内 PDF，返回结构化阻塞清单而非首个错误。
- [x] 验收入口接入 preflight：阻塞时打印可操作指引并以专用退出码结束，不打印栈追踪。
- [x] 接入 `npm run test:desktop`，确认它不进入默认 `npm test`，且 `scripts/tests/` 的 harness 单元测试在 `test:release-tools` 中保持全绿。
- [x] 确认门禁隔离：`lint`、`check:boundaries`、`check:product-units`、production build 与 bundle 预算在 harness 存在下全部通过。
- [x] 写入 harness README：运行方式、环境要求、安全约束与不可自动执行时的等价手工步骤。

## Verification

- 定向：`node --test scripts/tests/desktop-acceptance-*.test.mjs` 全绿，且 preflight 的 Red 曾被观察到。
- 阻塞降级：在缺少 vault、缺少构建与不存在的 Obsidian 路径三种情况下各观察到指名阻塞原因与非零退出。
- 集成：一次 `OBSIDIAN_TEST_VAULT=... npm run test:desktop` 真实通过；`npm test` 不执行桌面验收。
- 门禁：`lint`、`check:boundaries`、`check:product-units`、`npm run build`、`check:obsidian-submission` 通过；`test:release-tools` 仅出现既有 flaky。

## Abort / reshape triggers

- 如果 harness 的存在使任一既有门禁失败，停止并把 harness 移出被门禁覆盖的路径，而不是放宽门禁。
- 如果 preflight 无法区分「环境不具备」与「验收失败」，停止并重新设计退出码语义——两者混淆会让 CI 与人都误读结果。

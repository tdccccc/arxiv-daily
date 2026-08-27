# P3 — acceptance-scenarios

goal_ref: ../goal.md
updated: 2026-08-28

## Outcome

P7 遗留的四项桌面验收在真实 Obsidian 宿主中产出断言级证据：PDF `#page=N` 打开与页码降级、sidecar 默认关闭且无请求外泄、启用后探测失败回退、旧 settings 迁移，且全程零 console 错误。

## Assumptions

- 测试 vault 的 `test_library/` 内有 376 个真实 PDF，它们位于 vault 内部，因此 `resolveLibraryPdfOpenTarget` 会走 `kind: "vault"` 分支并产出 `<path>#page=N`。
- Obsidian 内置 PDF 视图能通过 `app.workspace.openLinkText` 接受 `#page=N` 子路径。视图当前页的读取方式尚未验证，可能需要从 leaf 的 ephemeral state 或 pdf viewer 实例读取；这条假设必须实测确认，不得假定。
- `Network.enable` 可在渲染进程中记录请求，从而把「sidecar 关闭时无请求」变成可断言事实而非推断。
- 当前 `data.json` 不含 `pdfParserSidecar`，因此天然是旧结构；但验收仍需安装受控 fixture，不依赖用户文件的当前内容。
- 分支构建 0.4.3 已包含 P6 的 sidecar 设置与 `migratePdfParserSidecarSettings`。

## Approach

先给 session 增加 `beforeLaunch` 钩子，使场景能在 Obsidian 启动前安装受控 settings fixture；fixture 的还原沿用既有守卫。随后每个场景是一个独立、可单独运行的断言单元，共享同一 session。先用一次实测确定 PDF 视图页码的可读取路径，再据此写断言，而不是先写断言再迁就实现。

## Test strategy

- change kind: behavior change（新增验收能力）
- strategy: 场景编排与判定逻辑走 strict Red–Green–Refactor（注入假 session）；真实宿主行为走 Green characterization
- Red / baseline signal: `node --test scripts/tests/desktop-acceptance-scenarios.test.mjs` 在实现前因模块缺失或断言不满足而失败
- Green / regression checks: `node --test scripts/tests/desktop-acceptance-*.test.mjs` 全绿；一次真实运行四个场景全部通过；`lint`、`check:boundaries`、`check:product-units` 通过；`git diff --check` 干净
- exception: 真实 PDF 视图与 Obsidian 内部 API 无法在单元层制造有意义的 Red，改为实测定性后写断言，并以「断言在错误页码下会失败」作为反向证据

## Tasks

- [ ] session `beforeLaunch` 钩子与 settings fixture 安装：fixture 在启动前写入，运行后由既有守卫还原。
- [ ] 实测确定 Obsidian PDF 视图当前页的可读取路径，并把结论写入 journal；据此实现 `#page=N` 断言与页码降级断言。
- [ ] 网络观察：`Network.enable` 记录请求，断言 sidecar 关闭时无任何指向 sidecar 端点的请求。
- [ ] sidecar 场景：旧 settings 迁移后默认关闭；启用并指向不可达 loopback 端点后，探测失败不产生 console 错误且插件继续可用。
- [ ] 验收编排：一条命令运行全部场景，逐项报告通过与否，任一失败即非零退出。

## Verification

- 定向：`node --test scripts/tests/desktop-acceptance-*.test.mjs` 全绿，且每个任务的 Red 曾被观察到。
- 端到端：一次真实运行四项场景全部通过，诊断为 complete 且零错误；反向证据表明断言在错误页码下确实失败。
- 安全边界：运行前后用户真实 Obsidian 会话存活；`data.json` 与 `workspace.json` 逐字节还原；无 harness 进程或沙箱残留。
- 门禁：`lint`、`check:boundaries`、`check:product-units` 通过；`test:release-tools` 仅出现既有 flaky；`git diff --check` 干净。

## Abort / reshape triggers

- 如果 Obsidian PDF 视图无法读取当前页，导致 `#page=N` 只能断言「文件被打开」而非「定位到该页」，停止并在 journal 中明确降级为部分证据，不得把弱断言描述为已验证页码定位。
- 如果启用 sidecar 会触发真实索引运行或长耗时后台工作，停止并把场景收敛为设置层与网络层断言，不在验收中跑完整索引。
- 如果任一场景需要修改被测插件代码才能断言，停止并 L2 reshape：harness 不得为了可测性改变被测行为。

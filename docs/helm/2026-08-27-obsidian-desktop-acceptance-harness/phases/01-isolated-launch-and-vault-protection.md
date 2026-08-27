# P1 — isolated-launch-and-vault-protection

goal_ref: ../goal.md
updated: 2026-08-27

## Outcome

harness 能以隔离配置启动一个只挂载 `plugin_test` 的真实 Obsidian，按进程组完整回收，在任何退出路径上还原测试 vault 的可变状态，并断言运行的是当前分支构建；用户常驻的真实 Obsidian 会话与真实 vault 列表全程不受影响。

## Assumptions

- 仓库既有 idiom 适用：`scripts/*.mjs` 导出可注入依赖（`spawn` / `fs` / `kill`）的纯函数，`scripts/tests/*.test.mjs` 用 `node --test` 驱动，与 `run-root-tests.mjs` 一致。
- 隔离 `XDG_CONFIG_HOME` 足以让 Obsidian 与用户真实配置完全分离；探针已验证它只读写该目录下的 `obsidian/obsidian.json`。
- 用户真实会话未开启 remote debugging，但 harness 仍不应硬编码 9222，需要能选空闲端口以避免与并行运行争用。
- 测试 vault 中会被改写的可变状态是 plugin settings store（`data.json`）与 `workspace.json`；探针观察到这两者在一次启动后均发生变化，`app.json` 未变。

## Approach

先把可测的决策部分从副作用中分离：vault 配置合成、状态备份/还原、进程组回收指令、构建部署与版本断言各自是一个可注入依赖的纯函数，用 `node --test` 做严格 Red–Green。随后只留一个薄的端到端骨架把它们串起来，做一次真实启动/回收，作为环境证据而非逻辑证据。

## Test strategy

- change kind: behavior change（新增 harness 能力）
- strategy: 可注入单元走 strict Red–Green–Refactor；真实启动骨架走 Green characterization（环境证据）
- Red / baseline signal: `node --test scripts/tests/desktop-acceptance-*.test.mjs` 在实现前因模块缺失或断言不满足而失败，失败原因指向被测契约本身
- Green / regression checks: 同一 `node --test` 目标转绿；`npm run check:boundaries`、`npm run check:product-units`、`npm run lint`、`npm run test:release-tools` 维持通过；`git diff --check` 干净
- exception: 真实 Obsidian 启动无法在单元层制造有意义的 Red，改为一次可观察的端到端 Green，并以「用户真实会话启动前后均存活」作为补偿验证

## Tasks

- [x] vault 配置合成：由测试 vault 绝对路径产出只含该 vault 且 `open: true` 的 `obsidian.json` 内容；拒绝相对路径、拒绝非指定 vault、拒绝在已有真实配置目录上写入。
- [ ] 测试 vault 状态保护：备份并还原 plugin settings store 与 `workspace.json`；成功、抛错与信号中断三条退出路径都完成还原，且还原具备幂等性。
- [ ] 进程组生命周期：以 `setsid` 启动并记录 PGID，按 `kill -PGID` 分级回收（TERM→KILL）；任何按进程名或命令行模式的回收路径在测试中被显式拒绝。
- [ ] 被测构建部署与版本断言：把当前分支构建的 `main.js` 与 `manifest.json` 部署进测试 vault，并在运行前断言加载版本等于被测版本；不触碰 vault 中既有的历史构建备份文件。
- [ ] 端到端骨架：串联上述四项，实际启动一次真实 Obsidian、确认 CDP 端口可达后干净回收，并在空闲端口选择下可重复执行。

## Verification

- 定向：`node --test scripts/tests/desktop-acceptance-*.test.mjs` 全绿，且每个任务的 Red 曾被观察到。
- 端到端：骨架运行一次后 CDP `/json/version` 曾返回，进程组已消失，测试 vault 的 `data.json` 与 `workspace.json` 与运行前逐字节一致。
- 安全边界：运行前后 `pgrep -f 'user-data-dir=/home/tiandc/.config/obsidian'` 均命中，证明用户真实会话存活；隔离配置目录中的 vault 列表只含 `plugin_test`。
- 门禁：`npm run check:boundaries`、`npm run check:product-units`、`npm run lint`、`npm run test:release-tools` 通过；`git diff --check` 干净。

## Abort / reshape triggers

- 如果 Obsidian 在隔离 `XDG_CONFIG_HOME` 下仍读写 `~/.config/obsidian`，停止并 L2 reshape 隔离方案，不得以「小心操作」代替隔离。
- 如果还原路径无法覆盖信号中断，导致测试 vault 可能被留在改写状态，停止并先解决状态保护，不进入 P2。
- 如果 harness 必须独占固定 CDP 端口或独占显示，与用户真实会话不能共存，停止并 L2 reshape 进程模型。

# Journal — Obsidian 桌面验收自动化 harness

## 2026-08-27 — note: CDP 可行性探针

- evidence: Obsidian 1.11.5 (Electron 39.2.6 / Chrome 142) 接受 `--remote-debugging-port`，`/json/version` 返回 `webSocketDebuggerUrl`。在隔离 `XDG_CONFIG_HOME` 中写入只含 `plugin_test` 的 `obsidian.json`（`open: true`）后，Obsidian 自动打开该 vault，绕开了旧 CLI 不支持 `--vault` 的阻塞。通过 Node 22 内置 WebSocket 直连 CDP，`Runtime.evaluate` 成功读到 `app.vault.getName() === "plugin_test"`、`app.vault.adapter.basePath`、40 个 markdown 文件与 21 条 `arxiv-daily:*` 命令；`Runtime.consoleAPICalled` 捕获到合成 warning，证明诊断收集通路成立。DOM 点击「Trust author and enable plugins」后 `app.plugins.plugins` 出现 `arxiv-daily`。
- change: 确认 CDP 为可行路线，据此建立本 initiative；否决了此前的窗口/像素自动化方向。
- disposition: 探针为一次性验证，未留下任何生产代码。测试 vault 的 `data.json` 与 `workspace.json` 在探针后已从备份还原（备份位于 `/tmp/plugin_test-backup-1787835341`），隔离配置目录已删除。
- next: P1 建立隔离启动、进程组回收与 vault 状态保护的 harness 骨架。

## 2026-08-27 — note: 两个必须固化的安全约束

- evidence: 探针期间 `pkill -f 'remote-debugging-port=9222'` 以退出码 144 杀掉了执行脚本自身，因为 bash 的命令行包含该模式字符串。同时观察到用户有常驻真实 Obsidian 会话（`--user-data-dir=/home/tiandc/.config/obsidian`），任何按进程名的回收都会误杀它。插件加载会改写测试 vault 的 `data.json`，Obsidian 会改写 `workspace.json`。
- change: 将「只用 setsid 进程组 + kill -PGID」与「测试 vault 可变状态备份/还原」写入 goal 的 Constraints 与 Success criteria，而不是留作实现细节。
- disposition: 保留这两条为硬约束；harness 的进程回收与状态保护必须有独立测试证据，不得依赖执行者记得手动做。
- next: 在 P1 的任务中把这两条各自作为一个可独立验收的分块。

## 2026-08-27 — note: 被测构建与 vault 现状不一致

- evidence: `/home/tiandc/Desktop/plugin_test/.obsidian/plugins/arxiv-daily/main.js` 是 0.4.5 构建，其 settings 只有 `llm/arxiv/detailSelection/output/schedule/advanced/email`，不含 P6 引入的 `pdfParserSidecar`。直接在该 vault 上验收会测到历史构建而非被测分支。
- change: 把「部署当前分支构建并断言运行的是被测版本」提升为 goal 的 success criterion，而非 P3 场景的隐含前提。
- disposition: 不修改用户 vault 中的历史构建备份文件；部署走覆盖 `main.js` 加运行前版本断言的路径。
- next: 在 P1 处理构建部署与版本断言。

## 2026-08-27 — L1 adjust: 版本漂移阻塞依赖安装

- evidence: 新 worktree 无 `node_modules`，`npm install` 以 E404 失败，因为 `plugin/package.json` 将 `@arxiv-daily/node-runtime` 钉在 `0.4.1` 而工作区包为 `0.4.3`，npm 无法从 workspace 满足该精确版本转而访问 registry。这正是 `2026-08-17-pdf-hybrid-library-foundation` 的 P7 journal 记录为「outside this change」的既有漂移。在缺依赖状态下 lint 退化为 30 个 `no-unsafe-*` 类型解析错误，属环境假象而非回归。
- change: 运行仓库自带的 `node scripts/sync-release-version.mjs 0.4.3`，仅产生两行改动（`plugin/package.json` 依赖版本、`package-lock.json` 对应条目）；`versions.json` 与 `plugin/versions.json` 已正确，无变化。随后 `npm install --ignore-scripts` 成功，workspace 包正常 link。
- disposition: 保留该修复。它是本 initiative 的前置条件而非顺带重构，使用仓库既定同步机制而非手改版本号。`onnxruntime-node` 的 postinstall 因网络 302 无法下载原生构建，跳过 postinstall 不影响本项目——插件使用 Transformers web 运行时，且 smoke 检查本就禁止导入该原生包。
- next: 以修复后的环境建立真实门禁基线，继续 P1 的可注入单元。

## 2026-08-27 — note: 门禁基线与一个既有 flaky

- evidence: 依赖就绪后 `npm run lint` 为 0 errors / 64 warnings 且退出 0，与 P7 记录基线一致；`check:boundaries`、`check:product-units` 通过。`npm run test:release-tools` 聚合运行时 `the real root npm entry sends a Core focus to Core only` 失败，但单独运行该文件在本 worktree 与干净对照 worktree 均通过；干净对照 worktree 的聚合运行同样复现该失败，证明其为既有并行敏感问题，与本 initiative 无关。
- change: 记录 release-tools 的验收基线为「仅允许上述既有 flaky 失败」。版本漂移修复使该聚合运行从 2 个失败降为 1 个。
- disposition: 不在本 initiative 修复该 flaky，它属于 root test runner 的并发行为，不在桌面 harness 范围内。后续验收引用本基线而非声称 release-tools 全绿。
- next: 继续 P1，实现测试 vault 状态保护分块。

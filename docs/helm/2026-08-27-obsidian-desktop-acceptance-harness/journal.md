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

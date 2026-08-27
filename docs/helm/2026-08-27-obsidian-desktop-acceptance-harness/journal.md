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

## 2026-08-27 — P1 complete, P2 started

- evidence: 五个分块各自观察到 Red 后转 Green，定向目标 61/61。两次真实运行在独立选取的空闲端口 40655 与 46347 上启动 Obsidian 1.11.5，CDP 应答、vault 页面打开、退出码 0。运行后测试 vault 的 `data.json`、`main.js`、`manifest.json`、`workspace.json` 四个文件 md5 与运行前完全一致，vault 回到自带的 0.4.5 构建，29 个 `main.js.bak-*` 历史备份未被读写，沙箱目录与 Xvfb 均无残留，用户真实 Obsidian 会话 4 个进程全程存活。`lint` 0 errors / 64 warnings、`check:boundaries`、`check:product-units` 通过，`test:release-tools` 仅剩既有 flaky。
- change: P1 标记 done，P2 转 active 并写入 `phases/02-cdp-session-layer.md`。
- disposition: 保留全部五个分块。部署产物纳入状态守卫的 `additionalPaths` 是本阶段的关键修正——若不纳入，每轮运行都会用被测构建永久覆盖 vault 自带的 0.4.5 构建。
- next: P2 先实现可注入 transport 的 CDP 客户端与求值能力，再定性启动期诊断的时间窗口问题。

## 2026-08-27 — note: 自匹配陷阱第二次出现

- evidence: 验证回收结果时用 `pgrep -f 'obsidian-acceptance-'` 统计残留，得到 2 个进程，一度判断回收失败。实际匹配到的是执行该命令的 shell 自身——其命令行包含该模式字符串。改用逐进程读取 `/proc/<pid>/cmdline` 并按 `user-data-dir` 前缀分类后，harness 残留为 0，用户真实会话 2 个渲染进程健在。
- change: 无生产代码变更；确认 harness 本身不含任何按模式匹配的进程逻辑，该陷阱只影响人工验证命令。
- disposition: 后续验证一律用 `/proc/<pid>/cmdline` 按 `user-data-dir` 分类，不用 `pgrep -f`。这与 goal 中「禁止按进程名或命令行模式回收」是同一条约束的两面：它既会误杀，也会误报。
- next: 无阻塞，继续 P2。

## 2026-08-27 — L1 adjust: 还原范围收敛为「不可重新生成的状态」

- evidence: 用户指出测试 vault 本就是用来测试的，替换其中的 `main.js` 不构成副作用。这否定了「部署产物必须还原」的前提本身，而不只是权衡其成本。据此重新划线：`main.js` / `manifest.json` 由 `npm run build` 一条命令即可重建；`data.json` 保存手工配置的 endpoint、密钥、topics 与输出路径，重新生成不了。P3 会真实切换 sidecar 设置，若不还原将永久改变用户配置。
- change: `runDesktopSession` 不再把部署产物纳入状态守卫，跑完保留分支构建；`data.json` 与 `workspace.json` 继续还原。守卫的 `additionalPaths` 能力与其测试保留，它表达的「受保护路径必须在 vault 内」约束仍可能被 P3 使用。
- disposition: 保留 P1 全部实现。此前把「构建产物」与「用户状态」当作同一类文件是设计错误，本次收敛而非新增能力。测试 vault 中原有的 0.4.5 构建已被 0.4.3 覆盖且未单独备份；它可由 `release/0.4.5` 分支重新构建，符合本次划定的可重建标准。
- next: 无阻塞，继续 P2 的 CDP 客户端与求值能力。

## 2026-08-27 — note: 符号链接部署方案实测可行但被否决

- evidence: 在 `/tmp` 一次性 vault 中把 `main.js` 与 `manifest.json` 符号链接到仓库构建，Obsidian 1.11.5 正常加载并报告版本 `0.4.3`，证明「零拷贝、永远运行当前构建」在技术上成立。实验全程未触碰 `plugin_test`。
- change: 仍采用复制部署。符号链接是对 vault 的持久改动，且本 initiative 运行在临时 worktree 中，worktree 删除后 vault 内插件即为断链，用户下次手动打开会遇到加载失败。
- disposition: 记录该方案可行以备将来在长期 checkout 中重新考虑；当前不实现。无论采用哪种部署方式，运行前的版本断言都必须保留——它防的是 Obsidian 加载陈旧 bundle，与部署方式无关。
- next: 无。

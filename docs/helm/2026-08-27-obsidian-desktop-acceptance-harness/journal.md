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

## 2026-08-28 — note: PDF 视图页码的可读取路径

- evidence: 实测 `app.workspace.openLinkText("<pdf>#page=4", "", false)` 后，`activeLeaf.getViewState().type` 为 `pdf`，但 `getEphemeralState()` 为空、`viewer.child.subpath` 与 `pdfViewer.currentPageNumber` 均为 undefined。真正承载页码的是 `viewer.child.pdfViewer.page` 与 pdf.js 自身的 `viewer.child.pdfViewer.pdfViewer.currentPageNumber`，两者在 `#page=4` 时均为 4。
- change: `pdfPageLocationScenario` 优先读 pdf.js 的 `currentPageNumber`，回退到 Obsidian 设置的 `pdfViewer.page`。
- disposition: 该断言证明的是「宿主真的定位到了指定页」，不是「文件被打开」。P7 的桌面证据缺口据此可以按实际强度描述，不需要降级为部分证据。
- next: 用负向对照确认断言非空洞。

## 2026-08-28 — note: 验收断言的负向对照

- evidence: 同一场景在 `#page=2` 下观察到 viewer 报第 2 页、在 `#page=4` 下报第 4 页，说明断言跟踪的是请求页码而非常量。向 `sidecarDisabledScenario` 伪造一条指向 capabilities 端点的请求后，场景如实失败并点名该 URL。
- change: 无生产逻辑变更；对照过程发现 `networkUrls()` ��� Obsidian 自身的 `data:` SVG 图标计为网络请求，导致「network requests 2」措辞失实，已改为同时排除 `data:` 与 `blob:`，真实运行后计数为 0。
- disposition: 保留负向对照作为「断言有效」的证据记录。四项场景的通过结论以此为前提，而非仅凭一次全绿。
- next: P4 集成为单条命令并处理环境阻塞降级。

## 2026-08-28 — P3 complete

- evidence: 一次真实运行中四项场景全部通过——旧 settings 迁移出 9 个 section 且 sidecar 保持关闭；sidecar 关闭时 0 个离开进程的请求触及其端点；`test_library/york_sloan_2000.pdf` 在嵌入式 viewer 中定位到第 4 页；启用指向不可达 `127.0.0.1:1` 的 sidecar 未引发渲染进程错误。诊断为 complete，零 console 错误。定向测试 116/116，`lint`、`check:boundaries`、`check:product-units` 通过，`git diff --check` 干净。运行后 `data.json` 与 `workspace.json` 校验和不变，29 个历史备份未动，无 harness 进程或沙箱残留，用户真实会话存活。
- change: P3 五项任务全部勾选。
- disposition: 保留全部场景实现。`sidecar-unreachable-falls-back` 断言的是「启用不可达端点不产生渲染错误」，不是「完整索引流程回退到 PDF.js」——后者需要真实索引运行，超出桌面验收范围，已在场景 detail 中如实表述。
- next: P4 把验收接成单条命令，处理环境不具备时的阻塞降级，并确认门禁隔离。

## 2026-08-28 — 真实中断暴露孤儿进程缺陷

- evidence: 单元层早已覆盖信号还原，但从未在真实运行中触发过。实际向运行中的验收进程发送 SIGTERM 后，`data.json` 确实被还原，却留下 2 个 Obsidian 进程、1 个 Xvfb 与 1 个沙箱目录。原因是信号处理器调用 `process.exit()` 会跳过所有 `finally`，而进程组回收与沙箱清理都写在 `finally` 里。顺序也不安全：还原 `data.json` 时 Obsidian 仍在运行，可能在还原后再次写入。
- change: `protect` 增加 `onInterrupt` 回调，信号处理器先执行调用方清理再还原，且清理失败不阻止还原；`runDesktopSession` 通过可变引用发布进程组回收器与沙箱清理。修复后重做同一中断实验：0 个遗留进程、0 个 Xvfb、0 个沙箱目录，`data.json` 已还原，用户真实会话未受影响。
- disposition: 保留该修复。这条缺陷说明「单元测试覆盖了信号路径」不等于「信号路径在真实进程中正确」——本 initiative 的其余安全断言都已有真实运行证据，唯独这条此前只有单元证据。
- next: 完成 P4 剩余集成并收尾 goal。

## 2026-08-28 — note: 自匹配陷阱第三次出现

- evidence: 清理上述孤儿进程时，我在脚本首行写了 `pkill -f 'obsidian-acceptance-'`，再次以退出码 144 杀掉脚本自身——与 P1 期间记录的完全相同，且我当时刚把这条写进 harness README。
- change: 无生产代码变更；harness 自身不含任何按模式匹配的进程逻辑，缺陷仅存在于人工命令。
- disposition: 该陷阱三次出现均发生在人工验证命令而非产品代码中，说明防线正确地建在 API 形状上（`assertProcessGroupTarget` 只接受数字进程组）。人工清理一律改为读 `/proc/<pid>/cmdline` 定位后按 PGID 回收。
- next: 无。

## 2026-08-28 — P4 complete, goal 达成

- evidence: preflight 在三种阻塞下均以退出码 2 指名原因并给出补救动作，且一次列全（缺 vault、缺 Obsidian、无 PDF 同时报出）；`npm run test:desktop` 真实通过四项场景；默认 `npm test` 不含桌面验收。门禁全部通过：`lint` 0 errors / 64 warnings、`check:boundaries`、`check:product-units`、`npm run build`、`smoke:build`、`check:obsidian-submission`，`test:release-tools` 仅剩既有 flaky，`git diff --check` 干净。完整 workspace 套件 2,753 tests 通过，与 P7 记录基线一致。harness 单元测试 129/129。
- change: P4 五项任务勾选，goal 的八条 success criteria 全部满足。
- disposition: 保留全部四个阶段的实现。harness 位于 `scripts/` 下，不被 plugin bundle 引用，因此 bundle 预算与 boundaries 均不受影响。
- next: 关闭 goal；桌面验收结论可供 `2026-08-17-pdf-hybrid-library-foundation` 的 P7 引用，但本 initiative 不修改其状态。

## 2026-08-28 — L1 adjust: `sidecar-unreachable-falls-back` 是空洞断言，重开 P3

- evidence: 回答分支去留问题时复查该场景，发现两处使其无法支撑 success criterion 6。其一，它直接给 `plugin.settings.pdfParserSidecar.enabled` 赋值，绕过了 `plugin.settingsChanges.changeValue` 这条承载校验、持久化与索引取消的真实事务路径。其二，真实探测发生在 `buildFullTextDocumentParser()` 中，仅赋值不会触发它——该次运行的网络请求计数为 0 即为佐证。因此场景实际证明的只是「改字段不崩」，而非「探测失败回退 PDF.js」。
- change: 将 goal 由 done 改回 active，P3 改回 active，并取消勾选 success criterion 6。场景将改为经真实设置事务启用，再调用 `buildFullTextDocumentParser()`，断言探测请求确实发出、返回 PDF.js fallback 而非 sidecar selector、且未产生 console 错误。
- disposition: 其余三项场景不受影响，其证据（迁移、默认关闭无请求、PDF 第 4 页定位与负向对照）依然成立。此处的教训是负向对照只做了两项：`pdf-page-location` 与 `sidecar-disabled-by-default` 各有反向证据，而 `sidecar-unreachable-falls-back` 没有——缺的恰好是它。
- next: 重写该场景并取得真实探测证据，然后重新收尾 goal。

## 2026-08-28 — 探测证据必须来自真实 socket，而非渲染进程网络域

- evidence: 强化后的场景在真实运行中如实失败：`no probe request to http://127.0.0.1:1 was ever attempted`。追查发现插件的 HTTP 经 `plugin/src/hosts/obsidian/http-client.ts` 走 Obsidian 的 `requestUrl`，在 Electron 主进程发起，渲染进程的 CDP `Network` 域结构上就看不到。因此 `sidecar-disabled-by-default` 的「0 个请求」同样是空洞的——它观察的那一层本就不会有插件流量。
- change: 新增 `probe-listener.mjs`，在 harness 进程内绑定真实 loopback socket 并对所有请求返回 503。两个 sidecar 场景改为把设置指向该监听器：关闭时断言构建 parser 未发出任何���求，启用时断言探测确实到达、被拒绝、且结构上返回 PDF.js 而非 sidecar selector（生产 bundle 类名被压缩，只能用结构判别）。删除 `network.mjs` 及其渲染进程观察，避免它再次诱发同类误判。
- disposition: 一次真实运行同时提供了两侧对照——同一监听器在关闭场景收到 0 个请求、在启用场景收到 1 个并拒绝。这比任何单侧断言都强，且不依赖渲染进程网络栈。
- next: 重新收尾 goal。

## 2026-08-28 — 修复前的中断竞态确实造成了真实数据丢失

- evidence: 最终校验时发现测试 vault 的 `workspace.json` 与原始基线不符，缺少 `showSearch`、`searchQuery` 与一个 ribbon 条目。连续两次运行前后该文件逐字节稳定，说明当前还原正常；漂移来自修复前的那次中断实验——当时信号处理器先还原了文件，而 Obsidian 仍在运行并随后再次写入，正是「必须先回收宿主再还原状态」所要防止的顺序问题。
- change: 从 `/tmp/plugin_test-backup-1787835341/workspace.json` 还原该文件，已确认与 harness 运行前的备份逐字节一致。
- disposition: 该事件把先前记为「理论风险」的竞态变成了已发生的事实，`onInterrupt` 先于 restore 执行的修复因此有真实依据而非推测。保留修复。
- next: 无。

## 2026-08-28 — goal 重新收尾

- evidence: 四项场景在真实 Obsidian 1.11.5 上全部通过，且每项都有反向证据——PDF 页码在 `#page=2` 与 `#page=4` 下分别报 2 与 4；sidecar 关闭时监听器收到 0 个请求、启用时收到 1 个并拒绝后回退 PDF.js；伪造请求会使关闭场景失败；无探测到达会使启用场景失败。harness 单元测试 132/132，完整 workspace 套件 2,753 tests 通过，`lint` 0 errors / 64 warnings，`check:boundaries`、`check:product-units`、`build`、`smoke:build`、`check:obsidian-submission` 通过，`test:release-tools` 仅剩既有 flaky。中断路径经真实 SIGTERM 验证：0 个遗留进程、0 个 Xvfb、0 个沙箱目录。
- change: goal 与 P3 重新标记 done，success criterion 6 重新勾选。
- disposition: 本次重开的教训值得留存——四项场景第一次全绿时，其中两项实际在测量一个结构上不可能观察到目标行为的地方。识破它靠的不是再跑一次，而是追问「这条断言在什么情况下会失败」。负向对照现已覆盖全部四项。
- next: 无。桌面验收结论可供 `2026-08-17-pdf-hybrid-library-foundation` 的 P7 引用；本 initiative 不修改其状态。

## 2026-08-28 — 更正：所谓「既有 flaky」是我的调用方式造成的

- evidence: 本 initiative 多处记录 `test:release-tools` 存在一个既有 flaky——`the real root npm entry sends a Core focus to Core only` 在聚合运行时失败、单独运行通过，并称在干净对照 worktree 上复现。该结论是错的。真实机制是 `npm run <script> --silent` 会为子进程设置 `npm_config_loglevel=silent`，使子 npm 不再打印 `> @arxiv-daily/core@x.y.z test` 这行 banner，而该测试的正向断言恰好匹配 banner 中的 `@arxiv-daily/core`。单独运行之所以通过，是因为没有 npm 父进程注入该变量；对照 worktree 之所以「复现」，是因为我在那里同样加了 `--silent`。不加 `--silent` 时套件 exit 0、0 失败。`npm_config_loglevel=silent node --test scripts/tests/root-test-runner.test.mjs` 可单独复现，确认机制。
- change: 将该测试的正向断言从 npm banner 改为 vitest 自己打印的 `packages/core` 运行路径——它不受调用方日志级别影响；三条反向路由断言保持不变。修改后在有无 `npm_config_loglevel=silent` 两种条件下均通过，完整套件在两种调用方式下均为 0 失败。同时更正各 phase 文件中「仅出现既有 flaky」的验收表述。
- disposition: 此前所有以「仅剩既有 flaky」描述的验收结论应读作「全绿」。这条错误的根源与本 initiative 反复出现的模式相同——观察到一个失败后，我为它编了一个合理的解释（并发敏感），并用一次同样带缺陷的对照实验确认了它，而没有追问「什么条件下它会通过」。
- next: 无。

## 2026-08-30 — L1 adjust: 加 P5（设置页场景 + 截图），并改写「不做截图」这条 non-goal

- evidence: `2026-08-30-library-setup-path` 的 P1c / P1d 改完后只有 happy-dom 单元测试覆盖，用户仍需手动开 Obsidian 肉眼确认分组顺序、按钮集合与就地授权对话框。happy-dom 没有布局引擎也没有 Obsidian 样式表，结构上无法回答「按钮是否在同一行、是否右对齐、是否溢出容器」，而这恰是本次改动最容易出问题的部分。
- change: goal 由 done 改回 active，新增 P5 与一条对应的 success criterion。原 non-goal「截图比对、像素级视觉回归与主题渲染验收」表述不准——它把「落盘截图」和「拿截图做自动比对」混为一谈。改写为：像素级视觉回归与主题渲染验收仍不做（不存基线、不比对、无断言读图），但落盘截图作为人工判断的输入是做的。不装作 non-goal 没变。
- disposition: 新增两个 harness 模块（`library-settings.mjs`、`screenshots.mjs`）与一个新 session；既有四项场景与它们的 fixture 一字未动。截图落 `.acceptance-out/`，已被 `.gitignore` 的 `.*` 覆盖，不入库。零新增依赖，仍用 Node 内置 WebSocket 与 CDP。
- next: P5 的布局几何断言当前为红，原因是被测分支的 CSS 缺陷而非断言问题；按 abort trigger 停在这里汇报，不在本 initiative 内改产品代码。

## 2026-08-30 — P6：`styles.css` 那次堵的是洞，这次堵的是类别

- evidence: 发布资产清单在仓库里有三份互不相干的硬编码——`build-deploy.mjs` 的 `ARTIFACTS`、`docs/release.md` 的 bullet 列表、`release.yml` 的 provenance `subject-path` 与 `gh release create` 位置参数。实测确认「只改一处」的两个危险方向都是静默通过的：只往 `docs/release.md` 加一条 `- \`plugin/extra.js\``，`test:release-tools` 报 `235 / 235 / 0`；只往工作流两处加 `plugin/extra.js`，同样 `235 / 235 / 0`。只改 `ARTIFACTS` 那一侧确实会红，但红在一句写死三条路径的 deep-equal 上——那是清单的第四份副本，它说的是「数组变了」，不是「和发布不一致」。
- change: 新增 `scripts/release-assets.mjs`（`RELEASE_ASSETS` 冻结常量，零 import，验收 harness 直接引用）与 `scripts/release-asset-sources.mjs`（解析文档 bullet 与工作流两份清单，按集合比对，失败信息点名两侧标签、两侧完整清单与多/少的具体文件）。`ARTIFACTS` 改为引用常量。`scripts/tests/release-assets.test.mjs` 32 例。
- disposition: 关键设计点是「解析失败绝不能表现为通过」。一个找不到清单就返回空数组的解析器，会在文档改版当天悄悄变成恒真断言——和它要防的失败模式一模一样。因此 marker 消失 / 重复、bullet 变散文或表格、bullet 形状不对、清单为空、工作流 YAML 不合法或步骤不唯一，一律抛异常；`verifyReleaseAssetSources()` 把异常收成 issue，返回空数组的唯一含义是「每一处都读出了非空清单且互相一致」。常量本身为空也记为缺陷而非「大家都没有资产所以一致」。
- disposition: 三份副本里只有一份成了来源，另两份仍是人写的。`docs/release.md` 没有变成生成物——只在清单下方加了一段说明它被机器读、与哪几处绑定。
- next: 无。P5 不受影响；真实桌面验收在改动后跑过一次，16 条断言全绿，几何数字与 P5 记录逐字一致，说明样式表照旧被部署进去。

## 2026-08-31 — P7：一次被证实的假绿，以及它暴露的三个洞

- evidence: 本机 inotify 配额（上限 65536）被用满，Obsidian 起来后显示自己的错误页——正文 `ENOSPC: System limit for number of file watchers reached, watch '/home/tiandc/Desktop/plugin_test/'`，下面是 `Reload app` / `Open another vault`，设置页根本没渲染。那一轮**17 项断言全部 PASS**（还带着 `description 176px wide, 26.8 characters over 4 lines` 这种精确测量值），**10 张截图全部静静落盘**（内容是错误页文字或纯色空白），整轮只因「渲染进程零错误」判红，退出码是 **1（失败）**而不是 **2（阻塞）**。把配额提到 524288 后同机重跑 17/17 真绿、截图正常——产品一直是好的，坏的是验收看不出环境已经塌了。
- change: 三条守卫。(1) `preflight` 用**功能探测**判监视器余量：`/proc/sys/fs/inotify/max_user_watches` 只给上限不给用量，用满的 524288 和空闲的 524288 读起来一模一样，所以改为在临时目录上真的建 128 个监视再全部释放，`ENOSPC` 判阻塞并给出 `sysctl -w` 与 `/etc/sysctl.d/` 两步修复。(2) 截图改为**先判定后写盘**：目标要在文档里、可见、有尺寸、至少一半落在被拍摄的视口内，回来的 PNG 还要不是整幅一色；拒绝时零文件落盘。(3) 新增 `app-state.mjs`，走查**前后各一次**确认应用处于可走查状态，判据是「有没有挂出 vault 窗口」这一正向能力，错误页文案只被转述不参与判定；后一次检查失败会把已经算出来的结果整个丢弃，因此坏掉的宿主结构上产不出一条 PASS。
- disposition: 纯色判据做了，且只做「整幅一色」这一条——不是比例阈值，不是基线比对。这些截图每张都必然含文字或控件，整幅一色不可能是它声称的状态；而合法的纯色**区域**不受影响，因为规则只在整帧再无他物时触发。为此写了个只用 `node:zlib` 的 PNG 解码器，并实测确认它能读渲染进程真实产出的十张图（8 位 RGB），而不是对每张真图静默弃权——后者等于这条判据不存在。
- disposition: **真实 ENOSPC 本轮未复现**：复现需要 sudo 调低配额，本 initiative 不修改系统配置。三条守卫的红全部来自注入。为补上「守卫是否真的接在真实路径上」这一层，在错误态判据里临时插入一条永不存在的能力跑了一次真实验收，得到退出码 **2**、全程零 PASS、页面内容如实转述；临时改动已撤回。
- disposition: 这与 P6 那次「漏拷 `styles.css` 却一直在量旧样式」是同一类问题，但更危险——那次是漏了一项证据，这次是伪造了一整套，而且带着精确到小数的数字。识破它靠的仍然不是再跑一次，而是追问「这条断言在什么情况下会失败」。
- next: 无。既有 17 项断言与其判据、三个几何常量一字未改，加固后真实验收仍 17/17、退出码 0。

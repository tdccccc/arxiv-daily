# P6 — native-platform-acceptance

goal_ref: ../goal.md
created: 2026-09-29T23:42:06+08:00
updated: 2026-09-30T23:43:06+08:00
revision: 10

## Outcome

在真实 macOS/Windows 与受支持的 Obsidian/Electron 宿主上取得可复现的原生存储、自动邮件防重复和分发验收证据；失败项修复并回归后，才能完成跨平台成功标准及整个 initiative。

## Assumptions

- P5 只验收实现、Linux 证据和 CI 配置就绪，没有把假平台字符串、happy-dom 或 Node 中的 Obsidian adapter 测试当作真实 Electron 结果。
- 当前会话本机只有 Linux；2026-09-30 用户选择 1A，授权推送当前 `fix/review-followups` 分支并创建 PR 触发原生 CI，以及本次验收所需的同分支修复重跑。仍不合并、发布或发送真实邮件；真实 Obsidian/Electron 验收与 CI 第一关分开记录。
- CI 的六个 native runner 使用同一源码的制品；当前 source hash 或二进制变更后，旧平台结果不能继续充当验收证据。
- 只用临时 Vault、假凭证及拦截后的 HTTP 验证，不向真实收件人发信，不修改用户当前 Vault 或线上 relay。

## Approach

取得可用的原生环境后，优先运行已经配置好的 native-storage 工作流或完全相同的本地命令。保留每个平台的 OS/架构、Node/Electron 版本、源码和产物摘要、测试报告及离线安装结果；再补真实桌面宿主加载。环境或发布动作需要额外授权时先停，不把本地提交等同于推送许可。

## Chunks

### Chunk 1 — 原生平台构建、文件系统和离线安装

- change kind: verification; any discovered code/config fix follows strict Red-Green-Refactor
- baseline: 当前缺失 macOS/Windows 观察结果，而不是一个可标为通过的空测试。
- checks: 按 `.github/workflows/native-storage.yml` 在 Intel/Apple Silicon macOS、Windows x64/arm64 上运行官方 SDK 校验、CMake 构建、native `storage.test.cjs`、私有存储/loader/共享锁测试、Node/Plugin adapter composition、离线 bundle 与 npm 安装冒烟。
- acceptance: 每个声明支持的目标都有真实通过结果；目录置换可被拒绝或被 OS 句柄固定阻止，Windows ACL 必须实际验证，不能用 chmod 数字替代。没有跳过全部关键用例或忽略失败。
- [x] native build, storage and installation evidence accepted — source 552f21c; run 36733252120 passes all six real OS/architecture jobs and the aggregate release asset assembly gate, including offline loading/install and artifact uploads.

#### PR #51 repair checkpoints (2026-09-30, L1)

Each fix remains a separate commit; local verification permits a CI candidate commit, not acceptance of an unobserved platform result.

1. **Root dependencies and portable lockfile** — dependency/configuration repair. Observed Red: root audit reports GHSA-82fw-gwwq-j7x9 (`@vitest/mocker`), fast-uri and js-yaml advisories; Linux arm64/macOS jobs cannot load their missing Rollup optional packages. Add a lockfile completeness test and observe Red before regenerating without an installed dependency tree. Select patched compatible releases explicitly, no forced audit fix. Green: unchanged moderate audit threshold, lockfile test, root tests/typecheck/lint/build and release-tool checks; native and Node-version CI must rerun on the new HEAD.
2. **Independent relay dependencies** — dependency repair. Observed Red: relay audit reports the same Vitest advisory plus sharp/undici through Wrangler/Miniflare. Green: independent moderate audit, 161 relay tests and typecheck using explicitly selected patched dependencies; no deploy or provider calls.
3. **Windows delay-load hook** — build bug fix. Observed Red: both Windows jobs report C2373 on `__pfnDliNotifyHook2`. Match the installed MSVC declaration without changing host lookup or access protections. Local compensating check: compile the real source against a narrow declaration fixture and rerun Linux native/build tests; actual Windows build/filesystem/host/offline checks remain mandatory and cannot be replaced by that fixture.
4. **Endpoint normalization** — performance/security bug fix. Observed Red: CodeQL check 109702178627 plus a bounded adversarial slash-input regression before changing production code. Green: linear normalization preserving URL results, provider/model-listing regression and new-HEAD CodeQL check; no suppression or live LLM requests.

5. **Windows exclusive-create directory collision** — native bug fix. Observed Red: run 36680572941, both Windows architectures, native TAP test 4 fails at createFile("link") with EACCES; all other native primitive tests pass and all four Linux/macOS jobs pass. Preserve that failing junction assertion and add ordinary-directory occupancy coverage. Normalize only confirmed directory collisions after CREATE_NEW/ERROR_ACCESS_DENIED; preserve all other permission failures and never follow/open the junction target. Local compensation: Linux native build and 10 primitive tests plus native runtime integration. Windows Green must come from the same existing test on both real runners; this Linux host cannot independently reproduce Win32 calls.
6. **Explicit relay test types** — configuration bug fix exposed after approved dependency installation. Observed Red: tsc cannot resolve node:fs/node:url or ImportMeta.url under Vitest 4. Explicitly include the installed Node types; Green: relay typecheck, all 161 tests, read-only preflight check and Wrangler dry-run.

7. **Claim guard platform assertion** — test contract correction, no production change. Observed Windows x64 Red in run 36713376132: fs.promises.rename succeeds, contradicting the test's unconditional Windows rejection assumption. The retained safety contract already allows either OS prevention or synchronous replacement detection. Exercise actual rename; tolerate only Windows permission/busy errors, otherwise replace the original directory and require guard.assertCurrent to reject the identity change. Verify the original claim content and released-handle rejection in both cases. Linux runtime 46 tests and Node typecheck are the local checks; both Windows runners must verify the revised contract.

8. **Windows test argument forwarding** — workflow repair. Observed Red: run 36713890755 Windows x64 passes primitive and guard/runtime tests, then npm exec runs all 26 Node adapter tests despite -t, and emits no requested JSON reports. Three unrelated POSIX-mode assertions fail. Preserve the intended composition filters and use direct installed Vitest entry points with explicit working directories. Added workflow regression fails before the change and passes afterward (4 tests). Executed the three exact commands on Linux and parsed successful JSON with 20 runtime, 1 Node composition and 1 plugin composition pass. Native Windows runtime/artifact and offline-install checks remain mandatory.

9. **Windows license notice extraction** — build bug fix. Run 36714759289 Windows x64 passes every native/runtime/host test and writes JSON evidence, then both product builds reject the existing pako notice. A temporary CRLF checkout fixture reproduces the same missing-notice Red on Linux. Normalize CRLF on read; all 9 release utility tests pass and exact license content/single banner inclusion remain asserted. Product build, offline native package smoke and offline CLI installation pass locally; Windows downstream execution remains required.

10. **Slow-disk contention test budget** — test harness repair after run 36719082645. Linux x64's 30-increment subprocess case reaches the default 5-second timeout (5029 ms); other cases/platforms pass. Injecting 100 ms per lock-record fsync reproduces the same timeout locally. Give only the two repeated multi-process contention cases a 30-second test budget; retain all exclusion/counter/recovery assertions and production lock deadlines. The same injected-delay run passes both cases in 11.14 seconds total; normal native runtime regression passes 20 tests. Rerun the six-platform CI on the candidate.

11. **Cross-platform source identity and complete assembly** — release build repair discovered by combining real CI artifacts. `readNativeAssets` rejected Windows assets from the otherwise-green matrix. Reproduced hashes match exactly the LF versus CRLF forms of the same source. A checkout fixture and missing-aggregate workflow contract both produced Red. Canonicalize source newlines, retain exact binary digests, and add same-run complete-matrix assembly. All 337 release-tool checks and local build/install checks pass; fresh run 36733252120 passes all seven native jobs and independently downloaded artifacts assemble locally.

After these candidates, retrieve complete logs for any newly exposed native failures and apply the same targeted Red/Green discipline. Keep all six platform gates and desktop acceptance obligations intact.

### Chunk 2 — 真实 Obsidian/Electron 验收

- change kind: verification; behavioral fixes require an observed targeted Red before implementation
- checks: 在隔离 Vault 中安装实际生成的插件三件套，验证匹配原生组件可加载、私有文件读写与恢复、自动投递能力提示及两个宿主重复投递保护。先禁用调度并拦截 HTTP，使用假投递响应，不调用真实 provider。
- acceptance: 记录 Obsidian/Electron/Node/Node-API 与 OS/CPU 版本；实际桌面加载成功，状态切换与失败路径可重现。happy-dom、源码检查和 Node CLI 结果不能替代本块。
- [x] real desktop-host evidence accepted — actual Obsidian 1.11.5 / Electron 39.2.6 on Linux x64, only `/home/tiandc/Desktop/plugin_test`; see ../evidence/desktop-native-acceptance.md. Settings/workspace restored; no real delivery.

### Chunk 3 — 最终回归与关闭

- change kind: verification and documentation; retained bug fixes remain isolated commits
- checks: 对任何原生修复先复现 Red、再观察 Green，并重跑所有受影响平台；执行 root lint/typecheck/test/build/boundaries/submission、release-tools、build/install smoke 及 relay 独立测试/typecheck。
- acceptance: 技术报告与实际结果一致，所有 success criteria 已满足或由用户明确重新决定；不因缺环境或 CI 权限静默豁免。各修复与状态变更分别提交，整个 goal 才能标记 done。
- [x] final cross-platform acceptance and close authorized by evidence — native matrix/assembly, Root, relay, CodeQL and VS Code pass on 552f21c; local release tools 337 and offline build/install pass; both technical reports synchronized and validated.

## Abort / reshape triggers

- 标准原生环境无法满足私有权限、原子替换或同步 namespace guard 时停止，先判断是局部实现修复还是路径需要 L2 调整。
- 需要改变投递身份、恢复规则、Vault 数据可移植性或安全边界时先做 L2/L3 评估，不删除失败测试换取 Green。
- 任何测试开始访问真实 provider、用户 Vault 或线上控制面时立即停止。

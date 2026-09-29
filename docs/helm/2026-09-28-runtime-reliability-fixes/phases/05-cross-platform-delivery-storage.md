# P5 — cross-platform-delivery-storage

goal_ref: ../goal.md
created: 2026-09-29T15:41:13+08:00
updated: 2026-09-29T17:45:20+08:00
revision: 3

## Outcome

Linux/macOS/Windows 的 CLI 与 Plugin product 具备保持现有防重复、崩溃恢复、路径边界和私有存储约束的自动邮件实现，并配置原生平台 CI 验收；实际 macOS/Windows 运行证据留给 P6，不从 Linux 测试推断通过。

## Assumptions

- 现有 generation claim/decision/result 协议和 v1-compatible delivery-state 仍是稳定契约；不通过清除阻断记录或缩短恢复窗口规避问题。
- 普通 Node `wx` 只保证最终路径不被覆盖，不能替代整个父目录链的身份锚定；Windows 不支持 libuv 的 O_DIRECTORY/O_NOFOLLOW，chmod 也不区分 owner/group/others。
- 当前 Linux 的 `/proc/self/fd` 实现不能直接作为 macOS/Windows 实现；跨平台能力需要另外证明，不能只移除平台检查。
- 2026-09-29 用户选择 1A，接受系统原生组件及其随应用分发成本。选用不依赖 V8/Electron ABI 的 Node-API v8 小型 C++ 后端，仅实现目录锚定、私有文件与原子发布，不迁移 Core 投递协议。
- POSIX 使用 openat/renameat/linkat/unlinkat 等目录描述符相对操作；Windows 对祖先目录持有不允许删除共享的句柄，拒绝 reparse point，创建文件即使用当前用户的 protected DACL，并重新验证已打开句柄。
- 官方插件更新器仅安装 main.js/manifest/styles，故发布构建将受控 CI 产出的匹配架构原生字节和摘要内嵌于 bundle；运行时只提取匹配平台字节，不下载代码、不要求终端用户编译。开发构建可只含当前平台，发布构建缺平台必须失败。

## Approach

先以真实文件系统测试验证最小 Node-API 能力，再接入两个宿主。稳定投递协议保留在 Core；系统调用、权限与 namespace guard 留在宿主边界。原生组件缺失、版本不匹配或文件系统能力不满足时明确拒绝自动发送，绝不退回普通 write-before-check。Linux 的既有实现保留为未携带原生资产时的安全兼容路径。

## Chunks

### Chunk 1 — 选择并验证安全存储能力与分发路径

- change kind: behavior change (native primitive and build tooling)
- strategy: strict Red-Green-Refactor; missing native API is the initial contract baseline, followed by a compilable no-op backend to observe behavioral Reds.
- Red / baseline signal: `node --test packages/node-runtime/native/tests/*.test.cjs` cannot acquire a real namespace, publish exclusive private files, or reject moved/symlink parents with the no-op backend.
- Green check: same real-filesystem suite verifies exclusive publication, private permissions/ACL, durable writes, atomic rename, traversal/reparse rejection, opened-parent movement, closed-handle errors and independent child processes.
- regression checks: native build on current Linux, node-runtime suite/typecheck, build-tool contract tests. Mac/Windows execution deferred explicitly to P6; no fake platform result counts.
- [x] approach and feasibility evidence accepted (8fe0c5b; missing API/no-op behavioral Reds, nine native filesystem/subprocess Greens on Linux; native Mac/Windows still unrun)
- build adjustment: use installed CMake/C++ and official Node SDK inputs, not a newly downloaded node-gyp package. Direct compiler Green served as the before-baseline for the equivalent CMake build.

### Chunk 2 — 实现并接入共享的私有投递存储

- change kind: bug fix / behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: host capability/composition tests fail for the selected backend; preserve and extend existing real-filesystem exclusive-create, pre/post-create parent swap, last-read namespace replacement, legacy recovery and private-artifact tests.
- Green check: both hosts advertise the complete capability set only when usable; duplicate automatic sends remain blocked, pre-attempt recovery remains possible, uncertain provider attempts remain blocking, and path replacement prevents HTTP invocation.
- regression checks: Core delivery suites, complete Node/CLI/Plugin suites, root lint/typecheck/build/boundaries/submission and relay tests/typecheck.
- exception: native macOS/Windows execution is unavailable in this Linux session. Linux integration and typed platform contracts are compensating evidence only; native execution must remain explicitly unaccepted until P6.
- [ ] implementation and tests accepted

### Chunk 3 — 打包与 CI 验收配置

- change kind: executable configuration / behavior change
- strategy: strict Red-Green-Refactor for package/config contracts; proportionate inspection for documentation
- Red / baseline signal: package and workflow contract checks expose missing selected support artifacts/platform matrix; tests must reject falsely green jobs that skip all platform-sensitive cases.
- Green check: clean build/install smoke finds the matching backend; native CI matrix covers Linux/macOS/Windows and uploads verification evidence without credentials or real email.
- regression checks: release-tools/package-boundaries checks plus root validation; document exact CI commands and unrun native checks for P6.
- [ ] implementation and tests accepted

## Phase verification

- The user-facing unsupported message changes only after actual host capability wiring is present; a fake platform string is not native evidence.
- Preserve existing delivery-state export/import compatibility and private recipient handling.
- No push, PR, live email, dependency publication, or release is authorized.

## Abort / reshape triggers

- If a candidate only checks path strings around ordinary writes, cannot enforce Windows privacy, or cannot synchronously revalidate the namespace immediately before HTTP invocation, reject it instead of weakening the tests.
- If a backend requires a change to persistent-state portability or delivery semantics, stop for L2/L3 assessment and a user decision.
- If native packaging is not accepted, retain the current safe Linux behavior and leave P5/P6 incomplete; do not waive the goal implicitly.

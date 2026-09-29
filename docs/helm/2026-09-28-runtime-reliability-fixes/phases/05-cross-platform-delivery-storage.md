# P5 — cross-platform-delivery-storage

goal_ref: ../goal.md
created: 2026-09-29T15:41:13+08:00
updated: 2026-09-29T15:41:13+08:00
revision: 1

## Outcome

Linux/macOS/Windows 的 CLI 与 Plugin product 具备保持现有防重复、崩溃恢复、路径边界和私有存储约束的自动邮件实现，并配置原生平台 CI 验收；实际 macOS/Windows 运行证据留给 P6，不从 Linux 测试推断通过。

## Assumptions

- 现有 generation claim/decision/result 协议和 v1-compatible delivery-state 仍是稳定契约；不通过清除阻断记录或缩短恢复窗口规避问题。
- 普通 Node `wx` 只保证最终路径不被覆盖，不能替代整个父目录链的身份锚定；Windows 不支持 libuv 的 O_DIRECTORY/O_NOFOLLOW，chmod 也不区分 owner/group/others。
- 当前 Linux 的 `/proc/self/fd` 实现不能直接作为 macOS/Windows 实现；跨平台能力需要另外证明，不能只移除平台检查。
- 候选路径是一个受限的系统原生存储支持组件。此路径可能增加各 OS/CPU 的构建和随应用分发产物，尚未经用户决定；在决定前不添加依赖、不改变打包方式、不修改投递生产逻辑。

## Approach

先确认是否接受系统原生支持组件及其分发成本，再验证最小私有文件操作能力。稳定投递协议保留在 Core；系统调用、权限与 namespace guard 留在宿主边界。原生组件缺失、版本不匹配或文件系统能力不满足时明确拒绝自动发送，绝不退回普通 write-before-check。

## Chunks

### Chunk 1 — 选择并验证安全存储能力与分发路径

- change kind: non-behavioral investigation; any executable proof then follows strict Red-Green-Refactor
- baseline signal: Linux-only capability checks in both hosts; official Node/libuv documentation confirms missing Windows flags and chmod limitations.
- verification: record the user's packaging decision, select a concrete backend, and define testable create/replace/recover/guard contracts before production edits. No backend has been selected or accepted yet.
- [ ] approach and feasibility evidence accepted

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

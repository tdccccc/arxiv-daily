# P1 — library-row-next-step

goal_ref: ../goal.md
updated: 2026-08-30

## Outcome

设置页文献库一行根据「有没有文件夹 / 本地还是远程 / 有没有授权」给出唯一主按钮：选文件夹、授权、或建索引。

## Assumptions

- 本地嵌入索引不读授权状态（`assertRemoteEmbeddingReady` 只在 remote 时检查）。
- 远程嵌入在未授权时索引会失败，所以远程的主按钮必须先是授权。
- 现有测试锁死「选文件夹后必出 Review & authorize」。那是旧路径，P1 改掉它。

## Approach

抽出「下一步」判定，供 declarative 与 legacy 设置共用。文献库行：未选文件夹 → Choose folder；本地已选 → Build index；远程未授权 → Review & authorize；远程已授权 → Build index。Change folder / Manage / Revoke 保持次要。Build index 调用现有 `indexPersonalLibraryFullText`。

## Test strategy

- change kind: behavior change
- strategy: strict Red–Green–Refactor
- Red / baseline signal: `settings-declarative-tab` 在默认 local、authorization-required 时仍断言主按钮是 Review & authorize
- Green / regression checks: 该文件覆盖 local/remote 两种下一步；plugin typecheck；相邻 library-connection 测试
- exception: 无

## Tasks

- [ ] 用测试固定：local + 已选文件夹 → Build index；remote + 未授权 → Review & authorize；remote + 已授权 → Build index。
- [ ] 文献库行按该判定渲染主按钮与说明。
- [ ] Build index 从设置页启动现有索引命令路径。
- [ ] 授权说明区分本地不离机 / 远程发全文。

## Verification

- 上述三种状态的按钮文案与 CTA 位置有测试。
- 未选文件夹仍只有 Choose folder。
- 命令面板索引入口仍可用。

## Abort / reshape triggers

- 如果本地索引其实仍强制授权，停下来先修授权契约，不要在 UI 上撒谎。
- 如果设置行放不下三个按钮，把 Revoke 收回 Manage，不要藏起主按钮。

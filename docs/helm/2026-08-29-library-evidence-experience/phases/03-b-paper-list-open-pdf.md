# P3b — paper-list-open-pdf

goal_ref: ../goal.md
updated: 2026-08-29

## Outcome

Dashboard、Similar papers 和命令搜索的文献结果只列出相关论文；有本地 PDF 时打开整份文件，不再展示段落或声称跳到某一页。

## Assumptions

- 论文排序已有固定评测；段落与页码没有。继续展示会把半成品当证据。
- Core 检索仍返回 hits；本阶段只改 plugin 产品面与打开路径。
- P1 的 vault-root `this` 绑定必须保留，否则打开会再次崩。
- 打开整份 PDF 用现有 opener 的无页码路径，而不是伪造 page=1。

## Approach

共享 renderer 改为论文行：标题、文件名、一个 `Open PDF`。打开回调不再接收 hit/page。`openPersonalLibraryFullTextEvidence` 改为打开文件本身。现有「段落/页码」测试改写为反向对照：正文和 `Page N` 不得出现。

## Test strategy

- change kind: behavior change
- strategy: strict Red–Green–Refactor
- Red / baseline signal: library-search-block / similar-papers / command modal 仍断言段落和 `Open PDF at page N`
- Green / regression checks: 上述测试改为论文列表 + `Open PDF`；lifecycle 打开路径不再带 `#page=`；plugin typecheck
- exception: 无

## Tasks

- [x] Renderer 只渲染论文行，隐藏 hits。
- [x] 打开动作打开整份 PDF；失败通知仍带具体原因。
- [x] 改写断言段落/页码的测试，并加反向对照。
- [x] 保留 P1 的 `getBasePath.call(adapter)` 回归。

## Verification

- 有 hits 的 fixture 不出现段落、章节、`Page N`。
- `Open PDF` 调用 opener 时不带页码。
- 无 `filePath` 时没有打开按钮。
- Observed Green (2026-08-29): plugin 44 tests（renderer / similar-papers / command modal / pdf-opener / lifecycle）与 typecheck 通过。有 hits 的 fixture 不出现段落、`Page N` 或 `Open page N`；opener 默认无 `#page=`。

## Abort / reshape triggers

- 如果打开整份 PDF 在 Vault 内仍失败，停在 opener，不要把段落加回去。
- 如果用户重新要求页码跳转，另开带评测的阶段，不在本阶段半做。

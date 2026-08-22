# P5 — 证据结果 UI 与 PDF 跳转

goal_ref: ../goal.md
updated: 2026-08-22

## Outcome

全文搜索的每个结果可展示论文、章节、原文片段和可证明的页码；用户可从 Dashboard、相似论文和命令入口打开对应本地 PDF，并在宿主不接受精确定位时仍清楚保留页码。

## Assumptions

- `KnowledgeBaseChunkHit` 已带稳定 `headings`、`locator`、一基页码和原文；P5 只投影这些证据，不改 chunk、generation、分数或排序。
- 索引记录的 PDF 路径相对已确认的 personal library root；vault 内文件可使用 Obsidian 的 PDF page subpath，外部库文件使用受控的 `file:` URL page fragment。
- 当前 PDF.js 路径没有可靠 bbox/highlight contract；页面跳转失败或宿主忽略 fragment 时，UI 的明确页码是必需降级而不是伪造高亮。

## Approach

先让 plugin 的全文检索结果为所有 indexed paper 保留可打开的 library-relative PDF 路径和 hit metadata。抽出纯 DOM evidence-result renderer，使用文本节点展示 heading、截断但可辨识的 passage 和 `Page N`，并把打开动作交给 host callback。Dashboard block、Library similar tab 与命令搜索结果共用该投影。最后添加受限 PDF opener：只接受已索引的相对路径，vault 内走 Obsidian page subpath，外部库走 canonical root 下的编码 `file:` URL；所有不能确认的精确跳转都保留 `Page N` 并退回打开文件。

## Test strategy

- change kind: user-visible behavior change in plugin presentation and host opening
- strategy: strict Red–Green–Refactor per behavioral chunk
- Red / baseline signal: renderer tests先因 evidence fields/action 不存在失败；lifecycle tests先因 full-text results丢失 hit/path 或 opener不可调用失败
- Green / regression checks: focused renderer/lifecycle/adapter tests，随后 Plugin、Core、Node suites，workspace typecheck、boundaries、production build和`git diff --check`
- exception: actual Obsidian PDF viewer interpretation of a `#page=N` fragment cannot be asserted in Vitest; unit tests verify the exact vault subpath/file URL and the UI always exposes the fallback page number

## Tasks

- [ ] 为全文匹配保留所有 indexed paper 的 library-relative PDF 路径和 `KnowledgeBaseChunkHit`，以失败测试锁定不改变排名与 fallback-file 行为。
- [ ] 建立并验收可复用 evidence-result renderer：安全文本投影论文、章节、原文片段、页码和每个结果的打开动作。
- [ ] 将 evidence renderer 接入 Dashboard library matches 与 Library similar tab，保留异步加载、错误和 stale-response 行为。
- [ ] 将命令搜索从 Notice 摘要升级为可操作的结果 surface，并验收空结果、查询错误和 action error 不逃逸。
- [ ] 实现并验收 vault 内与外部 selected library 的 PDF opener：拒绝非 logical path，使用 page target，不能精确导航时可靠回退到打开文件并保留页码。
- [ ] 完成 P5 定向与全量回归，并记录实际 Obsidian viewer 仍需在 P7 跨平台验收的边界。

## Verification

- Pending implementation.

## Abort / reshape triggers

- 如果支持外部 library PDF 必须扩大 `ScopedLibrarySource` 的读取权限、跟随符号链接或暴露任意绝对路径，停止并 reshape opener 边界。
- 如果页码/heading/片段投影改变 retrieval score、rank、chunk identity 或引入 query-time 全文读取，停止并回到纯 presentation adapter。
- 如果 Obsidian host 没有可测试的 page navigation primitive，保留打开文件 + 明确页码的降级，不伪造坐标高亮。

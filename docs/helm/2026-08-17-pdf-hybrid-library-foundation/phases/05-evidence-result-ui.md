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

- [x] 为全文匹配保留所有 indexed paper 的 library-relative PDF 路径和 `KnowledgeBaseChunkHit`，以失败测试锁定不改变排名与 fallback-file 行为。
- [x] 建立并验收可复用 evidence-result renderer：安全文本投影论文、章节、原文片段、页码和每个结果的打开动作。
- [x] 将 evidence renderer 接入 Dashboard library matches 与 Library similar tab，保留异步加载、错误和 stale-response 行为。
- [x] 将命令搜索从 Notice 摘要升级为可操作的结果 surface，并验收空结果、查询错误和 action error 不逃逸。
- [x] 实现并验收 vault 内与外部 selected library 的 PDF opener：拒绝非 logical path，使用 page target，不能精确导航时可靠回退到打开文件并保留页码。
- [x] 完成 P5 定向与全量回归，并记录实际 Obsidian viewer 仍需在 P7 跨平台验收的边界。

## Verification

- P5 observed Red：现有 Dashboard block、Library similar tab 和命令 Notice 都丢弃 `KnowledgeBaseChunkHit` 的 headings、text 与 page；PDF page opener 和可操作 command result modal 也不存在。随后 action-error handler 自身抛错会从 DOM event handler 逃出，Node builtin path/URL 方案又被 workspace boundary check 拒绝。
- P5 Green：所有 indexed match 投影稳定保留 catalog title 覆盖、ranking 字段、hits 和首个 library-relative PDF path。三个入口复用纯 DOM evidence renderer，显示章节、截断 passage、`Page N` 和只在本地 PDF 可用时出现的打开动作；async/error/stale 行为不变。打开前先确认当前 manifest 仍绑定 paperKey/path，并以 selected source 的一字节范围读取复核 root/logical-path/no-symlink 边界；vault PDF 使用 `#page=N`，外部库使用编码 `file:` URL，vault fragment 失败时退回同一文件且页码仍可见。路径 helper 仅使用 plugin-safe 字符串语义，覆盖 Unix、Windows drive、UNC 与 unsafe path，不引入 Node builtin。
- P5 verification：定向 Plugin 6 files / 40 tests、完整 Plugin 41 files / 616 tests、8 GiB heap Core 111 files / 2,000 tests、Node runtime 3 files / 45 tests、CLI 7 files / 71 tests 通过；workspace typecheck、`check:boundaries`、`check:product-units`、production build 与 `git diff --check` 通过。`npm run lint` 无 error，仍为既有 65 warnings/60 cap；elevated `smoke:build` 仅复现既有 bundle `canvas` forbidden-text；Obsidian submission check 仍由既有 1.53 MB bundle 超过 1 MiB 阈值阻止。实际 Obsidian PDF viewer 是否解释 `#page=N` 仍留 P7 跨平台验证。
- P5 technical-report handoff：`no-impact`；此阶段只使已存在的全文 evidence 可见、可操作，不改变 catalog、方向、日报或 consent 的领域关系。

## Abort / reshape triggers

- 如果支持外部 library PDF 必须扩大 `ScopedLibrarySource` 的读取权限、跟随符号链接或暴露任意绝对路径，停止并 reshape opener 边界。
- 如果页码/heading/片段投影改变 retrieval score、rank、chunk identity 或引入 query-time 全文读取，停止并回到纯 presentation adapter。
- 如果 Obsidian host 没有可测试的 page navigation primitive，保留打开文件 + 明确页码的降级，不伪造坐标高亮。

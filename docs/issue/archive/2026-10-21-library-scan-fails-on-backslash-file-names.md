# 文献库文件夹里只要有一个文件名带反斜杠，整个库的扫描和索引都会失败，错误信息还不指出是哪个文件

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 LIB-03
> 范围：个人文献库目录扫描（macOS、Linux 上的插件、CLI、Workbench） · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-21-batch1`，提交 74a02f2、19603de） · 置信度：已复现

## 问题

在 macOS、Linux 上，文件名里可以合法地包含反斜杠。常见来源是：在 Windows 上打的压缩包，被不识别路径分隔符的工具解压后，会得到名为 `old\2301.00002.pdf` 这样的单个文件。文献库文件夹里只要出现一个这样的条目（任何文件类型，PDF 或笔记都一样；目录名带反斜杠也一样），"扫描文献库"和"建立索引"（索引会先扫描）就都会失败，信息是：

- 扫描：`cannot persist invalid personal library catalog`
- 建立索引（插件）：`Could not scan the library folder before indexing: cannot persist invalid personal library catalog`；CLI 的 `library index` 直接显示不带前缀的原始信息

信息里没有文件名，用户无从判断哪里有问题；库里其它几百篇论文也全部不可用，直到把那个文件改名或移走。库根目录下以"单个字母加冒号"开头的名字（如 `A:b 2301.00002.pdf`）同样触发。

复现：用真实的 Node 目录清单（`openScopedLibrarySource`）打开一个临时文件夹，其中有 `papers/2610.00001.pdf` 和 `old\2301.00002.pdf`。清单返回 `["old\\2301.00002.pdf","papers","papers/2610.00001.pdf"]`，`LibraryWorkflow.scan()` 抛出 `cannot persist invalid personal library catalog`。把文件名换成 `notes\todo.docx`（非 PDF）结果相同。

## 原因

三层对"合法的逻辑路径"的定义不一致，而且清单这一层没有过滤：

- `packages/node-runtime/src/scoped-library-source.ts:87-125`：`inventory()` 把 `readdir` 返回的名字原样拼成逻辑路径，不检查。
- `packages/core/src/library/personal-library-reconciliation.ts:175-242`：`reconcilePersonalLibraryCatalog` 为每个清单条目（包括 `unrelated`）生成记录，不检查路径。
- `packages/core/src/library/personal-library-catalog.ts:458`、`626-634`：`decodeFileRecord` 用 `isLogicalRelativePath` 拒绝含 `\`、NUL、以 `/` 开头、以 `X:` 开头的路径，整个目录文档因此解码失败；`replace` 抛出上面的错误。
- 同一条件也让读取失败：`scoped-library-source.ts:314-329` 的 `validateLogicalPath` 拒绝含 `\` 的路径。

## 可能的解决方案

推荐：

1. 在 core 导出路径判定函数（`isLogicalRelativePath`），`reconcilePersonalLibraryCatalog` 和 `detectPersonalLibraryFolderChange` 都跳过不合法的条目，并把数量记入扫描摘要（例如 `lastScan.skipped`，需要作为可选新增字段，不升级目录 schema 版本）。
2. Node 目录清单把这类条目标成 `type: "ignored"`（新增 `ignoredReason: "unsafe-name"` 或沿用 `unsupported-entry`），遇到目录名不合法时不再下钻，预览窗口也能把它列出来。
3. 界面和 CLI 在摘要里提示"有 N 个文件因名称无法处理被跳过"，并给出第一个路径，用户知道该改哪个文件。

备选：只做第 1 点，扫描不再整体失败，但用户仍看不到被跳过的文件。

## 修复后应补的测试

- core：清单里带一个 `a\b.pdf` 和一个 `notes\x.md`，扫描成功，其它论文正常入库，被跳过的条目不写入目录。
- node-runtime：临时文件夹里创建 `old\x.pdf`，`inventory()` 把它标成 ignored，`readBinary("old\\x.pdf")` 仍被拒绝。
- 回归：同一份清单连续扫描两次，第二次 `detectPersonalLibraryFolderChange` 不因被跳过的条目报告变化。

## 修复情况

文件夹里出现名字带反斜杠（或以 `X:` 开头）的文件或文件夹时，扫描和建立索引不再整体失败：Node 目录清单把这类条目标成 `ignored`（新增原因 `unsafe-name`）且不进入这样的文件夹，对账和"文件夹有无变化"检查也会跳过它们，所以它们既不进目录，也不会被当成新增而让每次索引都重扫。跳过的数量记入扫描摘要的可选字段 `lastScan.skipped`（为 0 时不出现，目录 schema 不变），CLI 的日志警告会列出前三个路径，库目录预览列出全部并说明原因。可选的后续：插件的"目录摘要"对话框还没有 `skipped` 一行，插件自己的扫描路径也不输出这条日志。

测试：core 的 `personal-library-reconciliation`、`personal-library-folder-change`、`personal-library-catalog`（路径判定与摘要解码）、`library-connection`（预览）、`library-workflow`（日志，索引不把它们当作变化）；node-runtime 的 `scoped-library-source.test.ts` 在真实临时文件夹里验证清单、拒读，以及"扫描得到有效目录且之后视为未变化"（Windows 上跳过）。

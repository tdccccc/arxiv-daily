# VS Code 扩展面板按裸 arXiv 编号查找论文，而 papers.json（schema 5）的键是 `arxiv:<id>`：状态修改和所有打开按钮都失效

> 发现日期：2026-10-18 · 来源：main `21fe2ce` 深度代码审查 · 编号 P3
> 范围：`extensions/vscode-arxiv-daily` · 严重程度：高 · 状态：已修复（第一部分：分支 `fix/2026-10-18-review`，提交 89c2cc2；文件锁：分支 `fix/followup-vscode-lock`） · 置信度：已复现

## 问题

当前 core 写出的 `papers.json`（`PAPER_INBOX_SCHEMA_VERSION = 5`）以 `paperKey`（如 `arxiv:2606.00001`）为键。VS Code 面板中每一行的操作都会失败并提示"arXiv Daily: 2606.00001 is not in papers.json"：状态下拉框（`setStatus`）以及 Note / Daily / arXiv / PDF 四个按钮（`openResource`）。只有列表和筛选能用。

附带问题：

- 每次操作后重新赋值 `panel.webview.html`，页内状态（标签页、搜索、筛选）被重置。
- 即便查找成功，`updatePaperStatus` 也是对 `papers.json` 不加锁、非原子的读改写（`JSON.stringify` 直接写回），绕过 core 的 `PaperIndexStore`（变更队列、索引文件锁、`.bak/.tmp` 替换、schema 规范化）。与 Obsidian / CLI 并发写时可能丢更新，崩溃时可能截断文件。
- `openResource` 把 `entry.arxivUrl` / `pdfUrl` 直接交给 `vscode.env.openExternal(Uri.parse(...))`，不限制 scheme。工作区自带的 `papers.json`（如克隆的仓库）可以让按钮打开任意 URI scheme。

扩展测试只用 `schemaVersion: 2`、按裸编号作键的夹具，CI 里看不出问题。

## 原因

- `extensions/vscode-arxiv-daily/src/dashboard-model.js:24-27` 以 `entry.arxivId`（裸编号）构造行，webview 回传该编号。
- `extensions/vscode-arxiv-daily/src/dashboard.js:77-112` 用 `index.papers?.[arxivId]` 查找。
- core 自 schema 4 起在 `normalizeInbox`（`packages/core/src/services/paper-index.ts:931-940`）把条目改键为 `paperKey`。
- 测试夹具：`extensions/vscode-arxiv-daily/scripts/dashboard-test.mjs:13`。
- 扩展自行实现了索引读写与面板模型，与 core 重复。

## 可能的解决方案

推荐（本次）：
1. 按映射键或 `arxivId` 查找条目，行数据携带映射键。
2. `schemaVersion` 不是扩展理解的版本时拒绝写入。
3. 写入改为同目录临时文件 + rename 的原子替换。
4. `openExternal` 只允许 https（最好限定 `https://arxiv.org/` 或由编号重建 URL）。
5. 测试改用 schema 5、`arxiv:<id>` 作键的夹具。

后续：状态修改改走 core 的 `PaperIndexStore`（打包 core 或调用 CLI），共享文件锁。目前未与 core 共用文件锁，作为已知局限保留。

## 修复后应补的测试

- schema 5 夹具下 `setStatus` 与 `openResource` 成功。
- 非 https 的 `arxivUrl` 被拒绝打开。
- 未知 `schemaVersion` 时写入被拒绝。

## 修复情况

按映射键或 arXiv 编号查找条目；未知 `schemaVersion` 拒绝写入；`papers.json` 经临时文件 + rename 原子替换；打开链接由 arXiv 编号重建，存储的 URL 仅在为 https arxiv.org 时使用；页面更新改用 `postMessage`，标签页、搜索和筛选不再被重置；测试改用 core `PaperIndexStore` 生成的 schema 5 夹具。

**文件锁（后续项，已解决）**：扩展现在把 core 和 node-runtime 一起打包进 `out/extension.js`（只有 `vscode` 留在外面），状态修改直接用 core 的 `PaperIndexStore`，和 Obsidian、CLI 走同一套读写：同一把跨进程文件锁（锁目录名由库的真实路径加 `paper-index:arxiv-daily/.index/papers.json` 算出，三端一致）、同一个变更队列、`.bak` 备份和临时文件替换。因此别的程序正在改 `papers.json` 时，扩展的修改会等它放锁之后再读再写，双方的修改都会保留，不会再互相覆盖。扩展里自己写的原子写入代码和 schema 版本白名单已删除，面板列表的读取也改用 core。

保留的行为：按映射键或 arXiv 编号查找；打开链接仍由编号重建、只用 https；页面更新仍用 `postMessage`，标签页、搜索和筛选不会被重置。损坏的或版本比扩展新的 `papers.json` 不会被改写：core 遇到这种文件会退回 `.bak` 再保存，扩展在写之前先检查，一旦 core 实际读到的是备份就拒绝并提示用 Obsidian 或 CLI 处理。

构建与测试：扩展新增 `npm run build`（esbuild 打包，需要先在仓库根目录 `npm ci`），`vsce package` 会自动重新打包，VSIX 只含 `package.json`、README 和 `out/extension.js`（约 0.9 MB，因为带着 core）。新增锁互斥测试：测试里扮演另一端持有 core 的锁，扩展的写入会等待，放锁后双方的修改都在，并核对两端用的是同一个锁目录。扩展的 CI 工作流现在也先安装根目录依赖，并在 core、node-runtime 变化时触发。

已知限制：只支持本地文件夹工作区（虚拟工作区会提示不支持）；Obsidian 在 Windows 而扩展在 WSL 等不同主机/命名空间时，锁各自独立，这是 NodeFileLock 本身的范围。

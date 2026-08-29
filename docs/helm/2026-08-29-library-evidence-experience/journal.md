## 2026-08-29 — start

- evidence: 用户在 plugin_test 的 Library matches / Similar papers 上认为证据结果难看，且 `Open PDF at page 4` 通知失败。隔离会话复现：`searchPersonalLibraryFullText("attention is all you need")` 第一名 `arxiv:1706.03762`，插件 `openPersonalLibraryFullTextEvidence` 抛 `TypeError: Cannot read properties of undefined (reading 'basePath')`；同一文件 `openLinkText("small_library/1706.03762.pdf#page=4")` 打开嵌入式 viewer 第 4 页。`desktopVaultRoot` 将 `adapter.getBasePath` 抽成无绑定函数。现有测试用 arrow mock，覆盖不到。
- change: 新建本 goal。P1 修打开与失败可见性；P2 证据卡片呈现；P3 默认主证据去噪声。不把索引吞吐或真增量纳入本目标。
- disposition: 前一 initiative `2026-08-17-pdf-hybrid-library-foundation` 保持 done。并行 discovery / email helm 不改。
- next: P1 先观察 method-style mock 的 Red，再绑定 `this` 读取 vault root。

## 2026-08-29 — P1 done, start P2

- evidence: `desktopVaultRoot` 改为 `getBasePath.call(adapter)`。Lifecycle Red 为 method-style mock 的 `basePath` TypeError；Green 后 33 项 plugin 定向测试通过。隔离 Obsidian 上 `openPersonalLibraryFullTextEvidence` 打开 vault 内 `small_library/1706.03762.pdf` 第 4 页报 4、第 2 页报 2。失败通知改为带 `error.message`。
- change: P1 done，激活 P2。勾选打开与失败可见性两条成功标准。
- disposition: 生产改动仅 vault-root 绑定与通知文案。不改检索。P2 只改共享 evidence renderer 的层次与样式。
- next: 把证据块改成卡片结构，按钮短文案，正文压缩空白与点线。

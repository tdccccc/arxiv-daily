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

## 2026-08-29 — P2 done

- evidence: 共享 renderer 改为标题 / 文件名 / 引用式证据块；按钮可见文案 `Open page N`，aria-label 仍带页码；正文空白折叠、点线压成省略号、截断 180 字。定向 plugin 19 tests 与 typecheck 通过。未改 hit 选择或论文排序。
- change: P2 done。P3 仍 pending：版权页、`<EOS>`/`<pad>`、参考文献作为默认主证据。
- disposition: 测试 vault 的 `plugin_test` 插件文件未从本会话覆盖；用户需拷贝 worktree 的 `plugin/main.js` 与 `plugin/styles.css` 后重载。
- next: 等用户看过 P1+P2 再决定是否做 P3 的噪声 hit 选择。

## 2026-08-29 — L3 steer

- evidence: 用户确认卡片和打开可用，但匹配质量没有量化。现有评测只锁论文排序（Recall/MRR/nDCG），不锁段落。用户选择：只展示相关论文并打开整份 PDF，不展示具体内容、不跳到对应页，避免把半成品当产品。
- change: 修订 intent 与成功标准。P3 噪声 hit 选择 superseded。新 P3b：产品面论文列表 + 打开整份 PDF。CONTEXT.md「Library similarity」与 ADR 0006 改为检索仍可有 hit，默认 UI 不展示段落。
- disposition: 保留 P1 opener 绑定与失败通知。P2 卡片层次可复用标题/文件名，段落/页码/Open page N 从产品面移除。Core 检索与论文排序不改。
- next: 改 renderer 与 opener，先观察段落断言的 Red。

## 2026-08-30 — start

- evidence: PR 40 已合入 main。用户反馈不是检索引擎本身，而是选库 → 嵌入 → 授权 → 索引 → 搜索被拆成五段，建索引只在命令面板。现状：选文件夹后主按钮永远是 Review & authorize；授权后文案写「Metadata and abstracts authorized」；本地嵌入其实不需要授权也能 `indexPersonalLibraryFullText`。
- change: 新建本 goal。P1 只改设置页文献库一行的说明和主按钮。不撤回 PR 40。
- disposition: 检索、打开 PDF、generation 索引保持现状。发现闭环 / email helm 不改。
- next: 先写失败测试：本地模式下选文件夹后应出现 Build index，而不是必须授权。

## 2026-08-30 — P1 done, start P2

- evidence: 设置页文献库行按 embedding.mode 给出下一步。local + 已选文件夹主按钮为 Build index；remote 未授权为 Review & authorize，授权后为 Build index。Revoke 收回 Manage。plugin 设置/连接测试 62 项 + 新增判定 3 项与 typecheck 通过。
- change: P1 done，激活 P2。勾选成功标准 1–5。
- disposition: 命令面板索引入口保留。P2 再处理索引中的进度与「去 Dashboard 搜索」的可见下一步。
- next: P2 让建索引进行中与完成后的下一步可感知。

# Journal

## 2026-08-05 — note（P1 完成；环境怪癖与运行时未决项）

- evidence: P1 全文索引与相似检索完成并验收。核心验证：core 84 文件/1421 测试、plugin 382 测试、lint 0 error、typecheck/boundaries 通过；真实 5 篇 arXiv PDF 端到端（提取→分块→嵌入→存储→标题自检索 5/5 top-1 → 重建逐位一致 → 离线缓存可用）。
  - **本机测试环境怪癖**：本 worktree 机器上 `npm test`（根）因 core 套件 vitest worker OOM（默认 4GB V8 堆不足，`tests/pipeline.test.ts` 等组合即可触发）返回 EXIT=1——**预先存在，与 P1 代码无关**（排除 fulltext 测试仍复现）。完整跑 core 套件需 `NODE_OPTIONS=--max-old-space-size=8192 npx vitest run`（84 文件/1421 全过）。CI 不受影响（不同内存画像）。后续会话验收时注意区分。
  - T4 subagent 中途模型层失败（无产出报告），但留下完整实现与调试产物；owner 补齐验证（冒烟 6/6 PASS、修复 smoke 确定性检查的 batch 形状比较、修 6 个 lint error）后验收。
  - e5 嵌入确定性的正确判据是"同 batch 形状两次嵌入逐位一致"；batch=3 vs batch=1 存在预期浮点差异（ONNX 内核重排），非 bug。
  - Obsidian 运行时未决项（代码已注释 + 报告中按"仓库配置行为"表述）：`window.pdfjsLib` 在插件上下文的实际可用性、wasm 从 CDN 拉取在渲染进程的表现——需用户在 Obsidian 中实际运行 `index-personal-library-fulltext` 确认；Node 侧同一代码路径已全链路实测。
- change: 无 steer；goal.md P1 标 done；technical-report 已 update（全文知识库机制 + 插件宿主接线）。
- disposition: 全部保留。tmp/（pdf-extraction-validation、embedding-smoke、fulltext-e2e）为 scratch 验证资产，不入库。
- next: P2 聚类方向生成（goal.md 索引行 pending）；检索入口形态（Dashboard similar-papers 按钮 vs 检索栏）open question 待定；CLI host 的提取/嵌入实现仍为后续阶段。

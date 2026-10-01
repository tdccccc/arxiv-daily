# 0.5.0 中断修复收尾与发布准备

status: active
created: 2026-10-01T23:06:37+08:00
updated: 2026-10-02T01:01:26+08:00
revision: 5
owner: codex-root-2026-10-01

## Intent

保留已完成的可靠性修复，整合 feat/library-directions-drive-daily 的标题摘要索引、主题方向和日报流程。通过本地检查后交付新构建手测，再完成 0.5.0 版本同步与发行说明。

## Success criteria

- [x] 首次建索引按需扫描；已有 catalog 不自动重扫；取消与整体失败不继续索引（自动测试通过，真实宿主待手测）。
- [x] 索引只处理标题摘要；旧全文索引不被错误复用；arXiv查询失败的本地PDF仍可参与摘要搜索和方向生成（回归通过，实际资料库耗时待手测）。
- [ ] 审核对话框大屏有足够空间，小窗口可以滚动与操作。
- [x] 两个修复独立提交，要求的检查有实际记录，备份后部署手测文件且不碰 data.json。
- [ ] 用户手测通过后，完成版本同步、发行说明核对和发布前本地检查与提交。

## Non-goals

- 用户已授权本次本地整分支合并；push、修改PR、合并到main、tag、发布和清理仍需逐项授权。
- 不触碰任何已有 .worktree，包括 claude-code-research-plugi 与 agent-product-strategy。
- CLI 设置新功能明确放后续版本；先完成 0.5.0 收尾与手测，届时已授权独立本地提交。

## Constraints

- 分支 fix/review-followups；接手 HEAD f0662a2；保留全部现有未提交修改。
- 子任务不得继续派生；文件编辑范围不重叠；子任务不提交。
- 本轮修复已授权逐项本地提交；提交前看 staged diff；Why / What / Validation 使用多个 -m。
- 发行说明草稿留到版本同步提交；部署只复制三件资产，不启动 Obsidian。

## Phases

1. P1 — 两个中断修复通过本地验证、独立提交并部署手测 — status: done
2. P2 — 合并后手测通过再完成 0.5.0 元数据与发行说明、本地发布前验证 — status: pending
3. P3 — 整分支融合、标题摘要重建与新主题方向流程通过回归并部署 — status: done

## Open questions

- 合并后的构建需重新手测；P1旧构建验收不能代替新主题方向流程的验收。

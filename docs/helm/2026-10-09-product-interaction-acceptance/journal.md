# Product interaction acceptance journal

## 2026-10-09 — note

- evidence: 用户明确暂不做内容质量评测，要求减少每次改版的手动点击；随后授权创建同级 worktree，并选择模型 API 无人值守探索。
- change: 从 main 42e8335 创建 test/ui-regression；本目标跨 Reading workbench 与 Plugin product，独立于历史 desktop harness 的特定布局验收和旧工作台功能目标。
- disposition: 复用既有真实桌面会话、CLI、核心 pipeline 与普通测试；不修改其他 Helm 的 owner/status。新测试使用自有临时 vault，证据保存在 output/playwright/acceptance/。
- next: P1 的共享夹具/报告、工作台、Obsidian 与 API 探索模块在同一阶段独立实施并由父会话验收。

## 2026-10-09 — L1 adjust: explicit retry evidence

- evidence: 工作台真实启动后检查失败恢复链，发现 failed_permanent 被 isDone 视为结束，现有“重试日期”仍派发普通日期运行，可能跳过用户明确要求的重试。
- change: P1 增加独立 bug-fix chunk，先取得 workbench HTTP/CLI 失败回归，再修复最窄的手动重试入口。
- disposition: 保留自动调度对永久错误停止重试及已成功日期普通运行幂等的语义；不靠清除测试状态使场景通过。
- next: 工作台代理复现并修复；父会话继续统一入口与报告整合。

## 2026-10-09 — note: shared acceptance foundation checkpoint

- evidence: 共享夹具与报告先观察 12 个目标契约 Red，再 12/12 Green；统一入口先观察选项、执行与证据保留 Red，再 9/9 Green；公共 npm 入口先缺失，再能列出套件。既有生命周期与 runner 回归 43/43；boundaries、product inventory、diff 检查通过。
- change: 接受 P1 Chunk 1。实现已分为 3d564be（隔离夹具与报告）、05b0e62（统一命令与浏览器依赖）两个提交。
- disposition: 保留真实宿主构建与运行在各自模块中，父入口仅组织独立夹具、配置、执行、证据和清理。尚未把两端完整 UI 或真实模型调用标为通过。
- next: 完成宿主与探索模块，依据实际结果进入 P2 联合验收。

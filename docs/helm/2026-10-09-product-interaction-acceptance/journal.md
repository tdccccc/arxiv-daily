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

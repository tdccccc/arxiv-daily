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

## 2026-10-10 — L1 adjust: search blur overrides reading

- evidence: run-rcbLIe 的真实浏览器请求先查询列表、打开论文，再由失焦 change 的延迟搜索重新查询列表，导致阅读页被覆盖。
- change: P1 加入独立搜索导航修复；在组件测试中复现浏览器 input/change 顺序和未结束的 debounce。
- disposition: 保留真实验收的正常点击节奏；不通过增加人为停顿或自动重新打开论文掩盖产品缺陷。
- next: 修复重复查询事件与离开列表时的待执行搜索，再跑阅读链。

## 2026-10-10 — L1 adjust: host component disabled state

- evidence: obsidian-button-debug-WNZIgu 的真实监听器闭包显示 ButtonComponent.disabled=true、buttonEl.disabled=false。构建与部署哈希一致；通过组件 setDisabled(false) 后原样点击立即关闭日期对话框并进入真实 running，排除了选择器和配置问题。
- change: P1 加入日期对话框修复；测试 mock 补齐真实宿主内部禁用守卫，再从公共入口验证有效日期的点击与 Enter。
- disposition: 保留原生点击与对话框关闭断言，不靠测试修改组件状态来让验收通过；定向修改仅用于因果诊断，不进入正式场景。
- next: 产品通过组件接口同步禁用状态，再继续完整 Obsidian 流程。

## 2026-10-10 — note: P1 accepted, begin integrated verification

- evidence: 工作台 run-D9hPId 为 12/12；Obsidian run-Mq0Tdz 为 8/8；真实模型 run-LKWQzP 为 3/3（19 API calls、70.329 秒、105706 tokens），受控模型浏览器同样 3/3。核心2456、Node runtime73、CLI454、Plugin1022测试通过，另2个依赖真实语料的Core测试跳过；全部typecheck通过，lint为0 errors/16原有warnings。
- change: 接受 P1 所有实现块；已分别提交夹具、统一入口、两端场景、模型探索及三个产品修复。P2 开始处理完整命令、CI、文档和启动就绪稳定性。
- disposition: 保留所有真实失败与修复证据，不把早期fixture格式/文案/启动时机错误计为产品缺陷。已有产品修复为失败日期手动重试、搜索失焦抢占阅读、日期按钮组件禁用状态。
- next: 应用有界工作区就绪等待，补CI契约并运行最终默认验收。

# Architecture, Code Organization, and UI Audit

status: active
updated: 2026-08-10
owner: zcode-main-session

## Intent

基于当前实现深度检查 arXiv Daily 的架构边界、代码组织和前端设计，识别真正值得投入的改进，并形成有证据、可排序、可执行的审查结论。

## Success criteria

- [ ] 架构和代码组织结论覆盖主要产品、共享 Core、宿主边界及工程基础设施，并由当前源码或配置支撑。
- [ ] 前端审查覆盖 Dashboard、设置页及全局样式，明确视觉更新是否必要及其范围。
- [ ] 高优先级发现经过主会话复核，包含影响、证据位置、建议方案和投入优先级。
- [ ] 审查报告落入 `docs/reviews/`，可直接作为后续改进阶段的输入。

## Non-goals

- 本事项不直接实施大规模重构或视觉改版。
- 不对外部服务进行破坏性、负载或真实投递测试。
- 不重复收录已经解决、无法在当前源码中复现的历史问题。

## Constraints

- 审查工作在独立 worktree 和 `review/architecture-code-ui-audit` 分支完成。
- 以源码、测试、构建配置和可观察验证为准；技术报告仅用于定向。
- 建议需兼顾 Obsidian 原生设计语言、现有 Node CLI 和共享 Core 边界。

## Phases

1. P1 — 形成经复核的架构、代码组织与前端设计审查报告 — status: active

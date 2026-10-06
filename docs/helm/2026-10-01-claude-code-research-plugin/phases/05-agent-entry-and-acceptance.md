# P5 — Auxiliary agent entry and final acceptance

goal_ref: ../goal.md
created: 2026-10-01T21:22:11+08:00
updated: 2026-10-01T21:49:36+08:00
revision: 2

## Outcome

用户可直接运行完整产品，也可通过Claude Code辅助调用同一CLI；文档明确基础日报/详报优先、文献库可选、模型与数据边界，并有真实宿主证据。

## Assumptions

- 不增加第二套agent业务执行或记录格式；仅扩充已实现命令的工作流说明。
- 模型工具选择是外部非确定性行为，不做逐字prompt断言；真实宿主是否调用产品命令及实际产物作为证据。
- 桌面阅读界面、生产发布、多平台CPU模型验收不在本次本地实验完成范围。

## Chunks

### Chunk 1 — Align user instructions and commands

- change kind: documentation and agent integration
- strategy: proportionate command-contract validation and real-host verification
- check: README/Skill/reference给出同一binary、CLI TOML、正式Vault路径；列出基础与可选库任务、授权/恢复语义，移除过时“不支持文献库”的描述。
- exception: 不以随机模型输出制造Red；业务命令的Red/Green在P3/P4已验证，外部模型调用用真实会话和产物检查补偿。
- [x] instructions accepted — README/Skill/reference覆盖当前完整业务命令；基础日报/详报优先，库增强可选，明确配置、模型费用、授权、缓存、版本冲突与现有数据格式。

### Chunk 2 — Actual Claude Code invocation

- change kind: integration verification
- strategy: isolated host smoke
- check: 真实Claude Code加载Skill后调用产品CLI的status、完整日报/详报任务、papers，并读取同一输出；HTTP可用明确fixture避免生产数据/付费模型配置，不能让agent自己写测试日报。
- check: 报告本机Claude网关/模型可用性，测试级临时参数不修改全局配置。
- [x] host acceptance complete — 真实Claude Code 2.1.285加载研究Skill后调用产品status、run --date、run --id、papers；生成1份日报、2份详报、3条Paper Index记录，未直接写报告或初始化个人库。

### Chunk 3 — Final checks and handback

- change kind: non-behavioral acceptance
- check: 全量tests、typecheck、lint、build、boundaries、product-units、插件严格校验；本地commit与干净worktree；报告实际验证与限制。
- [x] final verification complete — 全量3088 tests passed，2项既有skip；6个独立进程/打包测试passed；build/typecheck/lint/boundaries/product-units/Obsidian submission/plugin strict validation全部通过，lint保留21条既有告警。

## Observed final evidence

- 完整测试日志：本地忽略文件 `.artifacts/final-tests-installed.log`。worktree初始借用父目录依赖可以运行测试；Obsidian构建的固定browser alias要求本地node_modules，因此运行npm ci安装锁定依赖后重新完成build、typecheck和全量测试。
- 真实宿主证据：`.artifacts/claude-core-smoke/verification.json` 与session日志。模型暂用现有网关兼容模型 `gpt-6-astra-cc-format[1m]`；未改全局模型配置。产品HTTP通过显式preload fixture，故此验收证明宿主调用真实业务流程，不证明真实供应商生成质量。
- Node实际解析/推理证据沿用P4：四份PDF原字节保持，CPU e5索引/混合检索正确；Node20.19冷进程离线加载也通过。
- 未执行生产发布/推送/部署、真实Obsidian桌面操作、Windows/macOS CPU运行或同库跨宿主并发重建。本地实验完成不代表这些后续验收完成。

## Abort / reshape triggers

- 如果Skill开始自行筛选/写权威报告，回到“完整产品命令”的接口，不放宽为聊天生成。
- 如果真实宿主不可用，保留未完成状态并说明实际阻碍，不把fixture或manifest校验冒充宿主验收。

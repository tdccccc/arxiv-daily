# P2 — bulk-confirmation

goal_ref: ../goal.md
created: 2026-08-31T22:59:11+08:00
updated: 2026-08-31T22:59:11+08:00
revision: 1

## Outcome

复审页可以一次接受一组方向候选、一次落盘：默认全选，证据单薄的默认不勾，划掉不想要的之后一个动作确认完；确认后 confirmed interest profile 非空。

## Assumptions

- **批量确认可由现有的单条确认折叠而成**。单条确认已经是纯变换 `(proposal, profile) → (proposal, profile)`，把前一条的输出喂给下一条即可得到批量结果，全部既有校验（文档兼容、目录清单、lineage 冲突、方向数上限）自动沿用。若折叠时发现某条校验只对「一次一条」成立，这条假设即失效。
- **一次落盘、一次 CAS**。中途失败则整批不落盘，proposal 与 profile 不会出现「confirm 了一半」的中间态。
- **「证据单薄」= 代表论文少于 2 篇**。一篇论文是一篇论文，不是一个方向。实测测试库 22 个候选里代表论文数为 1 的恰好只有 `Quantum PCP` 那一条，与 ADR 0009 §3 举的例子一致。阈值放在 core 导出，界面不自带魔法数字。若真实库里出现大量单篇候选，说明阈值或聚类粒度需要重新考虑。
- 候选的草稿取其自身文本（名称/描述/线索/代表论文），批量接受不改写任何一条；要改文本仍走单条编辑。

## Approach

core 新增「确认多条」的纯函数与对应的带存储版本，折叠既有单条确认后只写一次；新增并导出「证据单薄」判定。复审页 proposed 标签页每个候选前加勾选框（默认勾上，单薄的默认不勾且带标注），底部一个「接受选中的」按钮走批量确认。单条确认与编辑入口保留不动。

## Chunks

### Chunk 1 — core 一次确认多条候选，一次落盘

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 新增测试——两个候选一次确认，断言 profile 得到 2 个方向、proposal 候选清空、存储只发生一次写入；再加一条中途失败（第二条 lineage 冲突）的用例，断言整批不落盘且 proposal/profile 保持原状。红在函数不存在。
- Green check: `npm run test --workspace @arxiv-daily/core -- personal-library-interest-profile-review`
- regression checks: 单条确认的既有断言逐条仍绿（含幂等恢复分支）；`npm run typecheck`。
- **observed**: 两条纯变换测试与存储测试都先红在函数不存在；「整批失败」那条一开始**为错误的原因通过**（`toThrow()` 把「不是函数」也算通过），收紧为要求 `code: "conflict"` 后才真正变红。绿之后 review 15/15、store 36/36，其中既有的协调器断言全部改走批量路径仍绿——这是「单条 = 一元批量」这次重构行为保持的证据。
- exception: 无
- [x] implementation and tests accepted

### Chunk 2 — 复审页一次接受一组，证据单薄的默认不勾

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 新增测试——proposed 标签页渲染出勾选框，代表论文 ≥2 的默认勾上、=1 的默认不勾且带「证据单薄」标注；点「接受选中的」只提交勾上的那些，且只调用一次批量确认。红在没有勾选框与批量按钮。
- Green check: `npm run test --workspace obsidian-arxiv-daily -- personal-library-interest-profile-modal`
- regression checks: 单条确认/编辑/合并/禁用入口的既有断言仍绿；插件全量与 `npm run typecheck`。
- **observed**: 红的原文 `expected [ false, false ] to deeply equal [ true, false ]` 与 `missing button Accept selected`。绿之后 modal 23/23、plugin 696/696、core 各分片全绿、typecheck 四包、lint 0 error、boundaries OK。
- **撞坏一条既有测试，改的是源码不是测试**：提案加载失败时可能既没有 proposal 文档、又不走两条提前返回，新的预选逻辑必须守这个分支。
- **两处自行做的判断，留给用户推翻**：其一，「接受选中的」**不再弹二次确认**——逐卡片的确认按钮是一次无防护点击才需要弹框，而这里勾选本身就是深思，且接受后仍可编辑/禁用/移除。其二，proposed 页的勾选框标签从「Select for merge」改为「Select」，一个选择同时服务接受与合并；因此合并按钮在默认全选下会变为可点，但它本来就有一个写明数量与名称的确认框。
- exception: 无
- [x] implementation and tests accepted

## Phase verification

- **observed 2026-08-31**：core 各分片全绿；plugin 696/696（41 文件）；`npm run typecheck` 四包；`npm run lint` 0 error（20 warning，既有）；`npm run check:boundaries` OK。
- **不构成交付证据**：复审页只有 happy-dom 单测，桌面验收目前只覆盖设置页，这个模态框从未在真实 Obsidian 里被验收过。勾选框的实际可用性、默认勾选是否一眼看得懂、批量按钮的位置，必须由用户开一次看过才算数。
- **本阶段不证明**：确认后日报能跑（那是 P3 的前置检查）与推送质量（P5）。

## Abort / reshape triggers

- 若折叠单条确认时发现校验相互干扰（例如方向数上限对中间态判定过早），停下重塑（L2）：改为先整体校验再一次性构造，而不是继续折叠。
- 若用户看过复审页后认为「默认全选」比「单薄的默认不勾」更好，按 L1 就地改默认值并同步 ADR 0009 §3。

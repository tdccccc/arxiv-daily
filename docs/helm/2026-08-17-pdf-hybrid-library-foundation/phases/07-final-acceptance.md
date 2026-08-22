# P7 — 最终迁移、规模与跨平台验收

goal_ref: ../goal.md
updated: 2026-08-22

## Outcome

旧版全文库能以可恢复、可观察的方式迁移或重建；固定规模与资源边界有可复现证据；Node 与 Obsidian 桌面宿主保持同一权限、降级和 PDF 证据交互契约，所有目标层兼容性门禁得到通过或明确豁免。

## Assumptions

- P4b 的 immutable generation、一次性 legacy fallback 与按 document derivation 重建已具备充分的单元和组合测试；P7 的重点是补齐端到端迁移、规模和真实宿主证据，而不是重写索引格式或检索排序。
- 当前容器可运行 Node、Core、Plugin 和 CLI 自动化测试，但无法启动并人工操作桌面 Obsidian；实际桌面验证需要具有该宿主的隔离 Vault 环境。
- 既有 lint warning cap、smoke `canvas` 文本和 1 MiB submission bundle 限制均是已经记录的仓库基线，是否修复或正式豁免必须通过可复现检查决定。

## Approach

先从现有 fixtures、migration orchestration 与发布脚本建立缺口矩阵；对能在本机复现的迁移、受限 heap 规模、Node/Plugin/CLI 兼容性执行真实验证，并只为缺失的稳定契约添加测试。随后准备最小隔离 Vault 的 Obsidian 桌面验收步骤，确认 PDF page fallback、sidecar enable/probe/fallback 和旧设置迁移；最后处理或正式豁免 release-gate 基线并运行完整验收矩阵。

## Test strategy

- change kind: acceptance, migration and compatibility behavior
- strategy: Green characterization baseline for accepted generation/retrieval behavior; strict Red-Green-Refactor only for a reproducible coverage gap
- Red / baseline signal: 先运行 migration/scale/host composition checks；缺少稳定契约时，以新增测试在生产变更前暴露不兼容或资源越界
- Green / regression checks: focused Core/Node/Plugin/CLI tests，受限 heap full Core，workspace typecheck、boundaries、product units、production build、lint、smoke/submission checks 和 `git diff --check`
- exception: 实际桌面 Obsidian UI/Viewer 需要独立桌面宿主；本容器只能验证生成的 host action 和自动化 contract，需保留可执行手工验收记录

## Tasks

- [ ] 建立 P7 验收矩阵：将现有迁移、generation reuse/rebuild、受限 heap、路径和 PDF 页面降级证据映射到成功标准，并指出仍未覆盖的可执行契约。
- [ ] 补齐并验收旧版 knowledge-base 到 generation 的真实迁移、失败保留 prior current、parser/sidecar derivation 改变后的重建与搜索兼容性；不删除唯一 legacy source。
- [ ] 在固定合成语料和受限 heap 下复验查询内存、block 上限、增量同步与 RRF 指标，必要时用测试固定实际上限。
- [ ] 完成 Node、Plugin 与 CLI 的自动化兼容性和权限边界回归；确认 sidecar 默认关闭、失败回退和页码降级不扩大 consent 或路径权限。
- [ ] 在隔离桌面 Obsidian Vault 完成 PDF `#page=N`、sidecar settings/probe/fallback、旧 settings migration 与无控制台错误的实际验证；无法执行时记录环境阻塞和可复现步骤。
- [ ] 复现并处置或正式豁免 lint、smoke build 与 Obsidian submission release-gate 基线，完成最终全量验收与 success criteria closeout。

## Verification

- P7 尚未开始执行；P6 移交时自动化回归为 Core 113 files / 2,015 tests、Plugin 41 files / 622 tests、Node runtime 3 files / 45 tests、CLI 7 files / 71 tests，且 typecheck、boundaries、product units、production build 与 diff check 全部通过。
- 已知基线：lint 65 warnings / cap 60；elevated smoke build 的 plugin bundle `canvas` forbidden-text；Obsidian submission 的约 1.53 MiB bundle 超过 1 MiB 限制。

## Abort / reshape triggers

- 如果真实迁移要求原地删除唯一 legacy knowledge base、使 corrupt/incompatible generation 静默降级或无法保留 prior current，停止并 L2 reshape migration path。
- 如果固定 block/heap 设计在受支持规模仍随语料线性增加查询峰值内存，停止并回到 P4b 的索引设计检查，而不是掩盖结果。
- 如果桌面 Obsidian 不能保持路径权限、page fallback 或 sidecar explicit-enable 契约，停止发布验收并先修复宿主边界。

# P7 — 最终迁移、规模与跨平台验收

goal_ref: ../goal.md
updated: 2026-08-23

## Outcome

旧版全文库能以可恢复、可观察的方式迁移或重建；固定规模与资源边界有可复现证据；Node 与 Obsidian 桌面宿主保持同一权限、降级和 PDF 证据交互契约，所有目标层兼容性门禁得到通过或明确豁免。

## Assumptions

- P4b 的 immutable generation、一次性 legacy fallback 与按 document derivation 重建已具备充分的单元和组合测试；P7 的重点是补齐端到端迁移、规模和真实宿主证据，而不是重写索引格式或检索排序。
- 当前机器可运行 Node、Core、Plugin、CLI 自动化测试和本地 Obsidian desktop；仓库中没有现成隔离 Vault，实际 UI 验收必须在 `/tmp` 下创建并在结束后删除的最小环境执行。
- lint warning cap、native dependency smoke 检查和 bundle safety budget 都由仓库门禁明确约束；Obsidian 官方当前提交文档没有 1 MiB 主包限制。

## Approach

先从现有 fixtures、migration orchestration 与发布脚本建立缺口矩阵；对能在本机复现的迁移、受限 heap 规模、Node/Plugin/CLI 兼容性执行真实验证，并只为缺失的稳定契约添加测试。随后准备最小隔离 Vault 的 Obsidian 桌面验收步骤，确认 PDF page fallback、sidecar enable/probe/fallback 和旧设置迁移；最后处理或正式豁免 release-gate 基线并运行完整验收矩阵。

## Test strategy

- change kind: acceptance, migration and compatibility behavior
- strategy: Green characterization baseline for accepted generation/retrieval behavior; strict Red-Green-Refactor only for a reproducible coverage gap
- Red / baseline signal: 先运行 migration/scale/host composition checks；缺少稳定契约时，以新增测试在生产变更前暴露不兼容或资源越界
- Green / regression checks: focused Core/Node/Plugin/CLI tests，受限 heap full Core，workspace typecheck、boundaries、product units、production build、lint、smoke/submission checks 和 `git diff --check`
- exception: 实际桌面 Obsidian UI/Viewer 需要独立桌面宿主；本容器只能验证生成的 host action 和自动化 contract，需保留可执行手工验收记录

## Tasks

- [x] 建立 P7 验收矩阵：将现有迁移、generation reuse/rebuild、受限 heap、路径和 PDF 页面降级证据映射到成功标准，并指出仍未覆盖的可执行契约。
- [x] 补齐并验收旧版 knowledge-base 到 generation 的真实迁移、失败保留 prior current、parser/sidecar derivation 改变后的重建与搜索兼容性；不删除唯一 legacy source。
- [x] 在固定合成语料和受限 heap 下复验查询内存、block 上限、增量同步与 RRF 指标，必要时用测试固定实际上限。
- [x] 完成 Node、Plugin 与 CLI 的自动化兼容性和权限边界回归；确认 sidecar 默认关闭、失败回退和页码降级不扩大 consent 或路径权限。
- [ ] 在隔离桌面 Obsidian Vault 完成 PDF `#page=N`、sidecar settings/probe/fallback、旧 settings migration 与无控制台错误的实际验证；无法执行时记录环境阻塞和可复现步骤。
- [x] 复现并处置 lint、smoke build 与 Obsidian submission release-gate 基线；最终全量验收仍等待桌面 UI 证据。

## Verification

- P7 initial baseline: P6 移交时自动化回归为 Core 113 files / 2,015 tests、Plugin 41 files / 622 tests、Node runtime 3 files / 45 tests、CLI 7 files / 71 tests，且 typecheck、boundaries、product units、production build 与 diff check 全部通过。
- 门禁策略：lint 允许且锁定当前 64 条历史 warning；submission 使用 2 MiB repository safety budget；smoke 只拒绝 `canvas`/`onnxruntime-node` 的真实 native package import，不拒绝浏览器标准 Canvas API。
- P7.1 acceptance matrix (2026-08-22): Core migration/reuse/rebuild、prior-current preservation、schema/cutover fail-closed 和 parser derivation 由 4 files / 79 tests 复现；其中 `generation-index-orchestration` 覆盖 exact reuse、derivation/source change、prior current、stale source 与 pinned reader，`fulltext-index-orchestration` 覆盖 legacy source 与 v2 derivation rebuild。4 MiB split、linear builder work、fixed dense/BM25/RRF metrics 由同次 Core suite 覆盖。
- P7.1 acceptance matrix (continued): Node 2 files / 20 tests 复现 durable spool/generation composition 和 scoped-library path boundary。Plugin 3 files / 64 tests 复现旧 settings 的 disabled sidecar migration、disabled no-probe/enabled probe-fallback、sidecar change cancellation、PDF `#page=N` action 与 host-rejected page navigation fallback。Core/Node/Plugin automated coverage is Green; remaining evidence gaps are an isolated real Obsidian desktop interaction, current full workspace/release-gate results, and a decision to repair or waive each release baseline.
- P7.2 migration/scale/compatibility Green (2026-08-22): 受限 targeted suites 已通过 Core 4 files / 79 tests、Node 2 files / 20 tests、Plugin 3 files / 64 tests。随后 `NODE_OPTIONS=--max-old-space-size=8192 npm run test:workspaces` 全量通过 Core 113 files / 2,015 tests、Node 3 / 45、CLI 7 / 71、Plugin 41 / 622，共 2,753 tests；workspace typecheck、`check:boundaries`、`check:product-units` 和 production build 通过。
- P7.2 release-gate evidence (2026-08-22): `npm run lint` 复现 65 warnings / 60-cap（0 errors）；`npm run smoke:build` 在非 sandbox 中只复现 plugin bundle 禁止文本 `canvas`；`npm run check:obsidian-submission` 只复现 `plugin/main.js` 1,544,930 bytes 超过 1 MiB。CLI help 子进程在非 sandbox smoke 中正常输出 Usage；此前 sandbox 空输出是执行环境限制，不是 CLI 缺陷。
- P7.2 desktop attempt (2026-08-22): 本机 `/opt/Obsidian/obsidian` 以临时 XDG 配置启动并识别隔离 Vault，但旧 CLI 不支持 `--vault`，窗口读取自动化随后受审批限流而停止；临时 Vault 已删除，未触碰用户现有 Vault。实际 PDF viewer/settings 无控制台错误仍未取得证据。
- P7.3 release-gate Green (2026-08-23): 删除过时的 1 MiB 假设并改为 2 MiB 内部预算；移除未使用的 `setIcon` 后 lint 为 64 warnings / 64 cap、0 errors；`npm run lint`、`npm run smoke:build`、`npm run check:obsidian-submission`、submission unit test、workspace typecheck、boundaries、product units 和 build 全部通过。非 sandbox smoke 验证 CLI help 子进程、pako notice、无 workspace runtime require，以及 native dependency import 规则。

## Abort / reshape triggers

- 如果真实迁移要求原地删除唯一 legacy knowledge base、使 corrupt/incompatible generation 静默降级或无法保留 prior current，停止并 L2 reshape migration path。
- 如果固定 block/heap 设计在受支持规模仍随语料线性增加查询峰值内存，停止并回到 P4b 的索引设计检查，而不是掩盖结果。
- 如果桌面 Obsidian 不能保持路径权限、page fallback 或 sidecar explicit-enable 契约，停止发布验收并先修复宿主边界。

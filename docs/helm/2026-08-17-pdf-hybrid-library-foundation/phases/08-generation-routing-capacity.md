# P8 — generation-routing-capacity

goal_ref: ../goal.md
updated: 2026-08-28

## Outcome

generation 索引能够为真实规模的个人文献库构建并服务；路由引用与对象数各自受其真正的约束限制，且该容量由具备真实词汇密度的测试固定，而非由合成语料掩盖。

## Assumptions

- 实测数据成立：每个 dictionary block 恰好产生 256 个路由引用（真实词汇散布于全部桶），约每 126 chunk 一个 block。据此 22,819 chunk 需约 182 block、约 46,592 个路由引用。
- 路由表的真正约束是 descriptor 字节上限（`MAX_GENERATION_DESCRIPTOR_BYTES` = 1 MiB），不是对象数。以路径存储约 1,638 KiB 超限，以对象序号存储约 228 KiB 可容纳。
- 对象序号在 descriptor 的 `objects` 数组中已经存在且稳定有序，可作为路由目标而无需引入新的标识。
- 读取侧 `generation-bm25-index.ts` 目前按路径字符串比对路由（`descriptor.lexicalRouting[bucket].includes(reference.path)`），改为序号后需同步调整并保持既有 BM25 排序结果不变。
- 现有真实语料均为 legacy v1 提升所得，无法据此区分触发因素；需要以当前解析器/分块器产出的 v2 语料复核（本阶段任务之一）。

## Approach

先用具备真实词汇密度的 fixture 把缺陷固定为可复现的 Red——这是本缺陷得以长期存在的根因，必须先补上。随后把路由表由对象路径改为对象序号，并为路由引用引入独立于对象数的预算，其上限由 descriptor 字节约束反推。格式随之升版，读取侧与校验同步更新。最后以真实语料复核构建与检索，并单独定位重投影性能。

## Test strategy

- change kind: bug fix（容量记账缺陷）+ 随附的格式变更
- strategy: strict Red–Green–Refactor；Red 必须来自真实词汇密度的语料而非三词 fixture
- Red / baseline signal: 新增 fixture 以每 chunk 数百个不同词项构建约 2,000+ chunk 的语料，`buildFullTextGeneration` 以 `object-limit` 失败——即当前实现在真实密度下的确定性复现
- Green / regression checks: 同一 fixture 构建成功且 BM25/dense/RRF 排序与修复前在小语料上的结果逐项一致；`generation-*` 与 `fulltext-*` 全部既有测试保持通过；完整 workspace 套件、typecheck、boundaries、product-units、build 通过
- exception: 无

## Tasks

- [ ] 以真实词汇密度的 fixture 复现 `object-limit`，并固定「每 block 消耗 256 个路由引用」这一实测关系，使其成为回归测试而非一次性观察。
- [ ] 拆分预算：为路由引用引入独立上限（由 descriptor 字节约束反推），不再复用 `MAX_GENERATION_OBJECTS`；对象数仍按对象计。
- [ ] 路由表由对象路径改为对象序号，升 `GENERATION_DESCRIPTOR_SCHEMA_VERSION`，同步更新编解码、校验与读取侧路由匹配，保持 BM25 结果不变。
- [ ] 版本兼容：旧版 descriptor 的处理路径明确（拒绝并重建，或就地升级），不得静默误读。
- [ ] 以真实 legacy 语料（199 篇 / 22,819 chunk）完成构建，并验证 generation 与 legacy 两条检索路径的排序等价。
- [ ] 用当前解析器/分块器产出的 v2 语料复核，确认缺陷与 legacy 提升路径无关。
- [ ] 定位重投影性能：3 篇 197 秒的量级不合理，给出原因与结论（修复或记录为已知特征）。

## Verification

- Red：真实密度 fixture 在修复前以 `object-limit` 失败，失败点与实测的 16 block 边界一致。
- Green：同一 fixture 构建成功；真实 199 篇语料构建成功并与 legacy 路径排序等价；既有 generation/fulltext 测试全绿。
- 容量：路由引用数与 descriptor 字节均在各自上限内，且有测试固定实际余量。
- 门禁：完整 workspace 套件、typecheck、`check:boundaries`、`check:product-units`、production build 通过；`git diff --check` 干净。

## Abort / reshape triggers

- 如果以对象序号存储后 descriptor 仍无法容纳真实语料的路由表，停止并回到路由表设计本身（例如按桶分片存储或稀疏表示），而不是继续放宽上限。
- 如果格式升版无法在不破坏既有 generation 的前提下完成，停止并先确定兼容策略，不得让旧 descriptor 被静默误读。
- 如果 v2 语料复核显示缺陷仅出现在 legacy 提升路径，停止并把问题重新定位到该路径，而不是改动格式。

# P8 — generation-routing-capacity

goal_ref: ../goal.md
updated: 2026-08-28

## Outcome

generation 索引能够为真实规模的个人文献库构建并服务；路由引用与对象数各自受其真正的约束限制，且该容量由具备真实词汇密度的测试固定，而非由合成语料掩盖。

## Assumptions

- 缺陷可在不依赖任何真实语料的情况下复现：合成语料只要具备真实词汇密度即可，三词 fixture 不能。
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
- Red / baseline signal: 合成密集语料已复现该缺陷且与真实语料吻合——40 chunk×2000 词项得 5 个 block / 1,280 refs，80 chunk 得 10 block / 2,560 refs，160 chunk 以 `object-limit` 失败，即每 block 恰好 256 refs、上限 16 block
- Green / regression checks: 同一 fixture 构建成功且 BM25/dense/RRF 排序与修复前在小语料上的结果逐项一致；`generation-*` 与 `fulltext-*` 全部既有测试保持通过；完整 workspace 套件、typecheck、boundaries、product-units、build 通过
- exception: 无

## Tasks

<!-- 顺序经 2026-08-28 实测调整：吞吐缺陷先于容量修复，因为它决定了容量回归测试能否在可接受时间内运行。 -->
- [x] 定位并修复构建吞吐：实测约 2,400 词项/秒，且成本按词项而非按 chunk（10×8000 与 160×500 耗时相同）。该速率使任何触发 16-block 上限的测试需处理约 110 万词项、耗时约 8 分钟，因此它直接阻塞容量缺陷的可测性。
- [x] 以真实词汇密度的 fixture 复现 `object-limit`，并固定「每 block 消耗 256 个路由引用」这一实测关系，使其成为回归测试而非一次性观察。
- [x] 拆分预算：为路由引用引入独立上限（由 descriptor 字节约束反推），不再复用 `MAX_GENERATION_OBJECTS`；对象数仍按对象计。
- [x] 路由表由对象路径改为对象序号，升 `GENERATION_DESCRIPTOR_SCHEMA_VERSION`，同步更新编解码、校验与读取侧路由匹配，保持 BM25 结果不变。
- [x] 版本兼容：旧版 descriptor 的处理路径明确（拒绝并重建，或就地升级），不得静默误读。
- [ ] 以真实 legacy 语料（199 篇 / 22,819 chunk）完成构建，并验证 generation 与 legacy 两条检索路径的排序等价。
- [ ] 用当前解析器/分块器产出的 v2 语料复核，确认缺陷与 legacy 提升路径无关。
- [ ] 复核修复后的真实语料构建耗时，确认其与语料规模的关系可接受。

## Verification

- Red：真实密度 fixture 在修复前以 `object-limit` 失败，失败点与实测的 16 block 边界一致。
- Green：同一 fixture 构建成功；真实 199 篇语料构建成功并与 legacy 路径排序等价；既有 generation/fulltext 测试全绿。
- 容量：路由引用数与 descriptor 字节均在各自上限内，且有测试固定实际余量。
- 门禁：完整 workspace 套件、typecheck、`check:boundaries`、`check:product-units`、production build 通过；`git diff --check` 干净。

## Abort / reshape triggers

- 如果以对象序号存储后 descriptor 仍无法容纳真实语料的路由表，停止并回到路由表设计本身（例如按桶分片存储或稀疏表示），而不是继续放宽上限。
- 如果格式升版无法在不破坏既有 generation 的前提下完成，停止并先确定兼容策略，不得让旧 descriptor 被静默误读。
- 如果 v2 语料复核显示缺陷仅出现在 legacy 提升路径，停止并把问题重新定位到该路径，而不是改动格式。

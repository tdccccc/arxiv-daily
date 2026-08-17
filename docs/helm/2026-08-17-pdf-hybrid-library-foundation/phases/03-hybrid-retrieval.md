# P3 — BM25、dense 与 RRF 混合检索

goal_ref: ../goal.md
updated: 2026-08-18

## Outcome

固定、确定性的评测集可分别衡量 dense、BM25 与 hybrid 的 Recall@k、MRR 和 nDCG；Core 在不改变知识库 schema 的前提下，以 chunk-level BM25 和现有 centered dense 分别产生论文级候选，经论文级 RRF 融合为最终排序，并保留精确标题、短查询 alias、相似论文和证据定位语义。

## Assumptions

- P3 继续查询期逐 paper 加载；BM25 新增工作集有界，但整体 dense JSON/base64 装载与 centering 内存留给 P4。
- BM25 以 chunk 为文档、以 paper 为候选单位，每篇仅向 RRF 投一票。
- 中文词法 baseline 使用无依赖、确定性的 Han bigram；跨语言语义仍主要由 dense 召回。
- 现有 `searchKnowledgeBase` 保留为兼容 dense/legacy scorer，新建 hybrid 入口供生产 orchestration 使用。
- 文件型 KB schema、plugin UI、sidecar 和问答均不在本阶段修改。

## Approach

先实现 Unicode/CJK tokenizer 和 query-term-only 两遍 BM25，再实现论文级 RRF 与证据合并。将 title 和 compact alias 归入 lexical rank，而不是与 cosine 直接比数值。建立带 graded judgments 的纯 Core 评测 harness，最后把默认查询编排切换到 hybrid，同时保留 dense/lexical 模式用于基线和诊断。

## Test strategy

- change kind: user-visible ranking behavior plus pure retrieval infrastructure
- strategy: strict Red–Green–Refactor per chunk; current dense/title/compact tests are the compatibility baseline
- Red / baseline signal: tokenizer/BM25、RRF、metrics 和 hybrid orchestration tests 分别先因 API 或行为不存在失败；生产接线前现有 dense tests 保持 Green
- Green / regression checks: 每块运行新增测试与相邻 lexical/dense regression；阶段末完整 core/plugin、typecheck、boundaries、plugin build 和 deterministic evaluation gate

## Tasks

- [x] 实现并验收确定性 Unicode/CJK tokenizer、chunk-level BM25、论文聚合与有界 lexical top-k。
- [x] 实现并验收论文级 RRF、候选限制、稳定 tie-break 与 EvidenceChunk hit 去重合并。
- [x] 将 title exact/prefix 与短查询 compact alias 纳入 lexical rank，保留长 title+abstract 相似论文的 dense 语义。
- [x] 建立并验收固定 corpus/judgments、Recall@k、MRR、nDCG 与按类别对照；hybrid 总体不劣于 dense，且至少一个词法类别严格提升。
- [x] 查询 orchestration 默认接通 hybrid，保留 dense/lexical 模式及 model/cancel/corrupt/empty guards，完成阶段回归与规模检查。

## Verification

- P3 定向 Core：最终 7 个文件、86 项测试通过；包含 tokenizer/BM25、RRF 去重后截断、独立 cursor 证据合并、显式 lexical query 和 guards。
- 固定 corpus 由真实 `searchKnowledgeBase`、`searchKnowledgeBaseBm25`、`fusePaperRankingsRrf` 生成排名；$k=5$ 时 dense Recall/MRR/nDCG = 1/0.638889/0.726892，BM25 = 0.666667/0.666667/0.666667，hybrid = 1/1/1。
- exact-title、compact-alias、CJK 三类 hybrid 严格提升；semantic-rewrite、title-abstract、hard-negative Recall 不低于 dense。
- 合成规模 200 papers × 8 chunks：BM25 严格两遍、扫描 3,200 chunks，paper candidate 峰值 7、每篇 hit 峰值 2，均等于请求上限。
- Plugin 定向 UI/调用回归通过；RRF 与 BM25 不再显示为 cosine similarity，Find Similar 显式传 title lexical query。
- `NODE_OPTIONS=--max-old-space-size=8192 npm test --workspace @arxiv-daily/core`：104 个文件、1,830 项测试通过。
- `npm test --workspace obsidian-arxiv-daily`：37 个文件、591 项测试通过。
- `npm run typecheck`、`npm run check:boundaries`、plugin production build、`git diff --check` 全部通过。
- 多轮独立代码审查后无高/中严重问题。

## Abort / reshape triggers

- 如果实现需要新增 KB schema/index 文件或数据库才能正确工作，停止并留给 P4。
- 如果 BM25 构造全库 tokenized corpus、全文拼接副本或全 vocabulary inverted index，停止并改为 query-term-only 两遍 scan。
- 如果 BM25 与 cosine 必须直接做 max/加权和，停止并使用 rank-based RRF。
- 如果一篇论文因多个 chunk 在 RRF 中重复投票，停止并先做论文聚合。
- 如果 hybrid 的 semantic rewrite 或 title+abstract 类 Recall@10 比 dense 下降超过 0.05，停止总体调权并修复候选/长查询策略。
- 如果 exact title、Pan-STARRS alias、一次 query embedding、empty/model mismatch/cancel/corrupt guard 回归，停止生产接线。
- 如果新增 lexical 工作集在合成规模测试中无界，或必须触碰 plugin UI、Dashboard、sidecar，停止并重划阶段。

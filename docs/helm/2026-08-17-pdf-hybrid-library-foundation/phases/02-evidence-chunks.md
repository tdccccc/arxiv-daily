# P2 — 结构感知 EvidenceChunk 与版本化迁移

goal_ref: ../goal.md
updated: 2026-08-18

## Outcome

全文索引可以从结构化 `ParsedDocument` 生成带稳定 ID、章节上下文、页范围和可选精确定位的 `EvidenceChunk`；现有 v1 知识库可兼容读取和渐进升级，当前 PDF.js 路径的 chunk 文本、embedding 输入与排序保持不变。

## Assumptions

- 当前 PDF.js 只提供 page blocks，继续走 legacy page chunking；只有声明 `document-structure` 的 parser 才启用章节感知分块。
- v1 文档缺少结束页和字符映射，提升到 v2 时只保存可证明的起始页，不伪造 `pageEnd`、字符范围或 bbox。
- `index` 暂时保留为向量行序；`page` 暂时保留为 `locator.pageStart` 的兼容字段。
- 旧库不因 schema 升级自动全量重嵌入；文件变化、显式重建或未来高质量 parser 产出时渐进写入 v2。
- P2 不消费新 locator 的 UI，不改变现有 dense/title/token 排序和方向聚类算法。

## Approach

先定义 parser/chunker/embedding-input provenance、`EvidenceChunk`、locator 和 host-neutral 稳定 ID。新增结构化 chunker，对 page-only PDF.js 精确委托旧 `chunkFullText`，对有语义结构的 block 维护 heading stack 和 section 边界。随后升级知识库 schema 为双版本 decoder，接通 parser/extractor 双入口，并把证据 metadata 投影到 retrieval hit；旧调用方、旧文档与现有 UI 继续兼容。

## Test strategy

- change kind: additive behavior plus versioned persistence migration
- strategy: strict Red–Green–Refactor per behavioral chunk; v1 fixtures provide the compatibility baseline
- Red / baseline signal: evidence contract、structured chunker、v2 decoder、parser indexing input 和 retrieval metadata 的 focused tests 先因 API/schema 不存在失败；每块生产修改前现有 legacy tests 保持 Green
- Green / regression checks: 每块运行新增 focused tests 和相邻 legacy regressions；阶段末运行完整 core/plugin tests、全仓 typecheck、boundary check 和 plugin build

## Tasks

- [x] 定义并验收 parser provenance、`EvidenceChunk`、source locator、可选 normalized bbox、derivation versions 与稳定 host-neutral chunk ID。
- [x] 实现并验收结构感知 chunker；heading 作为上下文和 section 边界，PDF.js page-only 分支与旧 chunk/embedding 文本逐字段等价。
- [x] 实现并验收 manifest/paper schema v2、v1 双版本读取、legacy provenance/ID 提升、未知未来版本拒绝与混合版本渐进写入。
- [x] 索引编排接通 `ParsedDocument`，记录 derivation；当前 PDF.js、arXiv、unresolved rebind 和标题刷新保持 chunk/vector 兼容。
- [x] retrieval hit 投影 chunk ID、heading 和 locator，证明检索排序、聚类输入与现有插件搜索 UI 不变，并完成阶段回归。

## Verification

- Evidence/ParsedDocument：2 个文件、6 项测试通过。
- Structured/legacy chunking：2 个文件、19 项测试通过。
- Knowledge-base schema/store：2 个文件、51 项测试通过。
- Orchestration/retrieval/clustering/proposer：4 个文件、最终 78 项测试通过；包含 future-schema 拒写后不删 paper、不降级 manifest 的端到端回归。
- Plugin parser/extractor/search block：3 个文件、14 项测试通过。
- `NODE_OPTIONS=--max-old-space-size=8192 npm test --workspace @arxiv-daily/core`：101 个文件、1,817 项测试通过。
- `npm test --workspace obsidian-arxiv-daily`：37 个文件、591 项测试通过。
- `npm run typecheck`：core、node-runtime、CLI 和 plugin 全部通过。
- `npm run check:boundaries`：通过。
- `npm run build --workspace obsidian-arxiv-daily`：production build 通过。
- `git diff --check`：通过；三轮独立代码审查后无高/中严重问题。

## Abort / reshape triggers

- 如果 v1 文档必须重读 PDF 或重嵌入才能继续搜索，停止并 reshape 迁移模型。
- 如果 page-only parser 改变旧 chunk 的 `index/page/text`、embedding 输入、向量或排序，停止并保留 legacy 分支。
- 如果稳定 ID 依赖 paperKey、文件路径、随机数、Node crypto 或运行时对象，停止并重做 canonical identity。
- 如果 parser/chunker provenance 无法在索引前用于复用判断，停止并重做端口 identity，而不是静默复用不兼容向量。
- 如果 schema 版本升级会把未知未来版本误报为普通损坏并自动覆盖，停止写入并增加 incompatible 分支。
- 如果并行 active Helm 在 `plugin/main.ts` 或 Dashboard 出现未提交重叠修改，先隔离 core 工作，不写重叠文件。

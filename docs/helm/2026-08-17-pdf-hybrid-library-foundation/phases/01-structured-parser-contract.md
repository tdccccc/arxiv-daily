# P1 — 结构化解析契约与 PDF.js 兼容适配

goal_ref: ../goal.md
updated: 2026-08-17

## Outcome

Core 拥有稳定、host-neutral、可序列化的结构化文档解析契约，Obsidian PDF.js 能产生该契约并无损投影到现有 `PdfTextExtractor` 结果；现有索引、chunk、hash、向量和检索行为保持不变。

## Assumptions

- 扁平、按阅读顺序排列的 block 加明确 locator 足以支撑下一阶段；递归章节树可等真实解析器输出稳定后再扩展。
- PDF.js 本阶段只能可靠声明逐页文本、行级 typography 和文档标题，不声明语义章节、表格或公式能力。
- 旧 `PdfTextExtractor` 契约需要继续支持现有索引编排与测试 fake，不能继承或改成新返回类型。
- `packages/core/src/index.ts` 是与并行目标的已知共享点；本阶段只追加导出，不重排现有导出。

## Approach

新增通用 `ParsedDocument`、`ParsedBlock`、source locator、parser capability 和 `DocumentParser` 端口；在 full-text 边界增加到旧 `PdfExtractionResult` 的无损投影。Obsidian PDF.js 实现新 parser，旧 extractor 通过 composition 调用 parser 和投影器。索引编排继续只看旧端口，本阶段不改变任何持久化或检索逻辑。

## Test strategy

- change kind: behavior-preserving refactor with additive contract
- strategy: strict Red–Green–Refactor for each additive contract/adapter chunk, followed by compatibility regression
- Red / baseline signal: 新的 core 契约、兼容投影和 Obsidian parser 测试分别先因模块或导出不存在而失败；现有 PDF extractor 定向测试在生产改动前保持 Green
- Green / regression checks: core 新增测试和全文索引/chunk 回归；plugin parser/extractor 测试；core/plugin typecheck；boundary check；最后运行 core 与 plugin 相关完整测试

## Tasks

- [x] 新增并验收 host-neutral `ParsedDocument`、block、locator、capability 与 `DocumentParser` 契约及根导出。
- [x] 新增并验收 `ParsedDocument` 到旧 `PdfExtractionResult` 的无损兼容投影，固定空页、layout 能力和 metadata 语义。
- [ ] 新增并验收 `ObsidianPdfDocumentParser`，让旧 `ObsidianPdfTextExtractor` 通过 parser + 投影继续提供逐字段等价结果。
- [ ] 以索引兼容回归证明现有 text hash、chunk、embedding 输入、标题和检索产物不受影响，并完成阶段检查。

## Verification

- `npm test --workspace @arxiv-daily/core -- tests/parsed-document-contract.test.ts tests/pdf-text-compat.test.ts`
- `npm test --workspace obsidian-arxiv-daily -- tests/pdf-document-parser.test.ts tests/pdf-text-extractor.test.ts`
- `npm test --workspace @arxiv-daily/core -- tests/fulltext-index-orchestration.test.ts tests/chunking.test.ts tests/fulltext-retrieval.test.ts`
- `NODE_OPTIONS=--max-old-space-size=8192 npm test --workspace @arxiv-daily/core`
- `npm test --workspace obsidian-arxiv-daily`
- `npm run typecheck`
- `npm run check:boundaries`

## Abort / reshape triggers

- 如果结构化契约需要宿主对象、PDF.js proxy、sidecar transport 或非可序列化字段才能表达，停止并 reshape 契约边界。
- 如果旧结果投影改变页面文本、layout 对齐、metadata title、hash、chunk 或 embedding 输入，停止迁移并保留旧 extractor 实现，不以更新快照掩盖变化。
- 如果并行 active Helm 在相同 parser/full-text 文件出现未提交修改，停止写入重叠文件并先协调所有权。
- 如果 PDF.js 无法在不复制两套提取逻辑的情况下同时实现新旧契约，优先保留旧行为，重新评估 adapter 方向。

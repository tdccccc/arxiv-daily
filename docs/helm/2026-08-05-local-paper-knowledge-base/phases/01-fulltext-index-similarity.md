# P1 — 全文索引与相似检索（fulltext-index-similarity）

<!-- Filename 01-fulltext-index-similarity.md ↔ P1 -->
goal_ref: ../goal.md
updated: 2026-08-05

## Outcome

plugin host 上对个人文献库每篇已识别论文可完成：本地提取 PDF 全文（复用 Obsidian 内置 pdf.js）→ 分块 → multilingual-e5-small（q8）本地 CPU 嵌入 → 写入旁路知识库（按 scope/identification fingerprint 分片、CAS 版本化、按文件哈希增量复用、可删除重建）。给定一篇论文或一段描述，可离线检索库内 top-k 相似论文，带相似度与命中 chunk 证据。CLI host 的提取/嵌入不在本阶段。

## Assumptions

- Obsidian 桌面插件上下文可访问内置 pdf.js 且能按页提取全文；两栏/复杂排版的质量降级可接受（正文可读、页序正确）。
- transformers.js（onnxruntime-web）在 Electron renderer CPU 上可加载 multilingual-e5-small q8 并批量推理；首次下载需网络（支持 HF 镜像），之后离线可用。
- 全文提取与嵌入只在 plugin host 运行；Core 只定义端口与编排（host-neutral 约束）。
- 个人库规模（≤ 数千 chunk）下全库暴力余弦检索实时可接受；检索预计算、运行时只查 top-k。
- e5 系列需要 query/passage 前缀区分，入库与查询使用一致前缀策略。
- PDF 全文本地处理独立于模型授权（goal 已定）；全文内容不进入任何 LLM 输入。

## Approach

Core 新增 `packages/core/src/library/fulltext/`：定义 `PdfTextExtractor` / `EmbeddingModel` 端口（host 实现）、分块纯函数、KB 旁路 store（manifest 学 profile store 的 expectedRevision CAS，路径按 scope/identification fingerprint 分片）、暴力余弦 top-k 检索编排。plugin host 接线：PDF 字节经现有 `ScopedLibrarySource.readBinary` 读取，提取→分块→嵌入→增量更新由插件命令触发；重建 = 删除知识库目录重跑。检索入口形态（Dashboard 按钮 vs 检索栏）保持 open question，本阶段只交付引擎与最小触发入口。

## Tasks

- [x] Core 端口与 KB 文档类型：`PdfTextExtractor` / `EmbeddingModel` 端口、KB 记录（modelId、dimension、chunk 文本+page+向量、文本哈希）、store 接口与路径推导（scope/id fingerprint 分片）、严格 decoder。
- [x] 全文提取验证（plugin host）：接入 Obsidian 内置 pdf.js，对 3-5 篇真实 arXiv PDF 验证全文质量（页序、正文覆盖、垃圾行），记录验证结果；失败即触发 reshape。
- [x] 分块（core 纯函数）：按段落/标题切分，~512 token、带重叠、保留 page 号、噪声过滤；单测。
- [x] 本地嵌入（plugin host）：transformers.js 加载 multilingual-e5-small q8，CPU 批量推理，模型按需下载（HF 镜像）+ vault 外缓存；core 编排调用。
- [x] KB store：manifest（primary/backup、CAS expectedRevision、严格 decoder、语义 revision）+ 每论文 chunk/向量文件（内容寻址、幂等重写）；单测覆盖 stale/corrupt/重建。
- [x] 暴力余弦检索：查询向量 vs 存储 chunk 向量 → 论文级 top-k（相似度 + 命中 chunk 证据）；单测（合成向量）。
- [x] 端到端增量索引：scan→catalog→KB 增量更新（observationFingerprint/文本哈希复用）、重建验证、插件命令触发；用真实论文验证可解释检索结果。

## Verification

- `npm run typecheck && npm test && npm run check:boundaries` 通过。
- 分块/检索/store 单测通过（CAS stale 冲突、corrupt 恢复、同哈希跳过重嵌入）。
- 真实样例：3-5 篇 arXiv PDF 索引后，用其中一篇标题/摘要查询，top-1 为该论文自身、同主题论文靠前，命中段落与主题相符。
- 增量：修改一篇 PDF 再索引，仅该论文重新提取/嵌入，其余复用（可观察）。
- 重建：删除知识库目录重跑，检索结果一致。
- 离线：首次下载模型后断网仍可完成检索。

## Abort / reshape triggers

- Obsidian 内置 pdf.js 在插件上下文不可用或全文质量不可用（乱序/乱码/截断）→ L2：评估打包 pdfjs-dist 替代。
- transformers.js 在 Electron CPU 不可用或过慢（单 chunk > 500ms）→ L2：换更小模型或评估 Node 侧推理路径。
- 存储体积/写入性能失控（全库 > 200MB 或单论文 > 10MB）→ L1/L2：调整分块参数或向量存储格式（Float16/Binary）。
- 暴力检索在目标规模不达标（> 500ms/查询）→ L1：两级检索（论文级向量先粗筛再 chunk 精排）。

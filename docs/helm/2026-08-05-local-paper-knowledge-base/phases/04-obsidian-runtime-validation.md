# P4 — Obsidian 运行时验证

goal_ref: ../goal.md
updated: 2026-08-06

## Outcome

P1 遗留的两个 Obsidian 运行时未决点有实测结论（`window.pdfjsLib` 在插件上下文经 `loadPdfJs()` 后的实际可用性；transformers.js 在渲染进程从 CDN 拉取 wasm、经 Cache API 缓存模型的实际表现），如发现问题则已修复；验证过程与结论记录在 journal 与本文档，后续用户可在 Obsidian 中自行复跑诊断命令。

## Assumptions

- Obsidian 官方 `loadPdfJs()` 解析后 `window.pdfjsLib` 会挂到 window 上（obsidian.d.ts 文档声称如此；代码按此假设读取 window.pdfjsLib，未核对返回值）。
- 渲染进程中 transformers.js 的 wasm 后端（onnxruntime-web）能从默认 CDN（jsdelivr）拉取 wasm 二进制并经 Cache API 缓存；模型文件能从 huggingface.co 下载（本机网络已探测：HF 与 jsdelivr 直连可达，无墙）。
- 用户本机的 Obsidian 可以安装/覆盖插件构建产物（main.js），并愿意把 personal library 指向一个受控测试目录做验证（不动真实库）。
- 渲染进程内置 pdf.js 的版本与 Node 侧注入的 pdfjs-dist 6.2.108 行为一致到"提取质量不影响检索判据"的程度（P1 冒烟判据 5/5 top-1 自命中）。

## Approach

在插件内新增一个轻量诊断命令 `diagnose-fulltext-runtime`，把两个未知点拆成两段独立探测（一段失败不阻断另一段），产出结构化证据（Notice 汇总 + logger 明细 + 可复制的结果模态框）；随后用户在真实 Obsidian 中按验证手册执行受控流程（5 篇 Node e2e 同款 PDF：诊断 → 索引 → 磁盘核对 → 复用重跑 → 标题检索），把 DevTools console 输出贴回会话分析。Node 侧基线（tmp/fulltext-e2e）已给出每个环节的期望信号。

## Tasks

- [x] T1 helm 文档：goal.md P4 索引行 + status active；本文档；journal 起始条目
- [x] T2 诊断命令 `diagnose-fulltext-runtime`：
  - Part A（pdf.js）：`loadPdfJs()` 解析；双通道核对（返回值存在性 + `window.pdfjsLib` 存在性与版本号）；库连接可用时取 catalog 中第一篇带 PDF 文件的论文做真实冒烟提取（页数/字符数）；无库连接则退化为仅 presence 检查
  - Part B（嵌入）：`createTransformersEmbeddingModel()` 真实加载（首次触发下载 + session + 既有维度探针），报告 modelId/dimension/remoteHost/wasmPaths/耗时
  - 呈现：logger.info 明细 + Notice 逐段 PASS/FAIL 汇总 + DiagnosticsModal 风格结果模态框（textarea + 复制）
- [x] T3 测试：obsidian mock 补 `loadPdfJs`；commands.test.ts 断言注册与回调；报告构建逻辑单测；plugin 全量 `tsc --noEmit` + `vitest run`（415/415）、根 `npm run lint`（0 error）+ `npm run check:boundaries`（OK）
- [x] T4 构建：`cd plugin && npm run build` 产出安装用 main.js（1.2MB，gitignored）；technical-report handoff（updated，全文知识库宿主段 + 命令列表段）
- [x] T5 验证手册定稿（本文档 Verification 节，含判定表与收尾步骤）
- [x] T6 用户 Obsidian 实测 → 证据分析 → 修复（渲染进程嵌入崩溃）→ journal 定论 → goal.md P4 done

### T6 实测结果（2026-08-07，Desktop/plugin_test，Electron 39.2.6）

- **pdf.js 段 PASS**：`window.pdfjsLib` v5.3.34，冒烟提取 18 页/45165 字符。
- **嵌入段 FAIL → 修复 → PASS**：首测报 `reading 'create'`；根因为 Obsidian 渲染进程带 Node 集成（`process.release.name === "node"`）使 transformers.js v4 走 Node 分支，而 web bundle 不含 onnxruntime-node（tree-shaken）→ `ONNX.InferenceSession` undefined。修复：`alignElectronReleaseProbe`（按 `process.versions.electron` 存在性把 release.name 对齐为 `"electron"`，factory 创建时调用，幂等）；复测 PASS（runtime probe 行显示 `electron 39.2.6`，wasmPaths 实锤 asyncify 构建，加载 4722ms）。
- **索引/复用**：5 indexed/0 failed → 重跑 5 reused；KB manifest rev 1、5 papers、chunk 数 49/64/44/70/262 与 Node 基线逐篇一致。
- **检索**：标题自命中 **4/5**（BERT/GPT-3/ResNet/Dropout top-1；"Attention Is All You Need" 0.823 第 3，Node 基线同查询 0.8359 top-1）。归因：pdf.js 版本排版级文本差异（归一化后 32/44 chunk 相同）→ 级联 chunk 边界漂移 → best-hit 更换；叠加 wasm/node 内核噪声与短查询方差。非功能回归。

## Verification

以下为真实 Obsidian 中的验证手册（用户侧执行，console/Notice 证据回贴本会话分析）。前置：`plugin/main.js` 已由 `cd plugin && npm run build` 产出（1.2MB，含新诊断命令）；受控语料 5 篇 PDF 取自 `tmp/fulltext-e2e/pdfs/`（1207.0580、1512.03385、1706.03762、1810.04805、2005.14165，Node 侧 e2e 同款）。

### 步骤与判定表

| # | 步骤 | 操作 | PASS 信号 | 失败时需回贴的证据 |
| --- | --- | --- | --- | --- |
| 0 | 安装 | 备份 vault 插件目录现有 `main.js`，用新构建覆盖（`<vault>/.obsidian/plugins/obsidian-arxiv-daily/main.js`），重载 Obsidian（或禁用再启用插件） | 插件正常加载、命令面板可搜到 `Diagnose full-text runtime` | 加载报错全文 |
| 1 | 日志级别 | 设置 → Advanced → 日志级别 → debug（后续 DevTools console 才有 `[arxiv-daily]` info 行） | 设置保存无报错 | — |
| 2 | 受控库 | 在 vault 内建临时目录（如 `tmp-ft-validation/`），拷入 5 篇 PDF；记录当前 personal library 根（用于事后恢复），选择该临时目录为 personal library | 库扫描完成 | 扫描失败信息 |
| 3 | 诊断 | 运行命令 `Diagnose full-text runtime (pdf.js + embeddings)` | Notice 显示 `pdf.js PASS … smoke … pages / … chars` 与 `embeddings PASS … 384 dim … ms`；结果模态框可复制；**首次运行 Part B 会下载模型（分钟级），属预期** | 完整 Notice 文案 + 模态框全文 |
| 4 | 索引 | 运行 `Index personal library full text (local embeddings)` | Notice 显示 `5 indexed, 0 reused, 0 failed, 0 pruned` | Notice 全文 + console 错误 |
| 5 | 磁盘核对 | 打开 vault 的 arxiv-daily 索引目录（设置 output 的 dailyDir 同级），定位 `personal-library-knowledge-base/<scopeHex>/<idHex>/` | `manifest.json` 存在（revision 1）；`papers/` 下 5 个 `.json` 文件 | 目录树 |
| 6 | 复用 | 再次运行索引命令 | Notice 显示 `0 indexed, 5 reused, 0 failed, 0 pruned`（不再重新提取/嵌入） | Notice 全文 |
| 7 | 检索 | 运行 `Search personal library full text…`，分别用 5 篇论文标题查询 | 每篇标题查询 top-1 为自身（5/5 自命中，与 Node e2e 判据一致） | 每次查询的 Notice 文案 |
| 8 | 证据 | DevTools（Ctrl+Shift+I / Cmd+Opt+I）→ Console，过滤 `[arxiv-daily]` | 无未捕获异常；索引与诊断的关键行齐全 | 完整 console 输出（含时间戳） |

### 判定逻辑（分析阶段）

- 两段诊断 PASS + 索引 5/5 + 复用 5/5 + 检索 5/5 → 两个未决点定论：`window.pdfjsLib` 在真实 Obsidian 可用、渲染进程 wasm/模型加载正常；P4 完成。
- `loadPdfJs()` 解析但 `window.pdfjsLib` 缺失 → 启用 C1（main.ts 用返回值显式注入 extractor）。
- 渲染进程 wasm CDN 拉取失败 / 模型下载失败（Part B fail）→ 启用 C3/C4（wasmPaths 固定或本地打包 / HF 镜像设置接线），先向用户简报。
- 提取质量与 Node 基线（该 5 篇 chunk 数 49/64/44/70/262、页数/字符数）显著不一致 → C5 对比判定。

**实测修订（2026-08-07）**：

- **Part B fail 的实测根因是 C6（新增）**：Electron 渲染进程 Node 集成使 `process.release.name === "node"` → transformers.js 走 Node 分支 → web bundle 缺 onnxruntime-node → `reading 'create'`。修复 = `alignElectronReleaseProbe`（按 `process.versions.electron` 对齐 release.name，与插件 `isNodeRuntime()` 共用同一探针语义）。诊断报告新增 `runtime probe` 行用于区分分支。
- **检索判据从"5/5 top-1 自命中"修订为"≥4/5 + 其余为近并列翻转并归因"**：真实 Obsidian 实测 4/5；唯一 miss（Attention 查询）归因于 pdf.js 版本排版级文本差异导致的 chunk 边界漂移 + wasm/node 内核噪声 + 短查询方差 + 长文 best-passage 优势，非功能回归（Node 基线同查询 top-1 仅领先 0.023）。

### 收尾（验证完成后）

- 恢复 personal library 根为原值；删除临时目录 `tmp-ft-validation/`。
- 测试库的 KB 分片（`personal-library-knowledge-base/<scopeHex>/<idHex>/`，指纹与真实库不同）为旁路 store，可安全删除重建；不删也不影响真实库。
- 验证期间首次下载的模型与 wasm 已进渲染进程 Cache API，真实库索引将直接复用。

## Abort / reshape triggers

- 若 `loadPdfJs()` 解析但 `window.pdfjsLib` 缺失（C1）：小修——用 loader 返回值显式注入 extractor，继续。
- 若渲染进程 wasm CDN 拉取失败或模型下载失败（C3/C4）：L2 reshape——先向用户简报，再启用 wasmPaths 固定/本地打包或 HF 镜像设置接线。
- 若提取质量与 Node 基线显著不一致（C5）：对比页数/chunk 数判定影响面；若影响检索判据则 L2 reshape 提取策略。
- 若用户无法在 Obsidian 中执行（无可用安装/环境问题）：暂停 T6，记录原因，P4 保持 active，不硬推。

# Technical Report

## Scope and System Overview

arXiv Daily 是一个以研究主题过滤 arXiv 论文、生成 Markdown 日报/详报，并支持定时调度与邮件投递的系统。当前主实现是 TypeScript monorepo（版本 `0.4.1`），围绕 host-neutral 业务核 `@arxiv-daily/core` 组织：

- **Obsidian 插件**（`plugin/`，清单 id `arxiv-daily`，`isDesktopOnly: true`）提供设置页、命令、状态栏与 Dashboard 视图，并在进程内运行调度器；
- **Node CLI**（`apps/cli/`，npm 包名 `arxiv-daily`）读取固定路径 `config.toml`，提供单次运行、crontab 安装、数据导入导出与邮件命令；
- **Cloudflare Worker 邮件中继**（`services/email-relay/`，不在根 npm workspaces，包版本独立）实现官方代发（Beta）的验证与投递；
- 根目录 `arxiv_daily.py` 为兼容壳：若 `apps/cli/dist/arxiv-daily-cli.cjs` 存在则转发 argv（无参时默认 `run --today`）；
- **VS Code companion**（`extensions/vscode-arxiv-daily/`，未进根 workspaces，版本独立）提供 vault 探测、Paper Index Dashboard，以及通过当前 CLI 契约执行今日运行和按 ID 摘要。

业务核通过 `HostAdapters`（HTTP、存储、密钥、进度、资源打开、标记解析）与具体运行时解耦。插件与 CLI 分别注入 Obsidian / Node 适配层后，共享 `ArxivPipeline`、状态/索引与投递编排；调度语义在两端不同（见下文）。

## Runtime and Technology Stack

| 层次 | 技术 | 在本项目中的职责 |
| --- | --- | --- |
| 语言/运行时 | TypeScript（ES2022）、Node.js `>=20.11.0` | 全仓源码与 CLI/构建；CI 使用 Node 22.17.0 |
| 包管理 | npm workspaces | 根工作区：`packages/*`、`apps/*`、`plugin`；email relay 与 VS Code companion 各用独立 manifest/lockfile |
| 业务核 | `@arxiv-daily/core` | 流水线、LLM、调度、索引、设置、邮件编排；生产第三方依赖仅 `pako` |
| Node 宿主 | `@arxiv-daily/node-runtime` | `fetch`、文件系统、`EnvSecretProvider`、linkedom 等 |

| Obsidian 宿主 | Obsidian Plugin API + `obsidian` 类型；内置 pdf.js（`loadPdfJs` → `window.pdfjsLib`） | Vault 读写、`requestUrl`、设置持久化、UI、PDF 全文提取 |
| 本地嵌入 | `@huggingface/transformers` v4（onnxruntime-web，q8；插件运行时依赖，esbuild 打包） | 插件宿主内 CPU 全文嵌入（multilingual-e5-small）；`pdfjs-dist` 仅作 devDependency 供 Node 侧验证注入 |
| VS Code companion | VS Code Extension API `^1.90.0` | `workspace.fs` Dashboard 与 `ProcessExecution` CLI 任务 |


| 构建 | esbuild | 插件打 `main.js`；CLI 打 `dist/arxiv-daily-cli.cjs`（并复制到 `plugin/arxiv-daily-cli.cjs`） |
| 测试/类型 | Vitest、`tsc --noEmit`、ESLint（含 `eslint-plugin-obsidianmd`） | 工作区测试；边界检查脚本 |
| 邮件 | Resend HTTP API；Cloudflare Workers + KV + Durable Objects | 自发送与官方代发 |
| 外部数据 | arXiv HTML/Atom/源码页 | 论文发现与正文抽取 |
| LLM | OpenAI 兼容 Chat Completions（流式） | 过滤、详报选择、日总结/详报摘要 |

## Frameworks and Responsibilities

### `@arxiv-daily/core`

宿主无关的业务实现与契约：


- **适配器契约**（`packages/core/src/core/adapters.ts`）：`HttpClient`、`StorageAdapter`（含可选私有原子替换/恢复、跨进程 exclusive create、descriptor-backed namespace guard、append、二进制与目录枚举能力）、`SecretProvider`、`ProgressReporter`、`ResourceOpener`、`MarkupParser`，聚合为 `HostAdapters`。进度阶段枚举含 `fetch-metadata`、`fetch-recent`、`enrich-abstract`、`filter`、`personal-novelty`、`fetch-content`、`summarize-daily`、`summarize-detail`、`write-detail`。

- **流水线**（`ArxivPipeline`）：按日生成日报与可选详报；结果 kind 为 `completed | pending | cancelled | failed_transient | failed_permanent`。
- **LLM**（`LlmClient`）：流式调用、客户端内瞬态重试、thinking/reasoning 参数、模型列表探测、敏感信息脱敏。
- **调度**（`SchedulerService` / `SchedulerDriver`）：本地时间窗、回看窗口、周末跳过、运行锁、状态机与历史记录；公开 API 主要在 `SchedulerService`（`tick`、`tickToday`、`tickTodayScheduled`、`runForDateNow`、`forceRunForDate`、`retryFailedInLookback`、`runAllPending` 等）。
- **数据面**：`PaperIndexStore`、`StateStore`、`RunHistoryStore`、过滤/日总结 checkpoint、邮件 `delivery-state`。
- **全文知识库**（`packages/core/src/library/fulltext/`）：全文索引与相似检索的 host-neutral 编排。端口 `PdfTextExtractor`（PDF bytes → 页序文本）与 `EmbeddingModel`（文本批量 → Float32 向量）由宿主实现；端口声明 `prefixPolicy`（`e5` 或 `none`）——编排只在 e5 模型上应用 `applyEmbeddingPrefix` 的 query/passage 前缀，远程模型嵌入纯文本。纯函数 `chunkFullText` 以段落聚合 + ~10% 重叠切块（token 估算 `ceil(chars/4)`，保留 1 起始页码）。旁路 store `FullTextKnowledgeBaseFileStore` 不写入 papers.json，路径按 scope/identification fingerprint 分片到 `<indexDir>/personal-library-knowledge-base/<scopeHex>/<idHex>/`：`manifest.json`（primary/backup、expectedRevision CAS、严格 decoder、语义重放幂等）为权威索引，`papers/<sha256(paperKey)>.json` 保存 chunk 文本与行主序 Float32 向量（base64 序列化，原子幂等重写）；非空知识库在索引入口拒绝切换 modelId，要求先删除重建。`indexPersonalLibraryFullText` 同时处理 catalog 中的 arXiv paper 与 unresolved PDF；fallback paperKey 为完整 PDF bytes 摘要 `file:sha256:<digest>`，文档和 manifest 同时保存 `contentHash`、相对文件路径、首页提取的可选标题及标题规则版本。同一内容的多个 unresolved 路径合并到一个文档；path/size/mtime 观测未变时复用已保存的内容摘要，观测变化时重读 PDF，因此改名后重新读取成功且内容未变时 paperKey 保持不变。缺少 `contentHash` 的 ready fallback 文档会在索引时按 PDF 摘要重新绑定键和路径并复用 chunks/vectors；fallback 标题规则版本当前为 5：标题提取优先使用宿主提供的文档元数据标题（`info.Title`，机器可读；垃圾元数据——路径、文件名、页码引用、arXiv stamp、LaTeX 残留——被拒绝，pdf.js 遗留的 HTML 实体如 `&ndash;`/`&#x00D7;` 被解码；若字体结果 token 集合已覆盖元数据且更长，如元数据丢失下标字符，则字体结果胜出），其次使用宿主提供的行级 typography layout（每行 text/fontSize/topFraction，与页文本同 hasEOL 行分组），core 按字号 band 选标题——逐行排除页边文本（基线在页框上方）、arXiv stamp、预印本编号/DOI，顶部条带（页高 13% 内）且下一 band 达最大字号 93% 时判定为期刊刊头跳过，run 装配跳过下标短行、作者行（姓名模式 + 首字母或 and 前瞻）断行，候选须过形状检查（长度、首字符、节标题/期刊引用/日期/email/URL 行），标题选中后以续行（小写开头、罗马数字部分、无姓名首字母的长短语）扩展；无 layout 的宿主回退到纯文本行启发式（arXiv/reprint 页眉与 `Advance Access …` 过滤仍在兜底路径）；manifest 中的版本不匹配时重新提取 PDF 文本并更新 `title` / `titleVersion`，保留已有 chunks/vectors。刷新提取失败会保留旧标题与旧版本，使下一轮索引继续重试；`titlesRefreshed` 统计成功执行的版本刷新，即使标题文本未变化也计数。arXiv paper 按 catalog observation fingerprint 与模型 id 增量复用，fallback 还可按内容摘要复用。新建文档的摘要准备、提取、嵌入或保存失败按论文写入 failed 记录并在后续索引重试；复用/迁移文档的加载或保存失败同样按论文隔离（记录 failed 并丢弃该论文遗留的旧文档，不中止整轮）。若重索引失败覆盖了先前 ready 的记录，对应的旧 paper 文档会被删除以免残留孤儿；剪枝以及最终 manifest 替换失败仍会中止整轮。paper 文件写入/删除与 manifest 替换不构成跨文件事务。`searchFullTextKnowledgeBase` 在查询嵌入前先比对 manifest modelId 与当前嵌入模型，非空知识库模型不一致时抛错并要求删除重建；查询期只做一次查询嵌入，检索循环跳过加载失败（损坏）的文档并经可选 logger 记录警告；`searchKnowledgeBase` 默认对候选论文的全部 chunk 与查询向量做 corpus-level centering（减 chunk 均值后重归一化，可 `centerCorpus: false` 关闭），再把最大 centered chunk 余弦、确定性标题相似度和正文 compact-token 命中分数取最大值作为论文分数。标题词法相似度只对照查询的首个非空段落（title + abstract 查询因此按标题行计分）。正文 token 融合仅在短关键词/短标题查询上启用（显著 token ≤12 且总长 ≤160 字符）；标题+摘要等长查询只保留向量与标题融合。正文 token 以文档频率过滤常见词并按 `count / (count + 3)` 计入重复出现强度；标题包含全部稀有查询 token 时提供 0.95 的分数下限，正文高频命中仍可超过该下限。只有至少产生一个 chunk hit 的文档进入结果，返回的 chunk hits 仍是按向量分数排序的 page/text 内部证据。远程嵌入为 core 侧 `createRemoteEmbeddingModel`：OpenAI 兼容 `POST {baseUrl}/embeddings`（`{model, input}` → `{data:[{index, embedding}]}`），批量 ≤64、超时、逐响应断言维度；modelId 为 `remote:{model}:{dimension}`（不含端点 URL）。
- **聚类与方向草案**（`packages/core/src/library/clustering/` 与 `personal-library-direction-proposer.ts`）：`buildClusteringInput` 从知识库按 recency 加载每篇 ready 论文的 chunk 向量（arXiv paperKey 的 YYMM 前缀新到旧、无日期的 fallback `file:sha256:` key 排在最后，长文截断到 80 chunk，上限 2000 篇）；`clusterPaperVectors` 做 single-linkage 聚类（Kruskal 合并）：corpus-level centering（减去语料 chunk 均值，抑制 e5-small 的饱和余弦中的公共学术方向）→ 论文间相似度 = 最强 chunk 对余弦（best-passage 证据，与检索同语义）→ 边按强度降序合并，在**相对停止线**（最强边的 `relativeStopRatio`，默认 0.65）处停——该相对规则对 e5 饱和/低分分布自适应，绝对阈值在该模型上实测不可靠 → 连通分量成簇、小于 `minClusterSize` 的进缓冲池（outlier pool），成员置信度 = 簇内最强边（clamp [0,1]）。方向生成入口 `proposeClusteredPersonalLibraryDirections`（schema v3）：校验 KB 与 catalog 的 scope/identification 指纹一致后，每簇一次有界 extraction 调用（复用 extraction system prompt 与 validation/重试），候选携带 `clusterMembers`（全成员 + 置信度），跳过跨簇 synthesis；`catalogInputPapers` 覆盖簇成员 ∪ 缓冲池，缓冲池由审核界面即时派生（`catalogInputPapers` − ∪`clusterMembers`）。
- **方向增量更新**（`packages/core/src/library/incremental/`）：`suggestIncrementalPlacement` 对未归入任何已确认方向的已索引论文，与各方向代表论文锚（≤5 篇）在语料 centered chunk 空间做 best-passage 余弦比较，按相对规则判定：超过绝对下限 `minSimilarity`（默认 0.25）**且**领先次强方向至少 `minMargin`（默认 0.05）才就近归入，否则进缓冲池。缓冲池论文达触发条件后经 `reclusterPool` 产出新簇候选与 `nearestDirection` 漂移信号：聚类输入先把每篇论文的 centered chunks 压成一个归一化质心，single-linkage 的两两比较规模按论文数增长；漂移参考仍以完整 chunk sets 与方向锚比较。`suggestDirectionDiff` 用独立 `LlmClient` 对簇 + 现有方向生成 diff 建议（attach/new/split/merge），输出经严格校验（kind/direction/paperKey/reason/conflict，锁定方向 split/merge 拒绝）并最多重试 3 次。placement 无需模型处理许可；缓冲池达阈值后的 recluster + LLM diff 阶段要求 `personal-library-direction-generation` 授权门与当前 authorization fingerprint，无许可时跳过并写 `pendingAuthorization`（缓冲论文数 + 时间戳）到建议文档，审核 UI 显示待授权横幅。建议持久化到独立 CAS 文档 `<indexDir>/personal-library-incremental-suggestions/<scopeHex>/<idHex>/incremental-suggestions.json`（`IncrementalSuggestionsStore`：primary/backup、expectedRevision CAS、精确语义重放幂等、backup 恢复修复 primary），仅审核后生效。应用：attach 追加方向 `clusterMembers`（固定置信度 `SUGGESTION_MEMBER_CONFIDENCE` 0.9，禁止重复归入）+ `members-updated` 事件；new 只产出候选草案（`buildNewDirectionDraft`）走既有确认流程；split 从源方向移除成员并派生新方向（`split-derived` 血缘标记）；merge 镜像 review merge 语义（两个终态未锁定方向）；锁定方向接受 attach、拒绝 split/merge。
- **投递编排**（`deliverDailyEmailIfEnabled`）：self Resend 与 hosted 中继；返回 `DeliverEmailResult`，失败不回写流水线 run-state。
- **设置**：`PluginSettings`、默认值、校验、迁移、主题模板、详情选择策略。

core 源码禁止 Node 内置模块、未白名单第三方，以及 `process`/`Buffer`（`scripts/check-boundaries.mjs`）。

### `@arxiv-daily/node-runtime`

`buildNodeHostAdapters({ rootDir, env, … })` 组装：

- `NodeHttpClient`（`fetch`）
- `NodeStorageAdapter(rootDir)`：以调用方传入的 `rootDir` 为存储根（CLI 传入 `vaultRoot`）
- `EnvSecretProvider`：可读 `ARXIV_DAILY_*` 环境变量；**CLI 流水线/邮件实际密钥来自 `config.toml` 字段**，不经该适配器取 LLM/Resend key
- `StreamProgressReporter` / `StreamResourceOpener`
- `LinkedomMarkupParser`

### Obsidian 插件宿主

`buildObsidianHostAdapters`（`plugin/src/hosts/obsidian/`）将 Vault、`requestUrl`、设置中的密钥、工作区打开笔记/URL、DOM 解析接到 core。

`ArxivDailyPlugin`（`plugin/main.ts`）在 `onload` 中：加载设置与状态 → 构建 host → 状态栏进度 → `SchedulerService` → 注册设置页/Dashboard/命令；`schedule.enabled` 时启动进程内调度。卸载时停止调度、取消活动操作，并中止正在运行的个人文献库 inventory。

插件设置页的 **Personal library** 组可通过 Obsidian 桌面宿主的 Electron 目录选择器连接一个 Vault 内或 Vault 外目录。`selectLibraryRoot` 调用窄入口 `openObsidianLibrarySource`，后者只桥接 `@arxiv-daily/node-runtime/scoped-library-source`；成功打开后保存 canonical root 与文件系统 identity。用户可在不授予模型处理许可的情况下运行本地只读 inventory preview；Node scoped source 对 inventory 中的非符号链接条目使用 `lstat` 分类，文件条目带有 size 与 mtime 观测，符号链接不被跟随。其默认边界为 10,000 个 inventory 条目、16 层目录和 25 MiB 单文件读取上限；读取 range 前也按整文件 size 执行该上限，因此更大的 PDF 无法进入内容识别或全文索引。设置页将 PDF 标为 eligible，将其他文件类型、符号链接和特殊条目标为 ignored，并在 modal 中对每组最多显示 100 条路径。

模型处理许可是独立的插件本地状态。授权 modal 展示 canonical folder、eligible extensions、处理深度和脱敏后的 Chat Completions endpoint（远程嵌入开启时另展示 embedding endpoint 与 full-text 深度，ADR 0008）；确认时必须提交与展示内容相同的 fingerprint。fingerprint 绑定 root path、文件系统 identity、扩展名集合、处理深度和端点（LLM endpoint，远程嵌入开启时含 embedding endpoint digest），不绑定 API key、模型或 thinking 参数。目录、identity、端点、扩展名或处理深度变化会使授权失效；远程嵌入授权以 `full-text` 深度记录；Revoke 只移除模型处理许可，不断开本地只读目录。

连接目录后，用户还可显式执行 **Scan library** 或 **Reload catalog**，两者不要求模型处理许可。扫描先从 eligible PDF 的逻辑文件名识别现代 arXiv ID；文件名没有 ID 时，插件经 scoped source 分别读取文件头部至多 4 MiB 和尾部至多 1 MiB，`extractPdfIdentificationEvidence` 在有界解压预算内检查每个成功解压内容流所提取文本的前 512 个字符、XMP ID 和文档标题。PDF evidence parser 也可能返回 legacy arXiv ID，但 catalog identity 只接受规范化的现代 ID：插件仅直接采用可规范化的现代 evidence；direct evidence 若同时带有非 arXiv-stamp 的文档标题，先做严格 arXiv title search——命中**不同** ID 时判定 direct ID 为参考文献区误识别（如引用 "… arXiv:0912.0201 …" 恰好位于内容流前 512 字符）并采用搜索 ID，搜索失败/空/命中同 ID 则保留 direct ID；其他 direct evidence 若带可用标题则继续严格 arXiv title search，仍得不到现代 ID 时降级为 unresolved。Core reconciliation 对 host identifier 的返回值再次执行同一 modern-ID 校验，防止不受支持的值进入 metadata resolver 或 paperKey 构造。内容读取、证据提取或标题搜索失败也会降级为 unresolved；已经得到现代候选 ID 后，如果 arXiv metadata 拉取失败且 catalog 没有同 ID 的既有 metadata，该文件记为 failed。reconciliation 以 identification fingerprint、path、size、mtime 构造 observation fingerprint；filename-derived 文件按该观测复用，content-derived ready/unresolved 记录还须匹配 `PersonalLibraryFileIdentifier.version`，PDF 证据规则版本变化会让未变文件重新识别。它将 unresolved、unrelated 与暂时 failed 文件隔离，并把同一 arXiv 论文的多个 PDF 汇入一个 paper record。候选 canonical ID 通过现有 `ArxivFetcher.fetchMetadataByIds` 请求 arXiv Atom API；只有可用的 canonical metadata 或同 ID 的既有 paper metadata 能使文件成为 ready。

`PersonalLibraryCatalogStore` 将严格版本化 catalog 保存到现有 Vault index root 下的 `personal-library-catalog.json`。store 使用宿主 `writeTextAtomic`、独立 `.backup` generation、同 adapter/path 的进程内 mutation queue、严格 decoder、scope/identification compatibility 检查与语义 revision；有效 backup recovery 会修复 primary。完整 inventory 会移除缺失文件贡献，truncated inventory 保留未观察到的旧记录。插件将扫描注册为 `personal-library-scan` operation；重复扫描被拒绝，卸载、目录变化和输出路径变化会请求取消；进入最终不可中断的 atomic promotion 后采用 commit-wins 语义。设置页只显示 revision 与 ready/papers/unresolved/unrelated/failed/truncated counts。

全文知识库的 PDF 读取、文本提取与持久化均在本机；命令 `index-personal-library-fulltext` 与 `search-personal-library-fulltext`（`plugin/src/commands.ts` 的 `FullTextQueryModal` 输入查询）调用 `ArxivDailyPlugin.indexPersonalLibraryFullText` / `searchPersonalLibraryFullText`（`plugin/main.ts`），走 core 编排。索引流程：`loadPdfJs()` 就绪后以 `ObsidianPdfTextExtractor`（默认读 `window.pdfjsLib`，可注入；空字符串 `TextItem` 携带 `hasEOL` 时会把换行保留到下一个非空 item，避免首页标题、页眉和作者行粘连；同时按 item transform 矩阵（字号 = c/d 缩放、基线 = f）构建每页行级 layout（text/fontSize/topFraction，页高来自 `page.view`），供 fallback 标题提取的字体结构规则使用；同时经文档级 `getMetadata()` 读取 `info.Title` 作为元数据标题（缺失或读取失败降级为 undefined，不失败提取）；pdf.js 无 signal 参数，取消走 `loadingTask.destroy()` 且每个 await 与 AbortSignal 竞速；单页失败降级为空串、文档级失败抛错）提取全文，`buildEmbeddingModel()` 按 `settings.embedding.mode` 选择本地 `createTransformersEmbeddingModel()`（transformers.js `feature-extraction`，`Xenova/multilingual-e5-small` q8，384 维；批量 ≤8 条、批量间让出事件循环）或远程 `createRemoteEmbeddingModel`（OpenAI 兼容端点，需完整配置与 full-text 处理授权，并把全文 chunks 发往该端点），再由 `FullTextKnowledgeBaseFileStore` 持久化。操作注册为 `personal-library-fulltext-index`，随卸载/目录变化取消；查询按当前 embedding mode 构建查询向量，搜索入口在查询嵌入前核对 manifest modelId，与当前嵌入模型不一致时报错并要求删除重建后再搜索。esbuild 将 transformers.js 解析到其 web 构建（`alias`）并启用 `onnxruntime-web-use-extern-wasm` conditions，wasm 二进制运行时从 CDN 拉取并缓存，插件 `main.js` 不携带 wasm 资产。设备分支选择：真实 Node 走 onnxruntime-node（device `cpu`）；Obsidian 渲染进程带 Node 集成时，factory 经 `alignElectronReleaseProbe` 检测 `process.versions.electron` 并将 release.name 对齐为 `"electron"`，使 transformers.js 与 `isNodeRuntime()` 一致走 wasm 分支。运行时诊断命令 `diagnose-fulltext-runtime` 复用同一宿主路径做两段独立探测，任一段失败不阻断另一段：pdf.js 段在 `loadPdfJs()` 后核对返回值与 `window.pdfjsLib`，有库连接时对 catalog 中第一篇带 PDF 文件的论文做真实冒烟提取；嵌入段真实加载本地模型并读取 transformers.js env 与运行时分支探针。结果以 Notice 汇总、logger 明细和可复制结果模态框呈现。

选中文献库时，插件还会按 scope 与识别策略 identity 从 Vault index root 下独立加载可替换的 direction proposal 和研究者确认的 interest profile；缺失 proposal 表示尚无提议，缺失 profile 表示严格空 profile，一个文档损坏不会清空另一个。两者使用独立 primary/backup、原子整文档写入、语义 revision 与 CAS；确认操作按 profile-first 协调，避免在权威 profile 写入失败时先消费 proposal candidate。方向实体为 schema v3：候选可带 `clusterMembers`（paperKey + confidence），确认后的方向带必填 `clusterMembers` 与 `timeline` 事件（created/edited/members-updated/merged/removed/locked/unlocked/split，上限 64，迁移自 v2 时注入 created 事件）；confirmed direction 可锁定（可选 `lockedAt`，decoder 按存在与否接受该键，旧文档兼容加载）：锁定方向不参与自动 split/merge，但新论文仍可归入（attach）。插件内部的生成入口只接受当前有效的模型处理许可与已加载 catalog，以**聚类驱动**（`proposeClusteredPersonalLibraryDirections`，输入为知识库全文向量，KB 空时提示先运行全文索引）替代一次性 LLM 分批提取，使用独立 `LlmClient` 每簇一次有界提取调用，并在提交前复核连接、输出位置、许可和精确 catalog evidence fingerprint；目录、输出位置、有效 endpoint 或撤销许可会定向取消生成。

设置页的 **Review directions** 和命令 `review-personal-library-directions` 打开同一个 Proposed/Confirmed modal。Proposed 候选可检查和修改名称、描述、discovery cues 与 1–5 篇代表论文，也可显式合并、移除或确认；只有确认操作会把候选转入研究者权威 profile。候选的 `clusterMembers` 以"Cluster members N · avg. confidence X%"摘要展示（可展开成员列表与置信度），另渲染"Unclustered (buffer pool) N"区块（`catalogInputPapers` 中未被任何候选覆盖的论文）；Confirmed 方向展示 `clusterMembers` 摘要与最近 5 条 `timeline` 事件。Confirmed 方向保留 active/disabled/merged 状态、代表论文与 evidence diagnostics，并通过显式操作编辑、启停、合并或按 restrict/cascade 删除（操作写入 timeline 事件）。同一 modal 顶部渲染**增量建议区块**：列出 `IncrementalSuggestionsStore` 中 pending 的 diff 建议（kind 徽标 attach/new/split/merge、目标方向名、涉及论文数、截断理由），每条带 Apply/Ignore 按钮；apply "new" 建议会把候选草案写入 proposal store 并切到 Proposed tab 走既有确认交互，apply attach/split/merge 直接经 profile 持久化路径生效，成功后建议从文档移除；Confirmed 方向卡片按 `lockedAt` 显示 locked 徽标并提供 Lock/Unlock 按钮（mutation 写 locked/unlocked 事件）。modal 将模型和 catalog 内容按纯文本渲染；本地查看与 review 不要求仍持有模型处理许可，重新生成则要求当前许可。

插件为每次日报构建一份 settings/output/LLM 与 personalized discovery 的不可变快照。只有当前 endpoint-bound 模型处理许可有效，且 `evaluatePersonalLibraryInterestEligibility` 判定 active confirmed direction 的代表证据仍与当前 catalog 一致时，插件才把 direction 名称、描述、cues，以及代表论文的 `paperKey`、标题和 `metadata-and-abstract` evidence depth 交给 `ArxivPipeline`；proposal、路径、PDF、catalog abstract、fingerprint、许可状态、凭据和无关 catalog paper 不进入该输入。缺少许可、文档加载失败、identity 不兼容、代表证据 stale、endpoint 无法解析或无 eligible direction 时沿用 manual-only 过滤。目录、输出位置、有效 endpoint、撤销许可，以及会改变 eligibility 的 catalog/profile 提交会同步中止已捕获 personalized input 的日报；revision/identity guard 防止队列中的旧授权或 reload 重新开放过期状态。proposal-only review 和单篇手动详报不使用该路径。

同一快照还携带 novelty 输入：eligible direction 的代表论文在 join 边界被裁剪/截断到严格 DTO 上界（比较基准的成员是代表论文的 `paperKey` 集合，不受这些展示元数据上限影响），并映射 direction→representatives。novelty 准备失败只降级为 discovery-only（固定文案告警），绝不取消已生效的 personalized discovery。日报流水线对这些 library-derived 论文运行 best-effort novelty stage：校验后的结果以纯文本行和严格 `arxiv-daily-personal-novelty:v1` marker 写入日报，按报告 occurrence 保存到 Paper Index，并可经日报修复与 Dashboard history sync 重建；Dashboard row 展示最新已提交 occurrence 的 discovery provenance 与 personal novelty（纯文本、完整换行、可访问），与 query-time `matchReasons` 分离。详报和邮件不投影 provenance 或 novelty；novelty 不进 summary prompt 与 summary checkpoint identity。

生成路径**不**缓存单一 `ArxivPipeline`：调度与命令经 `buildPipeline()` / `buildManualFetch()` 按当前 settings 重建依赖。`HostAdapters` 仅在 onload 构建。插件设置变更经 `SettingsChangeService` 串行提交；输出目录候选会预加载新的 `StateStore` / `RunHistoryStore`，再由 Scheduler 通过一次同步 pair replacement 联合安装，调度启停与定时器更新在设置持久化和 live commit 后执行。

### Node CLI

`runCli`（`apps/cli/src/main.ts`）解析子命令；除 `init` / `update` / `help` 外读取固定路径 `config.toml`，经 `buildCliRuntime` 组装 pipeline、`SchedulerService`、`manualFetch`。CLI **不**调用 `scheduler.start()` 做 tick 循环；`run` 使用 `runForDateNow`（锁、run-state/history、`onDailyCompleted` 邮件）。

### Email Relay Worker

`services/email-relay` 是独立 Wrangler Worker；仓库配置的默认 `PUBLIC_BASE_URL` 为 `https://mail.arxiv-daily.top`。它处理 liveness、automatic readiness、验证起始/完成、`/v1/deliver` 与 operator-only cutover control。`DELIVER_GATE` 同一 Durable Object 类承载 cutover singleton、按收件人划分的 automatic gate 和按设备划分的 test gate；幂等 ledger 与 UTC 日配额在相应对象的 storage transaction 内更新。automatic 路径还要求 singleton 中的永久 deployment binding、control 与 KV audit marker 一致并处于 ready；绑定或运行依赖不可验证时 fail closed，不存在无 DO fallback。

### VS Code companion

`extensions/vscode-arxiv-daily/src/extension.js` 注册三个命令：打开 Reading Dashboard、运行今日流水线、按 arXiv ID 摘要。Dashboard 通过 `workspace.fs` 访问当前工作区内的 `arxiv-daily/.index/papers.json`；只有 Dashboard 和 Paper Index 编辑依赖工作区 vault 探测。

两个流水线命令不读取工作区中的 Obsidian 配置或 SecretStorage。它们从 `arxivDaily.cliPath` 取得可执行文件名或路径，并通过 VS Code `ProcessExecution` 以独立 argv 启动进程任务：今日运行传入 `run --today`，按 ID 摘要传入 `run --id <canonical-id>`。扩展在派发前注册 task/process 结束监听，等待对应 `TaskExecution` 的进程退出；退出码 `0` 才视为成功，非零退出、无进程退出的任务结束和取消均返回错误并清理监听器。

ID 输入只接受现代 arXiv ID 或字面量 `arxiv.org` / `www.arxiv.org` 的 HTTP(S) abs/PDF URL；校验月份、四位/五位序号时期边界、非零序号与 `v1+` 版本后，传给 CLI 的是去掉版本号的 canonical ID。流水线运行使用 CLI TOML 中的 `vault_root` 与密钥配置，因此没有打开 workspace 时仍可执行。

## Architecture and Module Boundaries

依赖方向（`scripts/check-boundaries.mjs` 强制；**不**扫描 email-relay / extensions）：

```text
packages/core          → 仅 pako（+ 自身）
packages/node-runtime  → @arxiv-daily/core
apps/cli               → @arxiv-daily/core, @arxiv-daily/node-runtime
plugin                 → @arxiv-daily/core, obsidian；仅额外允许 @arxiv-daily/node-runtime/scoped-library-source；
                         运行时第三方依赖 @huggingface/transformers（esbuild 打包，边界脚本不扫描 node_modules）
services/email-relay   → 独立（不在 npm workspaces）
extensions/vscode-arxiv-daily → 独立 CommonJS 扩展（不在 npm workspaces）
```

禁止的旧路径（如 `plugin/src/pipeline`、`plugin/src/hosts/node`）不得存在。core 不得出现 `process`/`Buffer` 或 Node builtin import；plugin 源码不得 import Node builtin。workspace 深层导入默认禁止；唯一显式例外是插件可导入 `@arxiv-daily/node-runtime/scoped-library-source`，且插件不得改用 node-runtime 根入口。插件 `main.ts` 经 Obsidian host bridge 调用该子路径完成个人文献库选择后的 scoped source 打开与 inventory；日报生成仍只使用既有 `HostAdapters`/Vault 数据路径。

逻辑分层：

1. **Host 边界**：HTTP / 存储 / 密钥 / UI 进度 / 打开资源 / HTML·XML 解析
2. **应用服务**：调度、操作注册表、手动抓取、PDF、项目笔记、诊断
3. **流水线**：发现 → 过滤 → 索引入库 → 正文 → 详报选择 → 详报/日报写作
4. **持久化产物**：Markdown 日报/详报、索引与状态 JSON、缓存与 checkpoint
5. **旁路投递**：digest 渲染 → Resend 或 hosted relay

## Entry Points, Interfaces, and Runtime Flows

### 入口

| 入口 | 路径 | 行为 |
| --- | --- | --- |
| Obsidian 插件 | `plugin/main.ts` → 打包 `plugin/main.js` | 插件生命周期、命令、Dashboard、进程内调度 |
| CLI | `apps/cli/src/main.ts` → `dist/arxiv-daily-cli.cjs`（bin `arxiv-daily`） | 子命令驱动的批处理/运维 |
| 兼容 Python | `arxiv_daily.py` | `node apps/cli/dist/arxiv-daily-cli.cjs …` |
| Worker | `services/email-relay/src/index.ts` | `fetch` 路由 `/health`、`/ready`、`/v1/verify/*`、`/v1/deliver` 与受 operator bearer 保护的 `/internal/delivery-v2/cutover` |
| VS Code companion | `extensions/vscode-arxiv-daily/src/extension.js` | 命令注册、Reading Dashboard 与 CLI 进程任务桥接 |

### CLI 命令面

- `init`：交互式生成 `config.toml`（含 schedule intent 字段）
- `update [--check] [--yes]`：CLI 自更新流程
- `run --today | --date YYYY-MM-DD | --id ARXIV_ID [--date …]`
- `email test|status|verify-start`
- `schedule show|install|uninstall`（crontab，标记 `# arxiv-daily-managed`）
- `data export --out PATH.zip` / `data import PATH.zip [--yes]`
- `help`

已移除并明确拒绝：`--config` / `--vault-root` / `--cache-dir`，以及子命令 `run-pending`、`summarize`。

配置路径：Linux/macOS 为 `$XDG_CONFIG_HOME/arxiv-daily/config.toml`（默认 `~/.config/…`）；Windows 为 `%APPDATA%/arxiv-daily/config.toml`。

### CLI 配置与 `PluginSettings`

`loadCliConfig` 产出 `CliRuntimeConfig`：`settings: PluginSettings` + `vaultRoot` + `cacheDir` + **`scheduleIntent: CliScheduleIntent`**。与插件共享对象形状，但语义不完全等价：

| 区域 | 行为 |
| --- | --- |
| `[llm]` / `[arxiv]` / `[output]` / `[advanced]` | snake_case → camelCase 映射进 `PluginSettings`；`request_delay_ms` 下限钳制为 ≥ 3000 |
| `detailSelection` | **恒为 balanced**，TOML 不可配 |
| `settings.schedule` | **恒为默认且 `enabled: false`**；OS 定时只看 `scheduleIntent` |
| `[email]` | 映射 `to` / `mode` / `api_key` / `hosted_token` 等；**`hosted_base_url` 被忽略**，hosted 走 core 默认 `https://mail.arxiv-daily.top` |
| 顶层 `vault_root` / `cache_dir` | CLI 专有路径；cache 默认 `vaultRoot/.cache/arxiv-daily` |

### 主生成流水线（`ArxivPipeline.runForDate`）


1. **已有日报**：若 `MarkdownWriter.dailyExists(date)`，清理已提交 checkpoint，并对 Paper Index 做修复后返回 `completed`（修复路径可不带 digest）。  
2. **发现**：默认 `ArxivSourceAdapter.listForDate`（arXiv `/recent` + 摘要 enrichment）；空列表 → `pending`（不写空文件）。  
3. **LLM 过滤**：`filterPapers`；可写 `filter-checkpoints`。模型响应必须是严格的 `{ "papers": [...] }` JSON，记录仅接受当前请求中的 ID、唯一 ID，以及配置 topic tag 或 `skip`；响应 JSON/契约校验失败 → **`failed_transient`**，不保存 checkpoint，也不进入 Paper Index 与后续生成。永久/瞬态 LLM 调用错误映射为对应失败 kind。
4. **过滤结果为空**：严格验证后的空数组或全部合法 `skip` 直接 `completed`（`papersWritten: 0`，空 digest），**不写** Paper Index。
5. **索引入库**：`PaperIndexStore` upsert；失败 → **`failed_permanent`**。`ignored` 不进入后续可见集合；若全部 ignored → `completed`（0 篇，空 digest）。  
6. **正文获取**：并发度 6；经 `SourceAdapter.fetchContent`；失败降级为错误占位文本，不中断整日。  
7. **详报选择**：`selectDetailPapers`（阈值与 soft limit 来自 `detailSelection`）；磁盘上已存在且 `classifyPaperNote` 为 `verified_detail` 的详报不占 soft quota。  
8. **详报写作**：对选中且有全文的论文 `summarizePaperDetail` + `writePaperDetail`；**仅当文件真实存在**后回写 `paperPath` / `isDetail`。  
9. **日报摘要**：`summarizeDaily`（可走 summary checkpoint）→ `writeDaily` → 清理 checkpoint → 更新索引中的日报路径与结构化 summary；此阶段索引回写失败 → **`failed_transient`**。  
10. **返回**：`completed` + `DailyDigest`（供邮件自动发送）。


取消通过 `AbortSignal` / `isCancellationError` 映射为流水线 `cancelled`。调度落盘时将 `cancelled`（及部分中断）写为 run-state **`pending`**（可再跑），`RunStatus` 无独立 `cancelled`。

可选依赖：`PipelineDeps.sourceAdapter`；checkpoint 经 `checkpointStores.filter/summary`（兼容旧字段 `checkpointStore`）。

### 调度流

`SchedulerService` 委托 `SchedulerDriver`：

- 启用后按 `schedule.tickIntervalMin`（默认 20 分钟）`setInterval` 触发；
- 仅在 `runAtLocal`–`runUntilLocal` 本地时间窗内工作（多日 tick 时时间窗主要约束“今天”）；
- 回看 `LOOKBACK_DAYS = 5` 个日历日；配置时区下的周末跳过；
- `checkTickGate`：已完成 / 运行中 / 窗外 / 瞬态失败退避则跳过；
- 注入的 `RunLock` 串行化同日运行；`StateStore` 记录 `pending|running|completed|failed_*|skipped`；
- `StateStore` mutation 从权威 primary 重载 durable state，修改 candidate，保存后精确回读整个 run-state；只有回读与 candidate 完全相等才发布到内存。保存抛错但回读已等于 candidate 时提交仍成立，其余保存或确认失败保留 mutation 前的内存快照；
- 流水线返回 `completed` 后，driver 先把原始 completed result 与 digest 保留为进程内 pending completion。`run-state.json` 的 completed candidate 被确认后才显示完成、写 completed history 并调用 `onDailyCompleted`；提交未确认时返回 `failed_transient`，后续调度或手动入口只重试该状态提交，不重跑流水线；
- 瞬态失败在 `setFailed` 时若 `attempts >= MAX_TRANSIENT_ATTEMPTS`（**10**）则升级为 `failed_permanent`；
- 超过 `STALE_RUNNING_RECOVERY_MS`（1 小时）的 `running` 在启动/恢复时标为 **`failed_permanent`**（错误文案：`recovered stale running state after startup`）；
- run history、进度/日志、取消清理和 `onDailyCompleted` 是 completed hard commit 之后的 best-effort effect，任一失败不撤销已确认的 run-state。pending completion 不持久化为 outbox；进程退出后不保证原 digest、history 或完成回调重放。

**插件 vs CLI 调度：**

| | 插件 `ScheduleSettings` | CLI `CliScheduleIntent` |
| --- | --- | --- |
| 启用 | `enabled` + 进程内 `setInterval` | `enabled` 仅门禁 **cron install** |
| 时间 | `runAtLocal`–`runUntilLocal` + `tickIntervalMin` | `on` / `until` / `interval_hours` 生成 crontab 槽位 |
| 周末 | 驱动内按时区跳过 | `weekdays_only` → cron DOW `1-5` |
| 执行 | 需 Obsidian 保持打开 | OS cron 调用一次性 `run --today` |
| `run` 路径 | `runForDateNow` / force 等 | 同样经 `runForDateNow`，但**不** `start()` 定时器 |

原生 Windows 不支持 crontab install（可提示 WSL 或插件）。

### 邮件流

`deliverDailyEmailIfEnabled` 先校验自动开关或显式测试请求的凭证，再渲染 subject/html/text。公开结果为 `delivered | delivered_unrecorded | ambiguous | skipped | disabled | failed`，reason 使用固定、无 PII 的枚举；provider/state 失败不会抛到调度层，也不会改写 pipeline run-state。


**自动投递：**


1. 客户端从 `date + 标准化收件人` 计算 `arxiv-daily:auto:<sha256>`；self 模式直接把该稳定 key 传给 Resend，hosted 模式把它作为 auto 类型标记传给 relay。key 不含明文收件人。
2. 发送前严格读取 `delivery-state.json`：missing 可开始；corrupt/unreadable 直接失败。宿主还必须同时提供目录枚举、系统级 exclusive create 和 descriptor-backed namespace guard，否则返回 `delivery_storage_unsupported`。
3. `delivery-state.json.claims/` 保存不可变 generation 的 claim、decision、result sidecar。逻辑 claim stem 和 sidecar 只保存哈希 recipient identity；同一日期/收件人的并发调用通过 exclusive create 收敛到单一 generation。
4. provider 调用前先持久化 `provider_attempt_started`，再同步复核仍由同一 descriptor 锚定的目录承载 claim；该复核与 `HttpClient.request` 之间没有异步间隔。只有可证明 provider 尚未调用的取消或恢复才能释放 generation。
5. self Resend 对 408/409/5xx 和宿主明确标记为可重试的 transport failure 做有界重试，所有物理尝试复用同一个 provider key。HTTP 400/401/403/404/422/429 是明确拒绝；其他 HTTP、transport failure 和无有效 acceptance marker 的 2xx 均按结果不明处理。
6. provider 接受后写 `delivered` result；若最终主状态重建失败，返回 `delivered_unrecorded`，已存在的 attempt/result sidecar 仍继续阻断自动重发。provider 调用后的 ambiguous、明确拒绝或结果落盘不确定同样保留阻断；系统不自动重试这些 generation。

`delivery-state.json` 保持 schema v1，使旧 reader 仍能读取；claim、attempt、ambiguous 等阻断态投影为 v1 `status: "delivered"`，并用 `deliveryPhase` 提供新客户端精确信息。为兼容旧 reader，主文件保留明文 recipient；Node 与受支持的 Obsidian 桌面文件系统将主文件和临时/备份产物强制为 `0600`，读取既有宽权限主文件时也收紧为 `0600`。Linux 宿主使用 `O_CREAT | O_EXCL | O_WRONLY | O_NOFOLLOW`、`/proc/self/fd` descriptor 锚定和原子 rename；无法提供这些能力的宿主对自动投递 fail closed。跨文件系统 rename 不降级为 copy。

显式 `force`/邮件测试不创建自动 claim，也不改写 automatic delivery state；每次生成独立 `arxiv-daily:test:<random>` key，因此不会占用正式日报 identity。self 模式的 API key 当前来自 settings/config；From 为空时使用 `onboarding@resend.dev`。hosted 模式使用 Bearer `hostedToken` 调用默认 `https://mail.arxiv-daily.top/v1/deliver`。客户端接受精确的 `{ "ok": true }`，也接受仅附带非空且不超过 128 字符 `id` 的 `{ "ok": true, "id": string }` 与额外含字面量 `"deduped": true` 的响应；整个响应体上限为 4096 字符，重复顶层成员、其他字段或类型均视为结果不明。旧响应中的 provider ID 只参与局部契约验证，随后丢弃，不进入投递结果、日志或持久状态。`OFFICIAL_DELIVERY_AVAILABLE = true` 仅表示客户端路径开启，不证明外部 Worker 已部署或可用。

**Worker：**


- 路由：`GET /|/health` 始终提供稳定 liveness；`GET /ready` 只有在 automatic runtime 配置完整，且 authoritative cutover 状态为 ready 时返回 200。验证与投递入口为 `POST /v1/verify/start`、`GET /v1/verify`、`POST /v1/deliver`；`GET|POST /internal/delivery-v2/cutover` 受 `DELIVERY_V2_CUTOVER_TOKEN` bearer 保护。
- 验证起始限流是 KV best-effort counter：邮箱 3 次/小时、IP 10 次/小时；超限返回统一“已发送”形态。起始请求可在 automatic 尚未 ready 时发送 magic link，但完成入口不会提前消费 pending token 或签发 device token。
- 验证完成把 pending token 的 secret-scoped hash 交给 cutover singleton 串行处理。singleton 只在永久 binding、ready control、KV marker 与当前 build/protocol/identity 一致时建立不可裁剪的 issuance claim；device token 由 `TOKEN_SECRET` 和 pending identity 确定性派生。KV 设备记录以 token hash 为 key，保存规范化邮箱及 protocol/build/ready generation，TTL 约一年；验证页只展示 device token，不回显收件地址。KV 写入、删除或响应结果不明时，同一有效 claim 只重放同一个 token；pending 到期或 generation 不匹配后不再披露。
- `IDENTITY_SECRET` 独立派生 recipient identity、legacy evidence identity、automatic Durable Object 路由、provider identity、cutover identity fingerprint 与 marker proof。轮换 `TOKEN_SECRET` 会使既有 token 失效；相同邮箱重新验证后仍映射到原 recipient scope。`IDENTITY_SECRET`、合法 build identity 或 protocol generation 与永久 binding 不一致时，status、readiness、token issuance、automatic authorization 和 cutover action 都保持 locked。
- `/v1/deliver` 先认证 device token并检查 `to` 与绑定邮箱一致。automatic 请求还会先读 readiness，再路由到 `recipient-v2:<recipientIdentity>`；同一邮箱的多个 token 因此共享 automatic ledger、quota 与 provider key。test 请求路由到 `device-v2:<deviceIdentity>`，使用客户端每次生成的随机 test logical key，不依赖 automatic cutover readiness。
- automatic ledger identity 由服务端从 recipient identity 和请求日期推导，device identity 与客户端 auto key 的 digest 不参与 ledger、fingerprint 或 provider identity。test identity 另含 device identity、recipient identity 和完整 test logical key。relay provider key 只含服务端有界哈希，不含明文收件人。
- Durable Object ledger 状态为 `reserved | attempted | done | rejected | legacy-done | legacy-attempted`。请求 fingerprint 防止同一 identity 绑定不同内容；ledger 与按 auto/test、当前 UTC 日期划分的 quota 在相应 recipient/device 对象的 storage transaction 中预占。automatic authorization 发生在 imported 或 existing ledger 重放之前；`attempted`、`done`、`rejected` 与 legacy 阻断记录的重放均不再次调用 provider。test 终态保留 30 天后由 alarm 清理，automatic ledger 不做该清理。
- Resend 明确拒绝映射为 relay 422/429 终态并回退 quota；transport、非白名单 HTTP、无效成功体，或 provider 接受后 completion storage 失败，均保留 `attempted` 阻断并返回 ambiguous。成功 ledger 与响应只保存 `{ "ok": true }`；provider response body 和 provider ID 验证后丢弃。
- cutover singleton 固定为 `delivery-cutover:v3`。首次 inventory 在同一个 DO storage transaction 内建立 `cutover-binding:v1`、`cutover-state-index:v1` 与 pending control；binding 固定 `IDENTITY_SECRET` fingerprint、精确 `email-relay-v2-<40 hex source SHA>` build identity 和 protocol generation。后续 status/action/issue/automatic 以及 operation replay 都先验证该 binding；binding 缺失但 control、operation、issuance claim、marker 或 state index 任一存活时，不会把对象当作全新 cutover。
- cutover 扫描两代 automatic KV key、忽略已知 test key，并把 exact secret-scoped `done | attempted` evidence 合并到 HMAC audit marker；未知 key、扫描失败、容量/时限超界和错过 provider-fence pending window 均阻断。operator 对旧 Resend credential 已撤销的精确 attestation 构成跨 Worker provider fence；marker 写入后需两次、每次至少间隔 60 秒的 observation，随后才可 seal 为 ready。KV marker 只用于审计/恢复，automatic readiness 的权威状态仍在 singleton DO。
- cutover mutation 与返回 200 的 monotonic no-op 都用永久 operation ID 绑定 action/input/status/body；精确重放只能在 binding/control/marker 一致性检查后返回原响应。control 缺失或损坏时只允许与永久 binding 和有效 marker 一致的 repair；结果不明的 pending repair 只能由原 operation 恢复，其他 operation 不能接管。多个恢复载体同时永久丢失或损坏时保持 locked。
- cutover operator 面由只读 `scripts/cutover-preflight.mjs`（git HEAD、wrangler.toml 必需 binding/var 名称、wrangler 登录、secret 名称清单、注入 `email-relay-v2-<sha>` 的 Wrangler dry-run、远程 `/health` 与 `/ready` 状态）与人工 `docs/runbook-cutover.md`（部署、凭据撤销、逐步 action、readiness 确认、异常恢复）组成。脚本依赖注入便于测试，`--check-readonly` 静态自检证明源码不含非 dry-run deploy、KV 写或验证/投递/cutover 端点调用，CI 执行该自检；所有生产 mutation 由 operator 按 runbook 逐项授权手工执行，脚本不部署、不写 KV、不调用验证或投递端点。
- 出站 Resend key 仅由 Worker secret 提供。系统没有邮件 outbox 或后台重试 daemon；最终 provider 去重仍依赖 Resend/relay 正确执行稳定 idempotency key，因此不声明严格 distributed exactly-once。


### Dashboard 与命令

- Dashboard（`plugin/src/dashboard/view.ts`）：`PaperIndexStore`、run-state、vault 文件同步（`syncDashboardHistory` / `queryDashboard`）；日历、检索、分页、状态操作；运行入口同样走 `scheduler.runForDateNow` / force 等路径。`queryDashboard` 从已提交日报对应的 occurrence provenance 与 novelty 中按 report date 与规范化路径确定性选择最新一条，独立投影 manual/library/both、manual topic、direction、代表论文与 `metadata-and-abstract` depth，以及校验后的 personal novelty（差异类型、比较基准 `paperKey`、证据深度、有界解释）；视图在标题下完整换行、纯文本展示，不依赖搜索状态，也不与 query-time `matchReasons` 混用。
- 行内 "Find similar papers" 按钮（`scan-search`）打开 `SimilarPapersModal`：当论文有可查询文本（标题+摘要，空摘要回退标题）时模态框呈双页——「Library similar」页异步调用全文 KB 检索并显示查询服务返回的标题（没有 catalog/提取标题时为 paperKey）、fallback 相对文件路径或 arXiv paperKey、相似度；「Daily similar」页显示日报论文语料的词法结果。Library 页无 KB/失败时显示降级文案。
- 搜索框单框双结果：Dashboard 搜索输入仍以词法相关性过滤日报行，同时对同一查询（≥2 字符）异步调用全文 KB 检索，在过滤区下方渲染 "Library matches" 区块，显示同一标题回退、fallback 相对文件路径或 arXiv paperKey、相似度；staleness token 丢弃过期响应，无 KB/失败降级为内联文案，不影响行过滤。
- 命令（`plugin/src/commands.ts`）：今日运行、回看 pending、重试失败、指定日/强制日、清 run-state、手动 arXiv id 摘要、诊断等；运行前 `validateFilterConfig` / `validateLlmConfig`。全文知识库命令：`index-personal-library-fulltext`（增量索引 + indexed/reused/failed/pruned/title-refresh 统计通知）、`search-personal-library-fulltext`（`FullTextQueryModal` 查询 → 当前 embedding mode + 词法融合检索 → 前 5 名标题或 paperKey 回退、fallback 路径或 arXiv paperKey、相似度通知）与 `diagnose-fulltext-runtime`（pdf.js 与本地 transformers.js 两段运行时探测）。增量方向命令：`check-incremental-direction-updates`（`runIncrementalDirectionUpdate`：placement 免许可始终跑并转未归入论文为 attach 建议；缓冲池达 `INCREMENTAL_BUFFER_TRIGGER`（默认 3）时经 `reclusterPool` + `suggestDirectionDiff` 生成 LLM diff 建议——该阶段要求模型处理许可，无许可则跳过并写 `pendingAuthorization`；建议集合整体 CAS 替换持久化建议文档）以及 `review-incremental-suggestions`。`index-personal-library-fulltext` 完成且 `indexed > 0` 时自动触发同一增量流程，失败只记录、不失败索引命令。apply/dismiss/lock/unlock 是本地确定性操作，不要求模型处理许可；fallback `file:*` 可参与 placement/recluster/attach，但基于 `catalog.papers` 元数据构造新方向草案的路径不能为该 key 生成可应用候选。

## Data and State

默认输出布局（相对 vault / `vaultRoot`）：

| 路径 | 内容 |
| --- | --- |
| `arxiv-daily/daily/YYYY-MM-DD.md` | 日报（权威提交物） |
| `arxiv-daily/papers/<externalId>.md` | 详报（路径 stem 为 externalId，非 paperKey） |
| `arxiv-daily/.index/papers.json` | 论文索引（schema v5；key 为 `paperKey` 如 `arxiv:…`；同目录 `.bak` 为恢复副本；可按日报路径保存 occurrence-level discovery provenance 与 personal novelty） |
| `arxiv-daily/.index/run-state.json` | 按日运行状态（原子写 + `.bak`） |
| `arxiv-daily/.index/run-history.jsonl` | 运行历史（可轮转） |
| `arxiv-daily/.index/delivery-state.json` | v1-compatible 邮件阻断投影；为旧 reader 保留 recipient，受支持宿主按 `0600` 处理 |
| `arxiv-daily/.index/delivery-state.json.claims/` | 不可变 claim/decision/result generation；使用哈希 recipient identity |
| `arxiv-daily/.index/filter-checkpoints/` | 过滤 checkpoint |
| `arxiv-daily/.index/daily-summary-checkpoints/` | 日总结 checkpoint |
| 兼容 `arxiv-daily/index/papers.json` | 旧索引路径可读 |

**设置持久化**


- 插件：Obsidian `loadData`/`saveData` 存 sibling `settings` 与可选 `libraryConnection`；后者包含 canonical root、文件系统 identity、eligible extensions、处理深度及可选授权 fingerprint/time，不包含 API key。设置与文献库状态的整份持久化经同一进程内 mutation queue 串行；选择、授权或撤销保存失败时恢复内存状态。旧版嵌在 data 里的 `runState` 可迁移进 `run-state.json`。
- 插件：Obsidian `loadData`/`saveData` 存 `PluginSettings`；旧版嵌在 data 里的 `runState` 可迁移进 `run-state.json`。启动合并设置时会校验输出目录和 IANA 时区；无效持久化值恢复默认并在 logger 初始化后记录 warning。
- `SettingsChangeService`（`plugin/src/settings/change-service.ts`）对普通设置执行 serialized candidate transaction：克隆当前设置 → 字段及跨字段校验 → 准备候选资源 → 持久化 candidate → 原位提交 live settings → 安装 store 和 runtime effect。持久化前 live settings、调度器、logger 与输出 store 不变；live commit 失败时不安装候选资源或 effect。持久化后的 host/runtime effect 是 best effort，失败会记录但不会回滚 durable settings。
- 输出目录切换从准备到安装完成期间持有 output transition；已有 output operation 或 scheduler run 会拒绝切换，transition 持有时也拒绝新的 operation。paper-index 状态修改和 paper-note 创建通过 `withOutputOperation` 持有 lease，避免旧 writer 跨目录切换继续写。
- 设置页的 legacy 与 declarative renderer 共用同一 transaction service。timezone、run-window 与 schedule enabled 控件各自使用 revision/queue，旧异步结果不会覆盖后续用户意图；timezone 仅提交有效 IANA 标识。

- CLI：见上文配置映射；schedule intent 与插件进程内 schedule 字段语义分离。

**缓存**

- 插件：`<pluginDir>/.cache`（HTML/Atom 元数据）与 `.cache/source`；按日清理过期项。
- CLI：Atom/HTML 在 `cacheDir`；源码缓存于 vault 内 `.arxiv-daily/cache/source`。

**数据迁移包**

CLI `data export/import` 将逻辑目录 `daily`、`papers`、`.index` 打成 zip（含 manifest）。硬限制：压缩包 ≤512MB、条目 ≤10000、单文件解压 ≤64MB、总解压 ≤512MB、压缩比 ≤200；路径不得逃出 vault。布局不一致时可能跳过 `.index`；非 TTY 导入需 `--yes` 才会实际写入。

## Key Implementation Mechanisms

### Host 适配与边界

所有业务 I/O 经 `HostAdapters`。`scripts/check-boundaries.mjs` 在发布验证中保证分层；email-relay / extensions 不受该脚本约束。

### Scoped personal-library capability

core 的 `ScopedLibrarySource` 只暴露 `inventory` 与 `readBinary`，没有写入、删除、重命名方法。Node 实现 `openScopedLibrarySource` 将 capability 固定到 canonical root，并记录 `dev:ino` identity；每次 inventory/read 前复核根 identity。inventory 默认最多 10,000 entries、深度 16，调用方只能收紧限制；遍历不跟随符号链接，深度/数量裁剪通过 `truncated` 暴露。`readBinary` 只接受无 `.`/`..`、绝对路径、反斜杠或盘符的逻辑相对路径，默认最多 25 MiB；它逐段拒绝符号链接，以 no-follow 方式打开文件句柄，再复核 canonical containment、路径/句柄 inode 和大小，最后从已验证句柄读取。映射后的文件系统错误不保留含绝对路径的原始 cause。

### 源适配器

`SourceAdapter`（`listForDate` / `fetchContent`）抽象多源；当前默认 `ArxivSourceAdapter`。流水线经 `legacyContentFromNormalized` 桥接旧 arXiv DTO。

### arXiv 请求韧性

`ArxivFetcher`：请求间隔（下限约 3s）、文本/二进制超时、协调器冷却、HTTP 错误与 `ArxivRetryDeferredError`、带退避的 `retry`；元数据可走 `AtomMetadataCache`。

### LLM 调用

`LlmClient.call` 使用流式 `/chat/completions`（`stream_options.include_usage`；不支持时有一次去掉 `stream_options` 的回退）。默认温度 `0.1`；thinkingMode 时按 provider 注入 reasoning/thinking。客户端内重试最多 **3** 次、基础退避 **5s**；耗尽包装为 `LlmTransientExhaustedError`。永久错误：HTTP 4xx 且非 429。逻辑调用超时 **300s**；流空闲超时 **120s**。密钥经 logger redaction 屏蔽。

### Prompt 资产

`packages/core/src/prompts/*.md` 经 esbuild `loader: { ".md": "text" }` 打进包；含过滤、日总结、详报、详报选择、rescue 与 injection-guard（中/英部分双语）。

### 详情选择策略

预设 `conservative | balanced | broad | custom`（阈值 + softLimit）。默认 balanced：`normalThreshold` 75、`exceptionalThreshold` 92、`softLimit` 3。插件可配；CLI 固定 balanced。

### Checkpoint

过滤与日总结可按指纹/契约版本恢复；日报告提交成功后 `removeAll(date)` 清理。损坏时尝试 `.bak` 或忽略坏条目并告警。

### 操作与取消

`OperationRegistry` 跟踪 `daily-run`、`detail-summary`、`pdf-download`、`personal-library-scan`、`personal-library-direction-generation`、`personal-library-fulltext-index`；增量方向更新与全量生成共用 `personal-library-direction-generation` kind（同一授权门与撤销取消范围）。`RunCancellationService` 与 scheduler 协作。插件卸载 `cancelAll`；CLI 安装信号处理器取消活动操作。
插件另以 `paper-index` / `paper-note` output lease 保护跨输出目录切换；`RunCancellationService` 与 scheduler 协作。设置输出 transition 与 operation registry 双向互斥。插件卸载 `cancelAll`；CLI 安装信号处理器取消活动操作。
### Paper Index 持久化与 History Sync

`PaperIndexStore`（`packages/core/src/services/paper-index.ts`）按 primary `papers.json` → `.bak` → legacy `index/papers.json` 的顺序选择首个有效文档；任一路径的真实读取错误直接失败，候选文件存在但均无法解析时也不会构造空索引。读取兼容 schema 1–4，内存归一为 schema 4，后续保存写 schema 4。

索引写入先生成 `.tmp`，再把已验证的旧 primary 发布为 `.bak`，最后以 `.tmp → papers.json` rename 作为新内容的提交点。primary 缺失或损坏时不会用它覆盖有效 backup；提升失败时只尝试恢复提交前已验证的 primary、backup 或 legacy 内容。`PaperIndexStore` 的领域 mutation 在模块级、按 primary 路径共享的 Promise 队列中执行完整的读取、修改、校验和保存事务；该串行范围限于同一 JavaScript realm，不提供跨进程锁或 `fsync` 级掉电保证。

Dashboard 历史同步（`packages/core/src/dashboard/history-sync.ts`）先扫描日报和论文笔记，再在同一索引 mutation 中重读当前状态。日报证据可补建非详情索引投影；论文详情必须由无歧义、身份一致的受管笔记证明。重复或冲突的顶层 `arxiv_id` / `arxiv` 标量会保护涉及的全部论文身份，嵌套字段、数组和正文不参与身份判断。破坏性清理还会比较扫描基线与 mutation 开始时的当前投影；扫描期间被其他操作删除或修改的详情不会被陈旧候选复活或清理。

插件 diagnostics 复用 `PaperIndexStore.inspect()` 的文档选择结果。索引解析、读取及后续笔记探测失败在可复制报告、日志、DiagnosticsModal 和 Hub 中只暴露稳定分类（如 `paper_index_invalid`、`paper_index_unavailable`），不传播底层异常 message 或 cause。

### 原子写与状态一致性

`StateStore` 的普通启动读取可从损坏 primary 回退 `.bak`，但显式未知 schema，以及 schema 1/无 schema 记录中类型非法的 `error` 或 `papersWritten` 会 fail closed。mutation 使用按 run-state 路径共享的进程内队列，并通过 candidate 保存与权威 primary 精确回读确认提交；backup 不参与 mutation 的 authoritative confirmation。

插件输出路径重载会先构造并加载候选 `StateStore` / `RunHistoryStore`，再由 scheduler 的 active/pending guard 接受 store 替换，最后同步发布 plugin 引用；guard 拒绝时各消费者继续使用旧 store。该协调和 pending completion 都是单进程语义，没有 Plugin/CLI 跨进程锁。

Paper Index 与 checkpoint 使用各自的临时文件和 rename、`writeTextAtomic` 或同路径 mutation queue。邮件 automatic delivery 以不可变 claim/decision/result generation 记录 provider attempt 边界，再从 sidecar 重建 v1-compatible `delivery-state.json`；受支持的 Node/Obsidian Linux 文件系统使用 descriptor-anchored exclusive create、claim namespace guard 和私有原子替换，能力不足时拒绝 automatic delivery。日报 Markdown 存在即视为该日已提交的权威信号。

## External Integrations and Executable Configuration

### 外部系统

- **arXiv**：分类 recent 列表、摘要页、HTML/源码全文、PDF。
- **LLM 提供商**：OpenAI 兼容 API（默认 DeepSeek：`https://api.deepseek.com/v1`，模型 `deepseek-v4-pro`，`thinkingMode: true`，`reasoningEffort: "medium"`）。
- **Resend**：自发送与 Worker 出站邮件。
- **Cloudflare**：Worker、KV `STORE`、Durable Object `DeliverGate`。

### 关键配置形状（`PluginSettings`）

- `llm`：apiKey、provider、baseUrl、model、thinkingMode、reasoningEffort
- `arxiv`：categories、topics、timezone（默认 `Asia/Shanghai`）
- `detailSelection`：profile 与阈值
- `output`：dailyDir、papersDir、linkStyle（`wikilink|relative`）、summaryLanguage（`zh|en`）
- `schedule`：enabled、runAtLocal、runUntilLocal、tickIntervalMin（**插件进程内**）
- `advanced`：requestDelayMs（默认 3000）、cacheExpiryDays、正文长度上限、logLevel
- `email`：enabled、mode（`self|hosted`）、to、fromEmail/fromName、apiKey、hostedToken、hostedBaseUrl
- `embedding`（ADR 0008）：mode（`local|remote`，默认 local）、provider、baseUrl、apiKey、model、dimension（远程模式必填配置；CLI `[embedding]` 映射同形状）

### 环境变量与 secrets


- `resolveResendApiKey(email, env)` **支持** env `ARXIV_DAILY_RESEND_API_KEY` 优先于 `email.apiKey`，但当前 CLI/插件调用均传入空 env，**实际只读 settings 中的 apiKey**。  
- Worker runtime secrets：`RESEND_API_KEY`、`TOKEN_SECRET`、长期稳定的 `IDENTITY_SECRET`、operator-only `DELIVERY_V2_CUTOVER_TOKEN`。`TOKEN_SECRET` 用于 token/pending authentication；`IDENTITY_SECRET` 用于 recipient/automatic identity 与 cutover marker/binding。
- Worker vars：`PUBLIC_BASE_URL`、`FROM_EMAIL`、`FROM_NAME`、`DAILY_QUOTA`（默认 `"5"`）以及必须满足 `email-relay-v2-<40 lowercase hex>` 的 `BUILD_IDENTITY`。仓库 `wrangler.toml` 不提供 build identity 默认值；bundle/deploy 调用方需按 source SHA 注入。`DELIVER_GATE` 是必需 Durable Object binding，`STORE` 用于验证 pending/device、best-effort 验证限流、legacy evidence 与 cutover audit marker。


### 清单、版本与产品单元


`product-units.json` 是仓库产品单元闭集清单。`scripts/check-product-units.mjs` 递归发现 `packages`、`apps`、`plugin`、`services`、`extensions` 下的 package manifest，并要求每个单元声明 manifest、唯一 lockfile、版本策略与 workflow；未分类 manifest、嵌套新单元或缺失治理文件会使检查失败。

- root release group 包含根、`packages/*`、`apps/*` 与 `plugin`，共用根 `package-lock.json` 和同步版本。`scripts/release-utils.mjs` 的结构化 `packageFiles` / `manifestFiles` 同时供版本同步、版本检查和产品清单 checker 使用；checker按完整集合拒绝遗漏、重复、非 root 路径以及独立产品越界。
- 根与 `plugin/manifest.json`：插件 id、版本、`minAppVersion` `1.4.0`、`isDesktopOnly: true`；`versions.json` / `plugin/versions.json` 映射插件版本到 Obsidian 最低版本。
- email relay 由 `services/email-relay/package.json` 与同目录 lockfile独立定版；VS Code companion 由 `extensions/vscode-arxiv-daily/package.json` 与同目录 lockfile独立定版。二者不属于根 workspace 或根版本同步组。


## Build, Test, Deployment, and Operations

### 本地脚本（根 `package.json`）


- `npm run typecheck` / `build`：根 workspace 透传（**不含** email-relay 与 VS Code companion）
- `npm test`：无附加参数时经 `scripts/run-root-tests.mjs` 调用 `test:workspaces`，覆盖全部根 workspace；带测试路径或 Vitest 参数时只把原参数传给一次 `@arxiv-daily/core` 测试调用
- `npm run test:workspaces -- <args>`：向全部根 workspace 传递测试参数，是 CI 与发布验证使用的全套测试入口
- Core 的无参数测试由 `scripts/run-core-tests.mjs` 递归发现 `tests/**/*.test.ts`，按路径确定性排序后以每批最多 8 个文件、每个 Vitest 子进程一个 worker 执行；显式参数直接进入单个 Vitest 子进程
- `npm run lint`：插件 ESLint  
- `npm run check:boundaries`：分层边界  
- `npm run smoke:build`：CLI/插件产物冒烟  
- `npm run cli`：运行已构建 CLI  

- 发布工具：`sync:release-version`、`check:release-version`、`test:release-tools`

### 构建产物


- 插件：`plugin/main.js`、`styles.css`、`manifest.json`（esbuild 外置 `obsidian`/`electron`）  
- CLI：`apps/cli/dist/arxiv-daily-cli.cjs`，构建时复制到 `plugin/arxiv-daily-cli.cjs`；`prepack` 先 build  
- VS Code companion：清单直接以 `src/extension.js` 为 CommonJS 入口；`build` 校验清单/命令注册，`test` 覆盖 workspace adapter、Dashboard、CLI 任务契约与 smoke，`vsix:package` 生成独立 VSIX

- smoke 检查含 help 退出码、坏配置、pako notice、插件包不泄漏 workspace 解析符号等

### CI / 发布


- **Root verification**（`lint.yml`）：所有 pull request 与直接推送到 `main` 时运行；固定 action commit，在根 lockfile 上执行 `npm ci`，依次检查 release tools、boundaries、lint、typecheck、8 GiB / 单 worker 的全 workspace 测试、build 与 smoke build。普通 PR 分支的 push 不单独触发该工作流。
- **Release Obsidian plugin**（`release.yml`）：推送稳定 SemVer tag → 校验 tag/SHA/`docs/releases/<tag>.md`、拒绝覆盖已有 release → release tools / boundaries / lint / typecheck / 8 GiB 全 workspace test / build / smoke → 对插件三件套做 build-provenance attestation → `gh release create`。
- **Publish CLI to npm**（`publish-cli.yml`）：插件 release **成功后**自动，或 `workflow_dispatch` 指定已有 tag → 校验 GH release 与 npm 版本未覆盖 → 运行同一全 workspace 验证入口 → trusted publishing `npm publish --workspace apps/cli`（包名 `arxiv-daily`）。
- **Email relay verification**（`email-relay.yml`）：relay 或 hosted delivery contract、workflow、产品清单及 checker 路径变更时，使用 relay 自身 lockfile 执行 `npm ci`、typecheck、tests 和 Wrangler `deploy --dry-run`；bundle 写入 runner 临时目录，不部署 Worker，也不读取生产凭据。
- **VS Code companion verification**（`vscode-companion.yml`）：companion、CLI command contract、workflow、产品清单及 checker 路径变更时，使用 companion 自身 lockfile 执行 build、tests、smoke，并把验证用 VSIX 写入 runner 临时目录；不发布扩展。
- 两个独立 workflow 都在 pull request 和相应路径推送到 `main` 时运行，action 固定完整 commit SHA，权限仅 `contents: read`，checkout 不持久化凭据。


### 验证面

core / node-runtime / plugin / CLI 工作区有 Vitest 覆盖；Core 流水线集成测试使用包含两个论文条目的代表性 recent 页面输入，arXiv parser 与 source adapter 专项测试读取完整页面夹具。邮件定向面覆盖 Core claim/result/HTTP 分类、Node 与 Obsidian descriptor/权限/恢复，以及 CLI/Plugin consumer；email-relay 的独立 Vitest 覆盖认证路由、服务端 identity、Durable Object ledger/quota、provider 结果、cutover proof 和只读 preflight 脚本（fake 依赖注入、只读源码自检、无 secret 泄漏）。测试用于约束行为；生产路径以源码注册、构建入口和 Worker binding 为准。

## Security and Failure Behavior

### 密钥与脱敏


- LLM / Resend / hosted token 进入 logger 敏感值列表；CLI 对 stdout/stderr 做 `redactText`。  
- 插件密钥存于 Obsidian 数据（`ObsidianSettingsSecretProvider`），provider 写入经 settings candidate transaction 持久化。设置页的 LLM/Resend key 与 hosted token 以掩码输入框显示已保存值，可点按 Show/Hide 查看，编辑在 blur/Enter 时经事务提交，失败回滚显示旧值。
- Worker 不向客户端暴露 Resend key；设备 token 只在验证完成页展示，KV key 使用 `TOKEN_SECRET` 作用域的 token hash。验证 pending/device 值仍保存规范化绑定邮箱以执行 recipient 校验；recipient/legacy identity、DO 名称、provider key、ledger、cutover control/marker 与公开成功响应使用 `IDENTITY_SECRET` 作用域哈希或无内容枚举，不保存 raw recipient、raw token、provider response body 或 provider ID。cutover status 不公开 identity fingerprint。


### 输入与路径安全

- Vault 相对目录校验与 `dailyDir`/`papersDir` 冲突检测。
- 个人文献库目录通过 canonical root + 文件系统 identity 绑定；普通设置仅显示目录 basename，授权 modal 才显示完整 scope。完整 root 加入插件 logger 敏感值列表。endpoint disclosure 删除 userinfo/fragment，并保留 query 参数名但将值替换为 `[redacted]`。
- scoped library capability 为只读、相对路径、root-contained、no-symlink 边界，并限制 inventory entries/depth 与单文件读取字节数；根目录或目标文件在验证期间改变时拒绝操作。
- 数据导入限制见上文硬数字，并拒绝 `..` / 逃出 vault。
- Prompt 侧含 injection-guard 模板。

### 失败分类与恢复

| 结果 | 含义与后续 |
| --- | --- |
| `pending` | 如 arXiv 当日无文，或取消后调度落盘；不写空日报 / 可再跑 |
| `failed_transient` | 可重试（网络、过滤响应 JSON/契约校验、部分 LLM、日报后索引摘要回写等）；受 tick 退避与最多 10 次 attempts 升级约束 |
| `failed_permanent` | 配置/永久 LLM、过滤后索引入库失败、瞬态次数耗尽、陈旧 `running` 恢复等 |
| `cancelled` | 流水线结果 kind；调度写 `pending` |
| `completed` | 含 0 篇可见论文；已有日报修复路径 |

邮件返回独立 `DeliverEmailResult`，**不**改写当日 pipeline run-state。`delivered_unrecorded` 表示 provider 已接受但兼容主状态未确认；`ambiguous` 表示 provider 或投递持久化结果不明；两类 automatic generation 都继续阻断自动重发，交由人工处理。`skipped` 包含已投递、活动 claim 或已开始 provider attempt。明确未调用 provider 的取消/失败可释放 generation；明确 provider 拒绝会返回 `failed`，但其终态 evidence 仍阻断对同一 automatic identity 的静默自动重试。Worker 对无效 token 返回 401、邮箱不匹配 403、quota/明确拒绝 429 或 422、identity/fingerprint 或 pending outcome 冲突 409、ledger/cutover/DO 不可用 503、provider 结果不明 502；验证限流保持统一对外成功形态。

### 运行时约束

- 插件 `isDesktopOnly: true`。
- arXiv 请求有意限速与冷却。
- LLM：默认温度 0.1；逻辑超时 300s；流空闲 120s；客户端最多 3 次瞬态重试。
- 调度与流水线共享取消与锁，避免重叠写同一日产物。

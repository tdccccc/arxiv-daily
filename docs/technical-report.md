# Technical Report

## Scope and System Overview

arXiv Daily 是一个以研究主题过滤 arXiv 论文、生成 Markdown 日报/详报，并支持定时调度与邮件投递的系统。当前主实现是 TypeScript monorepo（版本 `0.4.1`），围绕 host-neutral 业务核 `@arxiv-daily/core` 组织：

- **Obsidian 插件**（`plugin/`，清单 id `arxiv-daily`，`isDesktopOnly: true`）提供设置页、命令、状态栏与 Dashboard 视图，并在进程内运行调度器；
- **Node CLI**（`apps/cli/`，npm 包名 `arxiv-daily`）读取固定路径 `config.toml`，提供单次运行、crontab 安装、数据导入导出与邮件命令；
- **Cloudflare Worker 邮件中继**（`services/email-relay/`，不在根 npm workspaces，包版本独立）实现官方代发（Beta）的验证与投递；
- 根目录 `arxiv_daily.py` 为兼容壳：若 `apps/cli/dist/arxiv-daily-cli.cjs` 存在则转发 argv（无参时默认 `run --today`）；
- `extensions/vscode-arxiv-daily` 为独立配套扩展（未进 workspaces），提供 vault 探测与 webview 骨架，但当前仍调用已移除的 CLI 参数/子命令（`--config` / `--vault-root`、`run-pending`、`summarize`），与 0.4.x 固定路径 CLI **不兼容**，不能当作可用生产包装。

业务核通过 `HostAdapters`（HTTP、存储、密钥、进度、资源打开、标记解析）与具体运行时解耦。插件与 CLI 分别注入 Obsidian / Node 适配层后，共享 `ArxivPipeline`、状态/索引与投递编排；调度语义在两端不同（见下文）。

## Runtime and Technology Stack

| 层次 | 技术 | 在本项目中的职责 |
| --- | --- | --- |
| 语言/运行时 | TypeScript（ES2022）、Node.js `>=20.11.0` | 全仓源码与 CLI/构建；CI 使用 Node 22.17.0 |
| 包管理 | npm workspaces | 工作区：`packages/*`、`apps/*`、`plugin`；`services/email-relay` 独立 |
| 业务核 | `@arxiv-daily/core` | 流水线、LLM、调度、索引、设置、邮件编排；生产第三方依赖仅 `pako` |
| Node 宿主 | `@arxiv-daily/node-runtime` | `fetch`、文件系统、`EnvSecretProvider`、linkedom 等 |
| Obsidian 宿主 | Obsidian Plugin API + `obsidian` 类型 | Vault 读写、`requestUrl`、设置持久化、UI |
| 构建 | esbuild | 插件打 `main.js`；CLI 打 `dist/arxiv-daily-cli.cjs`（并复制到 `plugin/arxiv-daily-cli.cjs`） |
| 测试/类型 | Vitest、`tsc --noEmit`、ESLint（含 `eslint-plugin-obsidianmd`） | 工作区测试；边界检查脚本 |
| 邮件 | Resend HTTP API；Cloudflare Workers + KV + Durable Objects | 自发送与官方代发 |
| 外部数据 | arXiv HTML/Atom/源码页 | 论文发现与正文抽取 |
| LLM | OpenAI 兼容 Chat Completions（流式） | 过滤、详报选择、日总结/详报摘要 |

## Frameworks and Responsibilities

### `@arxiv-daily/core`

宿主无关的业务实现与契约：

- **适配器契约**（`packages/core/src/core/adapters.ts`）：`HttpClient`、`StorageAdapter`（含可选 `writeTextAtomic` / `appendText` / 二进制 / `list`）、`SecretProvider`、`ProgressReporter`、`ResourceOpener`、`MarkupParser`，聚合为 `HostAdapters`。进度阶段枚举含 `fetch-metadata`、`fetch-recent`、`enrich-abstract`、`filter`、`fetch-content`、`summarize-daily`、`summarize-detail`、`write-detail`。
- **流水线**（`ArxivPipeline`）：按日生成日报与可选详报；结果 kind 为 `completed | pending | cancelled | failed_transient | failed_permanent`。
- **LLM**（`LlmClient`）：流式调用、客户端内瞬态重试、thinking/reasoning 参数、模型列表探测、敏感信息脱敏。
- **调度**（`SchedulerService` / `SchedulerDriver`）：本地时间窗、回看窗口、周末跳过、运行锁、状态机与历史记录；公开 API 主要在 `SchedulerService`（`tick`、`tickToday`、`tickTodayScheduled`、`runForDateNow`、`forceRunForDate`、`retryFailedInLookback`、`runAllPending` 等）。
- **数据面**：`PaperIndexStore`、`StateStore`、`RunHistoryStore`、过滤/日总结 checkpoint、邮件 `delivery-state`。
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

插件设置页的 **Personal library** 组可通过 Obsidian 桌面宿主的 Electron 目录选择器连接一个 Vault 内或 Vault 外目录。`selectLibraryRoot` 调用窄入口 `openObsidianLibrarySource`，后者只桥接 `@arxiv-daily/node-runtime/scoped-library-source`；成功打开后保存 canonical root 与文件系统 identity。用户可在不授予模型处理许可的情况下运行本地只读 inventory preview；Node scoped source 对 inventory 中的非符号链接条目使用 `lstat` 分类，文件条目带有 size 与 mtime 观测，符号链接不被跟随。设置页将 PDF 标为 eligible，将其他文件类型、符号链接和特殊条目标为 ignored，并在 modal 中对每组最多显示 100 条路径。

模型处理许可是独立的插件本地状态。授权 modal 展示 canonical folder、eligible extensions、处理深度和脱敏后的 Chat Completions endpoint；确认时必须提交与展示内容相同的 fingerprint。fingerprint 绑定 root path、文件系统 identity、扩展名集合、处理深度和 endpoint，不绑定 API key、模型或 thinking 参数。目录、identity、endpoint、扩展名或处理深度变化会使授权失效；Revoke 只移除模型处理许可，不断开本地只读目录。

连接目录后，用户还可显式执行 **Scan library** 或 **Reload catalog**，两者不要求模型处理许可。扫描只从 eligible PDF 的逻辑文件名识别现代 arXiv ID，不读取 PDF bytes；Core reconciliation 以 path、size、mtime 和识别策略构造 observation fingerprint，复用未变文件，将 unresolved、unrelated 与暂时 failed 文件隔离，并把同一论文的多个 PDF 汇入一个 paper record。需要补齐的 canonical ID 通过现有 `ArxivFetcher.fetchMetadataByIds` 请求 arXiv Atom API；Atom metadata 保留完整作者数组、标题、日期、分类与摘要，未发送绝对/相对文件路径或 PDF 内容。

`PersonalLibraryCatalogStore` 将严格版本化 catalog 保存到现有 Vault index root 下的 `personal-library-catalog.json`。store 使用宿主 `writeTextAtomic`、独立 `.backup` generation、同 adapter/path 的进程内 mutation queue、严格 decoder、scope/identification compatibility 检查与语义 revision；有效 backup recovery 会修复 primary。完整 inventory 会移除缺失文件贡献，truncated inventory 保留未观察到的旧记录。插件将扫描注册为 `personal-library-scan` operation；重复扫描被拒绝，卸载、目录变化和输出路径变化会请求取消；进入最终不可中断的 atomic promotion 后采用 commit-wins 语义。设置页只显示 revision 与 ready/papers/unresolved/unrelated/failed/truncated counts。

选中文献库时，插件还会按 scope 与识别策略 identity 从 Vault index root 下独立加载可替换的 direction proposal 和研究者确认的 interest profile；缺失 proposal 表示尚无提议，缺失 profile 表示严格空 profile，一个文档损坏不会清空另一个。两者使用独立 primary/backup、原子整文档写入、语义 revision 与 CAS；确认操作按 profile-first 协调，避免在权威 profile 写入失败时先消费 proposal candidate。插件内部的生成入口只接受当前有效的模型处理许可与已加载 catalog，使用独立 `LlmClient` 调用有界 Core proposer，并在提交前复核连接、输出位置、许可和精确 catalog evidence fingerprint；目录、输出位置、有效 endpoint 或撤销许可会定向取消生成。

设置页的 **Review directions** 和命令 `review-personal-library-directions` 打开同一个 Proposed/Confirmed modal。Proposed 候选可检查和修改名称、描述、discovery cues 与 1–5 篇代表论文，也可显式合并、移除或确认；只有确认操作会把候选转入研究者权威 profile。Confirmed 方向保留 active/disabled/merged 状态、代表论文与 evidence diagnostics，并通过显式操作编辑、启停、合并或按 restrict/cascade 删除。modal 将模型和 catalog 内容按纯文本渲染；本地查看与 review 不要求仍持有模型处理许可，重新生成则要求当前许可。active confirmed direction 的 eligibility projection 已在该 modal 中用于显示 stale/missing evidence，但尚未接入 `ArxivPipeline`、日报、详报、Paper Index、Dashboard 论文查询或邮件。

生成路径**不**缓存单一 `ArxivPipeline`：调度与命令经 `buildPipeline()` / `buildManualFetch()` 按当前 settings 重建依赖。`HostAdapters` 仅在 onload 构建；输出路径变更时由 `reloadStateStoreForOutputPaths` 替换 `StateStore`/`RunHistoryStore`；调度启停经 `restartScheduler` / `setScheduleEnabled`。

### Node CLI

`runCli`（`apps/cli/src/main.ts`）解析子命令；除 `init` / `update` / `help` 外读取固定路径 `config.toml`，经 `buildCliRuntime` 组装 pipeline、`SchedulerService`、`manualFetch`。CLI **不**调用 `scheduler.start()` 做 tick 循环；`run` 使用 `runForDateNow`（锁、run-state/history、`onDailyCompleted` 邮件）。

### Email Relay Worker

`services/email-relay` 使用 Wrangler 部署；默认 `PUBLIC_BASE_URL = https://mail.arxiv-daily.top`。处理健康检查、验证起始/完成与 `/v1/deliver`；投递优先经 Durable Object `DeliverGate` 串行化同一幂等键，并配合 KV 幂等与 UTC 日配额。

## Architecture and Module Boundaries

依赖方向（`scripts/check-boundaries.mjs` 强制；**不**扫描 email-relay / extensions）：

```text
packages/core          → 仅 pako（+ 自身）
packages/node-runtime  → @arxiv-daily/core
apps/cli               → @arxiv-daily/core, @arxiv-daily/node-runtime
plugin                 → @arxiv-daily/core, obsidian；仅额外允许 @arxiv-daily/node-runtime/scoped-library-source
services/email-relay   → 独立（不在 npm workspaces）
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
| Worker | `services/email-relay/src/index.ts` | `fetch` 路由 `/health`、`/v1/verify/*`、`/v1/deliver` |

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
3. **LLM 过滤**：`filterPapers`；可写 `filter-checkpoints`；永久/瞬态 LLM 错误映射为对应失败 kind。  
4. **过滤结果为空**：直接 `completed`（`papersWritten: 0`，空 digest），**不写** Paper Index。  
5. **索引入库**：`PaperIndexStore` upsert；失败 → **`failed_permanent`**。`ignored` 不进入后续可见集合；若全部 ignored → `completed`（0 篇，空 digest）。  
6. **正文获取**：并发度 6；经 `SourceAdapter.fetchContent`；失败降级为错误占位文本，不中断整日。  
7. **详报选择**：`selectDetailPapers`（阈值与 soft limit 来自 `detailSelection`）；磁盘上已存在且 `classifyPaperNote` 为 `verified_detail` 的详报不占 soft quota。  
8. **详报写作**：对选中且有全文的论文 `summarizePaperDetail` + `writePaperDetail`；**仅当文件真实存在**后回写 `paperPath` / `isDetail`。  
9. **日报摘要**：`summarizeDaily`（可走 summary checkpoint）→ `writeDaily` → 清理 checkpoint → 更新索引中的日报路径、结构化 summary，以及日报中严格 marker 可解析时的 occurrence-level discovery provenance；此阶段索引回写失败 → **`failed_transient`**。已有日报修复与 Dashboard history sync 可从绑定 report date、arXiv ID 的 canonical marker 重建 provenance；marker 结构无效时保留既有 projection，不做破坏性覆盖。
10. **返回**：`completed` + `DailyDigest`（供邮件自动发送）。

取消通过 `AbortSignal` / `isCancellationError` 映射为流水线 `cancelled`。调度落盘时将 `cancelled`（及部分中断）写为 run-state **`pending`**（可再跑），`RunStatus` 无独立 `cancelled`。

可选依赖：`PipelineDeps.sourceAdapter`；checkpoint 经 `checkpointStores.filter/summary`（兼容旧字段 `checkpointStore`）。

### 调度流

`SchedulerService` 委托 `SchedulerDriver`：

- 启用后按 `schedule.tickIntervalMin`（默认 20 分钟）`setInterval` 触发；
- 仅在 `runAtLocal`–`runUntilLocal` 本地时间窗内工作（多日 tick 时时间窗主要约束“今天”）；
- 回看 `LOOKBACK_DAYS = 5` 个日历日；配置时区下的周末跳过；
- `checkTickGate`：已完成 / 运行中 / 窗外 / 瞬态失败退避则跳过；
- `RunLock` 保证同日互斥；`StateStore` 记录 `pending|running|completed|failed_*|skipped`；
- 瞬态失败在 `setFailed` 时若 `attempts >= MAX_TRANSIENT_ATTEMPTS`（**10**）则升级为 `failed_permanent`；
- 超过 `STALE_RUNNING_RECOVERY_MS`（1 小时）的 `running` 在启动/恢复时标为 **`failed_permanent`**（错误文案：`recovered stale running state after startup`）；
- 完成时调用可选 `onDailyCompleted`（接邮件）；回调失败不得改写 run-state。

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

`deliverDailyEmailIfEnabled`：

1. 校验 `enabled`（自动）或 `force`（测试）与凭证；结果 kind：`delivered | skipped | disabled | failed`（provider/state 失败不抛到调度层）。  
2. 读取 `delivery-state.json`；同日同收件人任一通道已 `delivered` 则跳过（跨 self/hosted）。  
3. 渲染 subject/html/text。  
4. **self**：settings 中的 Resend API key → Resend（当前 CLI/插件调用 `resolveResendApiKey(email, {})`，**不读取**进程环境变量）；From 空时用 `onboarding@resend.dev`；客户端侧可有限次重试。  
5. **hosted**：Bearer `hostedToken` → `{hostedBaseUrl 或默认}/v1/deliver`；正式日幂等键为 `date|to`；force 测试为 `test|date|to|iso`，避免与正式日撞键。  
6. 非 force 成功则落盘 delivered；force 不写“已送达”日历态。

`OFFICIAL_DELIVERY_AVAILABLE = true` 打开客户端 hosted 路径；请求成功仍依赖已部署 Worker 与有效 token。默认 base：`https://mail.arxiv-daily.top`。

**Worker：**

- 路由：`GET /|/health`、`POST /v1/verify/start`、`GET /v1/verify`、`POST /v1/deliver`。  
- 验证限流：邮箱 3 次/小时、IP 10 次/小时；超限仍返回“已发送”形态，降低枚举。  
- 设备 token：验证页展示；KV 存 `TOKEN_SECRET` 加盐哈希，TTL 约 1 年。  
- 投递：`to` 必须等于绑定邮箱；UTC 日配额默认 `DAILY_QUOTA=5`；401/403/429/409/502 等语义。  
- 幂等：KV 预占 + `DeliverGate` DO 按键串行；绑定缺失时回退无 DO 的 `runDeliver`（竞态窗口更大）。  
- 出站仅 Worker 持有 `RESEND_API_KEY`。

### Dashboard 与命令

- Dashboard（`plugin/src/dashboard/view.ts`）：`PaperIndexStore`、run-state、vault 文件同步（`syncDashboardHistory` / `queryDashboard`）；日历、检索、分页、状态操作；运行入口同样走 `scheduler.runForDateNow` / force 等路径。  
- 命令（`plugin/src/commands.ts`）：今日运行、回看 pending、重试失败、指定日/强制日、清 run-state、手动 arXiv id 摘要、诊断等；运行前 `validateFilterConfig` / `validateLlmConfig`。

## Data and State

默认输出布局（相对 vault / `vaultRoot`）：

| 路径 | 内容 |
| --- | --- |
| `arxiv-daily/daily/YYYY-MM-DD.md` | 日报（权威提交物） |
| `arxiv-daily/papers/<externalId>.md` | 详报（路径 stem 为 externalId，非 paperKey） |
| `arxiv-daily/.index/papers.json` | 论文索引（schema v5；key 为 `paperKey` 如 `arxiv:…`，可按日报路径保存 occurrence-level discovery provenance） |
| `arxiv-daily/.index/run-state.json` | 按日运行状态（原子写 + `.bak`） |
| `arxiv-daily/.index/run-history.jsonl` | 运行历史（可轮转） |
| `arxiv-daily/.index/delivery-state.json` | 邮件投递记录 |
| `arxiv-daily/.index/filter-checkpoints/` | 过滤 checkpoint |
| `arxiv-daily/.index/daily-summary-checkpoints/` | 日总结 checkpoint |
| 兼容 `arxiv-daily/index/papers.json` | 旧索引路径可读 |

**设置持久化**

- 插件：Obsidian `loadData`/`saveData` 存 sibling `settings` 与可选 `libraryConnection`；后者包含 canonical root、文件系统 identity、eligible extensions、处理深度及可选授权 fingerprint/time，不包含 API key。设置与文献库状态的整份持久化经同一进程内 mutation queue 串行；选择、授权或撤销保存失败时恢复内存状态。旧版嵌在 data 里的 `runState` 可迁移进 `run-state.json`。
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

`OperationRegistry` 跟踪 `daily-run`、`detail-summary`、`pdf-download`；`RunCancellationService` 与 scheduler 协作。插件卸载 `cancelAll`；CLI 安装信号处理器取消活动操作。

### 原子写与状态一致性

`StateStore` / 索引 / checkpoint / delivery-state 普遍使用临时文件 + rename 或 `writeTextAtomic`；同路径 mutation 队列避免并发写撕裂。日报 Markdown 存在即视为该日已提交的权威信号。

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

### 环境变量与 secrets

- `resolveResendApiKey(email, env)` **支持** env `ARXIV_DAILY_RESEND_API_KEY` 优先于 `email.apiKey`，但当前 CLI/插件调用均传入空 env，**实际只读 settings 中的 apiKey**。  
- Worker secrets：`RESEND_API_KEY`、`TOKEN_SECRET`（`wrangler secret put`）。  
- Worker vars：`PUBLIC_BASE_URL`、`FROM_EMAIL`、`FROM_NAME`、`DAILY_QUOTA`（默认 `"5"`）。

### 清单与版本

- 根与 `plugin/manifest.json`：插件 id、版本、`minAppVersion` `1.4.0`、`isDesktopOnly: true`  
- `versions.json` / `plugin/versions.json`：插件版本到 Obsidian 最低版本映射  
- 工作区版本由 `scripts/sync-release-version.mjs` / `check-release-version.mjs` 对齐；email-relay 版本独立

## Build, Test, Deployment, and Operations

### 本地脚本（根 `package.json`）

- `npm run typecheck` / `test` / `build`：工作区透传（**不含** email-relay）  
- `npm run lint`：插件 ESLint  
- `npm run check:boundaries`：分层边界  
- `npm run smoke:build`：CLI/插件产物冒烟  
- `npm run cli`：运行已构建 CLI  
- 发布工具：`sync:release-version`、`check:release-version`、`test:release-tools`

### 构建产物

- 插件：`plugin/main.js`、`styles.css`、`manifest.json`（esbuild 外置 `obsidian`/`electron`）  
- CLI：`apps/cli/dist/arxiv-daily-cli.cjs`，构建时复制到 `plugin/arxiv-daily-cli.cjs`；`prepack` 先 build  
- smoke 检查含 help 退出码、坏配置、pako notice、插件包不泄漏 workspace 解析符号等

### CI / 发布

- **Lint and typecheck**（`lint.yml`）：push/PR 上 `npm ci`、lint、typecheck（**不**跑 test / boundaries / smoke）。  
- **Release Obsidian plugin**（`release.yml`）：推送稳定 SemVer tag → 校验 tag/SHA/`docs/releases/<tag>.md`、拒绝覆盖已有 release → boundaries / lint / typecheck / test / build / smoke → 对插件三件套做 build-provenance attestation → `gh release create`。  
- **Publish CLI to npm**（`publish-cli.yml`）：插件 release **成功后**自动，或 `workflow_dispatch` 指定已有 tag → 校验 GH release 与 npm 版本未覆盖 → trusted publishing `npm publish --workspace apps/cli`（包名 `arxiv-daily`）。  
- **email-relay**：目录内 `wrangler deploy` / 本地 `npm test`；**不在** GitHub Actions 默认路径。

### 验证面

core / plugin / CLI 工作区有大量 Vitest。email-relay 仅有本地 crypto/KV 幂等测试。测试用于约束行为；生产路径以源码注册与打包入口为准。

## Security and Failure Behavior

### 密钥与脱敏

- LLM / Resend / hosted token 进入 logger 敏感值列表；CLI 对 stdout/stderr 做 `redactText`。  
- 插件密钥存于 Obsidian 数据（`ObsidianSettingsSecretProvider`）。  
- Worker 不向客户端暴露 Resend key；设备 token 经验证页展示，KV 仅存哈希。

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
| `failed_transient` | 可重试（网络、部分 LLM、日报后索引摘要回写等）；受 tick 退避与最多 10 次 attempts 升级约束 |
| `failed_permanent` | 配置/永久 LLM、过滤后索引入库失败、瞬态次数耗尽、陈旧 `running` 恢复等 |
| `cancelled` | 流水线结果 kind；调度写 `pending` |
| `completed` | 含 0 篇可见论文；已有日报修复路径 |

邮件失败返回 `DeliverEmailResult` 并可记 `delivery-state`，**不**改写当日 pipeline run-state。Worker 对无效 token 401、邮箱不匹配 403、配额 429、并发幂等 409；验证限流对外成功形态。

### 运行时约束

- 插件 `isDesktopOnly: true`。  
- arXiv 请求有意限速与冷却。  
- LLM：默认温度 0.1；逻辑超时 300s；流空闲 120s；客户端最多 3 次瞬态重试。  
- 调度与流水线共享取消与锁，避免重叠写同一日产物。

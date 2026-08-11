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
| Obsidian 宿主 | Obsidian Plugin API + `obsidian` 类型 | Vault 读写、`requestUrl`、设置持久化、UI |
| VS Code companion | VS Code Extension API `^1.90.0` | `workspace.fs` Dashboard 与 `ProcessExecution` CLI 任务 |
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

`ArxivDailyPlugin`（`plugin/main.ts`）在 `onload` 中：加载设置与状态 → 构建 host → 状态栏进度 → `SchedulerService` → 注册设置页/Dashboard/命令；`schedule.enabled` 时启动进程内调度。卸载时停止调度并 `operations.cancelAll`。

生成路径**不**缓存单一 `ArxivPipeline`：调度与命令经 `buildPipeline()` / `buildManualFetch()` 按当前 settings 重建依赖。`HostAdapters` 仅在 onload 构建。插件设置变更经 `SettingsChangeService` 串行提交；输出目录候选会预加载新的 `StateStore` / `RunHistoryStore`，再由 Scheduler 通过一次同步 pair replacement 联合安装，调度启停与定时器更新在设置持久化和 live commit 后执行。

### Node CLI

`runCli`（`apps/cli/src/main.ts`）解析子命令；除 `init` / `update` / `help` 外读取固定路径 `config.toml`，经 `buildCliRuntime` 组装 pipeline、`SchedulerService`、`manualFetch`。CLI **不**调用 `scheduler.start()` 做 tick 循环；`run` 使用 `runForDateNow`（锁、run-state/history、`onDailyCompleted` 邮件）。

### Email Relay Worker

`services/email-relay` 使用 Wrangler 部署；默认 `PUBLIC_BASE_URL = https://mail.arxiv-daily.top`。处理健康检查、验证起始/完成与 `/v1/deliver`；投递优先经 Durable Object `DeliverGate` 串行化同一幂等键，并配合 KV 幂等与 UTC 日配额。

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
plugin                 → @arxiv-daily/core, obsidian
services/email-relay   → 独立（不在 npm workspaces）
extensions/vscode-arxiv-daily → 独立 CommonJS 扩展（不在 npm workspaces）
```

禁止的旧路径（如 `plugin/src/pipeline`、`plugin/src/hosts/node`）不得存在。core 不得出现 `process`/`Buffer` 或 Node builtin import；plugin 源码不得 import Node builtin；禁止深层 `@arxiv-daily/*/…` 导入。

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
| `arxiv-daily/.index/papers.json` | 论文索引（schema v4；key 为 `paperKey` 如 `arxiv:…`；同目录 `.bak` 为恢复副本） |
| `arxiv-daily/.index/run-state.json` | 按日运行状态（原子写 + `.bak`） |
| `arxiv-daily/.index/run-history.jsonl` | 运行历史（可轮转） |
| `arxiv-daily/.index/delivery-state.json` | 邮件投递记录 |
| `arxiv-daily/.index/filter-checkpoints/` | 过滤 checkpoint |
| `arxiv-daily/.index/daily-summary-checkpoints/` | 日总结 checkpoint |
| 兼容 `arxiv-daily/index/papers.json` | 旧索引路径可读 |

**设置持久化**

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

`OperationRegistry` 跟踪 `daily-run`、`detail-summary`、`pdf-download` 及插件的 `paper-index` / `paper-note` output lease；`RunCancellationService` 与 scheduler 协作。设置输出 transition 与 operation registry 双向互斥。插件卸载 `cancelAll`；CLI 安装信号处理器取消活动操作。

### Paper Index 持久化与 History Sync

`PaperIndexStore`（`packages/core/src/services/paper-index.ts`）按 primary `papers.json` → `.bak` → legacy `index/papers.json` 的顺序选择首个有效文档；任一路径的真实读取错误直接失败，候选文件存在但均无法解析时也不会构造空索引。读取兼容 schema 1–4，内存归一为 schema 4，后续保存写 schema 4。

索引写入先生成 `.tmp`，再把已验证的旧 primary 发布为 `.bak`，最后以 `.tmp → papers.json` rename 作为新内容的提交点。primary 缺失或损坏时不会用它覆盖有效 backup；提升失败时只尝试恢复提交前已验证的 primary、backup 或 legacy 内容。`PaperIndexStore` 的领域 mutation 在模块级、按 primary 路径共享的 Promise 队列中执行完整的读取、修改、校验和保存事务；该串行范围限于同一 JavaScript realm，不提供跨进程锁或 `fsync` 级掉电保证。

Dashboard 历史同步（`packages/core/src/dashboard/history-sync.ts`）先扫描日报和论文笔记，再在同一索引 mutation 中重读当前状态。日报证据可补建非详情索引投影；论文详情必须由无歧义、身份一致的受管笔记证明。重复或冲突的顶层 `arxiv_id` / `arxiv` 标量会保护涉及的全部论文身份，嵌套字段、数组和正文不参与身份判断。破坏性清理还会比较扫描基线与 mutation 开始时的当前投影；扫描期间被其他操作删除或修改的详情不会被陈旧候选复活或清理。

插件 diagnostics 复用 `PaperIndexStore.inspect()` 的文档选择结果。索引解析、读取及后续笔记探测失败在可复制报告、日志、DiagnosticsModal 和 Hub 中只暴露稳定分类（如 `paper_index_invalid`、`paper_index_unavailable`），不传播底层异常 message 或 cause。

### 原子写与状态一致性

`StateStore` 的普通启动读取可从损坏 primary 回退 `.bak`，但显式未知 schema，以及 schema 1/无 schema 记录中类型非法的 `error` 或 `papersWritten` 会 fail closed。mutation 使用按 run-state 路径共享的进程内队列，并通过 candidate 保存与权威 primary 精确回读确认提交；backup 不参与 mutation 的 authoritative confirmation。

插件输出路径重载会先构造并加载候选 `StateStore` / `RunHistoryStore`，再由 scheduler 的 active/pending guard 接受 store 替换，最后同步发布 plugin 引用；guard 拒绝时各消费者继续使用旧 store。该协调和 pending completion 都是单进程语义，没有 Plugin/CLI 跨进程锁。

Paper Index、checkpoint 与 delivery-state 使用各自的临时文件和 rename 或 `writeTextAtomic` 流程；相关同路径 mutation 队列在进程内串行化读改写。日报 Markdown 存在即视为该日已提交的权威信号。

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
- **Email relay verification**（`email-relay.yml`）：relay 或 hosted delivery contract、workflow、产品清单及 checker路径变更时，使用 relay 自身 lockfile执行 `npm ci`、typecheck、tests和 Wrangler `deploy --dry-run`；bundle写入 runner临时目录，不部署 Worker，也不读取生产凭据。
- **VS Code companion verification**（`vscode-companion.yml`）：companion、CLI command contract、workflow、产品清单及 checker路径变更时，使用 companion 自身 lockfile执行 build、tests、smoke，并把验证用 VSIX写入 runner临时目录；不发布扩展。
- 两个独立 workflow 都在 pull request 和相应路径推送到 `main` 时运行，action固定完整commit SHA，权限仅 `contents: read`，checkout不持久化凭据。

### 验证面

core / plugin / CLI 工作区有大量 Vitest。Core 流水线集成测试使用包含两个论文条目的代表性 recent 页面输入；arXiv parser 与 source adapter 专项测试继续读取完整的真实页面夹具。email-relay 仅有本地 crypto/KV 幂等测试。测试用于约束行为；生产路径以源码注册与打包入口为准。

## Security and Failure Behavior

### 密钥与脱敏

- LLM / Resend / hosted token 进入 logger 敏感值列表；CLI 对 stdout/stderr 做 `redactText`。  
- 插件密钥存于 Obsidian 数据（`ObsidianSettingsSecretProvider`），provider 写入经 settings candidate transaction 持久化。设置页只用 `Configured` 表示已有密钥；stored LLM/Resend/hosted secret 不写入输入 DOM，用户通过 Replace/Cancel/Clear 明确变更。
- Worker 不向客户端暴露 Resend key；设备 token 经验证页展示，KV 仅存哈希。

### 输入与路径安全

- Vault 相对目录校验与 `dailyDir`/`papersDir` 冲突检测。  
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

邮件失败返回 `DeliverEmailResult` 并可记 `delivery-state`，**不**改写当日 pipeline run-state。Worker 对无效 token 401、邮箱不匹配 403、配额 429、并发幂等 409；验证限流对外成功形态。

### 运行时约束

- 插件 `isDesktopOnly: true`。  
- arXiv 请求有意限速与冷却。  
- LLM：默认温度 0.1；逻辑超时 300s；流空闲 120s；客户端最多 3 次瞬态重试。  
- 调度与流水线共享取消与锁，避免重叠写同一日产物。

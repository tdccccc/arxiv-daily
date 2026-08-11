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

- **适配器契约**（`packages/core/src/core/adapters.ts`）：`HttpClient`、`StorageAdapter`（含可选私有原子替换/恢复、跨进程 exclusive create、descriptor-backed namespace guard、append、二进制与目录枚举能力）、`SecretProvider`、`ProgressReporter`、`ResourceOpener`、`MarkupParser`，聚合为 `HostAdapters`。进度阶段枚举含 `fetch-metadata`、`fetch-recent`、`enrich-abstract`、`filter`、`fetch-content`、`summarize-daily`、`summarize-detail`、`write-detail`。
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

生成路径**不**缓存单一 `ArxivPipeline`：调度与命令经 `buildPipeline()` / `buildManualFetch()` 按当前 settings 重建依赖。`HostAdapters` 仅在 onload 构建；输出路径变更时由 `reloadStateStoreForOutputPaths` 替换 `StateStore`/`RunHistoryStore`；调度启停经 `restartScheduler` / `setScheduleEnabled`。

### Node CLI

`runCli`（`apps/cli/src/main.ts`）解析子命令；除 `init` / `update` / `help` 外读取固定路径 `config.toml`，经 `buildCliRuntime` 组装 pipeline、`SchedulerService`、`manualFetch`。CLI **不**调用 `scheduler.start()` 做 tick 循环；`run` 使用 `runForDateNow`（锁、run-state/history、`onDailyCompleted` 邮件）。

### Email Relay Worker

`services/email-relay` 是独立 Wrangler Worker；仓库配置的默认 `PUBLIC_BASE_URL` 为 `https://mail.arxiv-daily.top`。它处理健康检查、验证起始/完成、`/v1/deliver` 与 operator-only cutover staging。认证后按设备路由到唯一 `DeliverGate` Durable Object，幂等 ledger 与 UTC 日配额在该对象的存储事务内更新；`DELIVER_GATE` 或 v2 cutover proof 不可用时自动投递 fail closed，不存在无 DO fallback。

## Architecture and Module Boundaries

依赖方向（`scripts/check-boundaries.mjs` 强制；**不**扫描 email-relay / extensions）：

```text
packages/core          → 仅 pako（+ 自身）
packages/node-runtime  → @arxiv-daily/core
apps/cli               → @arxiv-daily/core, @arxiv-daily/node-runtime
plugin                 → @arxiv-daily/core, obsidian
services/email-relay   → 独立（不在 npm workspaces）
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
| Worker | `services/email-relay/src/index.ts` | `fetch` 路由 `/health`、`/v1/verify/*`、`/v1/deliver` 与受 operator bearer 保护的 `/internal/delivery-v2/cutover` |

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

`deliverDailyEmailIfEnabled` 先校验自动开关或显式测试请求的凭证，再渲染 subject/html/text。公开结果为 `delivered | delivered_unrecorded | ambiguous | skipped | disabled | failed`，reason 使用固定、无 PII 的枚举；provider/state 失败不会抛到调度层，也不会改写 pipeline run-state。

**自动投递：**

1. 客户端从 `date + 标准化收件人` 计算 `arxiv-daily:auto:<sha256>`；self 模式直接把该稳定 key 传给 Resend，hosted 模式把它作为 auto 类型标记传给 relay。key 不含明文收件人。
2. 发送前严格读取 `delivery-state.json`：missing 可开始；corrupt/unreadable 直接失败。宿主还必须同时提供目录枚举、系统级 exclusive create 和 descriptor-backed namespace guard，否则返回 `delivery_storage_unsupported`。
3. `delivery-state.json.claims/` 保存不可变 generation 的 claim、decision、result sidecar。逻辑 claim stem 和 sidecar 只保存哈希 recipient identity；同一日期/收件人的并发调用通过 exclusive create 收敛到单一 generation。
4. provider 调用前先持久化 `provider_attempt_started`，再同步复核仍由同一 descriptor 锚定的目录承载 claim；该复核与 `HttpClient.request` 之间没有异步间隔。只有可证明 provider 尚未调用的取消或恢复才能释放 generation。
5. self Resend 对 408/409/5xx 和宿主明确标记为可重试的 transport failure 做有界重试，所有物理尝试复用同一个 provider key。HTTP 400/401/403/404/422/429 是明确拒绝；其他 HTTP、transport failure 和无有效 acceptance marker 的 2xx 均按结果不明处理。
6. provider 接受后写 `delivered` result；若最终主状态重建失败，返回 `delivered_unrecorded`，已存在的 attempt/result sidecar 仍继续阻断自动重发。provider 调用后的 ambiguous、明确拒绝或结果落盘不确定同样保留阻断；系统不自动重试这些 generation。

`delivery-state.json` 保持 schema v1，使旧 reader 仍能读取；claim、attempt、ambiguous 等阻断态投影为 v1 `status: "delivered"`，并用 `deliveryPhase` 提供新客户端精确信息。为兼容旧 reader，主文件保留明文 recipient；Node 与受支持的 Obsidian 桌面文件系统将主文件和临时/备份产物强制为 `0600`，读取既有宽权限主文件时也收紧为 `0600`。Linux 宿主使用 `O_CREAT | O_EXCL | O_WRONLY | O_NOFOLLOW`、`/proc/self/fd` descriptor 锚定和原子 rename；无法提供这些能力的宿主对自动投递 fail closed。跨文件系统 rename 不降级为 copy。

显式 `force`/邮件测试不创建自动 claim，也不改写 automatic delivery state；每次生成独立 `arxiv-daily:test:<random>` key，因此不会占用正式日报 identity。self 模式的 API key 当前来自 settings/config；From 为空时使用 `onboarding@resend.dev`。hosted 模式使用 Bearer `hostedToken` 调用默认 `https://mail.arxiv-daily.top/v1/deliver`，客户端只接受精确的 `{ "ok": true }` 成功响应。`OFFICIAL_DELIVERY_AVAILABLE = true` 仅表示客户端路径开启，不证明外部 Worker 已部署或可用。

**Worker：**

- 路由：`GET /|/health`、`POST /v1/verify/start`、`GET /v1/verify`、`POST /v1/deliver`，以及受 `DELIVERY_V2_CUTOVER_TOKEN` bearer 保护的 `POST /internal/delivery-v2/cutover`。
- 验证限流仍是 KV best-effort counter：邮箱 3 次/小时、IP 10 次/小时；超限返回统一“已发送”形态。验证完成后 KV 以 secret-scoped token hash 为 key 保存绑定邮箱与 v2 创建时间，设备记录 TTL 约一年；验证页只展示 device token，不回显收件地址。
- `/v1/deliver` 先认证 device token，检查 `to` 与绑定邮箱一致，再按 secret-scoped device identity 路由到单一 `DeliverGate`。`DELIVER_GATE` 缺失时返回 503，不执行 provider。
- auto ledger identity 由服务端从 device identity、recipient identity 和 date 推导；客户端 auto key 的 hash 不参与 ledger/provider identity。test identity 另含每次测试的随机 logical key。relay provider key 只含有界哈希，不含明文收件人。
- Durable Object ledger 状态为 `reserved | attempted | done | rejected`。请求 fingerprint 防止同一 identity 绑定不同内容；ledger 与按 auto/test、UTC 日期划分的 quota 在同一 storage transaction 中预占。`attempted`、`done` 和 `rejected` 重放都不再次调用 provider；测试终态保留 30 天后由 alarm 清理，automatic ledger 不做该清理。
- Resend 明确拒绝映射为 relay 422/429 终态并回退 quota；transport、非白名单 HTTP、无效成功体，或 provider 接受后 completion storage 失败，均保留 `attempted` 阻断并返回 ambiguous。成功 ledger 与响应只保存 `{ "ok": true }`；provider response body 和 provider ID 验证后丢弃。
- automatic 流量还要求 Durable Object 中的 v2 cutover proof ready。proof 合并旧 KV 的正向 delivered/attempted evidence，并经过 visibility 与 pending-TTL barrier；KV 空扫描不证明旧投递不存在。pre-cutover/legacy device 无法证明 absence 时永久 fail closed，只有 proof ready 后签发的 v2 device identity 可在无 legacy evidence 时开始新的 automatic delivery。该协议按 quiesced single-version cutover 设计，不提供 mixed-version 并行或安全回滚保证。
- 出站 Resend key 仅由 Worker secret 提供。系统没有邮件 outbox 或后台重试 daemon；最终 provider 去重仍依赖 Resend/relay 正确执行稳定 idempotency key，因此不声明严格 distributed exactly-once。

### Dashboard 与命令

- Dashboard（`plugin/src/dashboard/view.ts`）：`PaperIndexStore`、run-state、vault 文件同步（`syncDashboardHistory` / `queryDashboard`）；日历、检索、分页、状态操作；运行入口同样走 `scheduler.runForDateNow` / force 等路径。  
- 命令（`plugin/src/commands.ts`）：今日运行、回看 pending、重试失败、指定日/强制日、清 run-state、手动 arXiv id 摘要、诊断等；运行前 `validateFilterConfig` / `validateLlmConfig`。

## Data and State

默认输出布局（相对 vault / `vaultRoot`）：

| 路径 | 内容 |
| --- | --- |
| `arxiv-daily/daily/YYYY-MM-DD.md` | 日报（权威提交物） |
| `arxiv-daily/papers/<externalId>.md` | 详报（路径 stem 为 externalId，非 paperKey） |
| `arxiv-daily/.index/papers.json` | 论文索引（schema v4；key 为 `paperKey` 如 `arxiv:…`） |
| `arxiv-daily/.index/run-state.json` | 按日运行状态（原子写 + `.bak`） |
| `arxiv-daily/.index/run-history.jsonl` | 运行历史（可轮转） |
| `arxiv-daily/.index/delivery-state.json` | v1-compatible 邮件阻断投影；为旧 reader 保留 recipient，受支持宿主按 `0600` 处理 |
| `arxiv-daily/.index/delivery-state.json.claims/` | 不可变 claim/decision/result generation；使用哈希 recipient identity |
| `arxiv-daily/.index/filter-checkpoints/` | 过滤 checkpoint |
| `arxiv-daily/.index/daily-summary-checkpoints/` | 日总结 checkpoint |
| 兼容 `arxiv-daily/index/papers.json` | 旧索引路径可读 |

**设置持久化**

- 插件：Obsidian `loadData`/`saveData` 存 `PluginSettings`；旧版嵌在 data 里的 `runState` 可迁移进 `run-state.json`。  
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

`OperationRegistry` 跟踪 `daily-run`、`detail-summary`、`pdf-download`；`RunCancellationService` 与 scheduler 协作。插件卸载 `cancelAll`；CLI 安装信号处理器取消活动操作。

### 原子写与状态一致性

`StateStore`、索引与 checkpoint 使用各自的原子替换或 mutation queue。邮件 automatic delivery 另以 immutable generation sidecar 作为 provider-attempt 的持久化依据，再重建 v1-compatible `delivery-state.json`；受支持的 Node/Obsidian Linux 文件系统用 descriptor-anchored exclusive create 和私有原子替换保证 claim 与主状态边界，能力不足时拒绝 automatic delivery。日报 Markdown 存在即视为该日已提交的权威信号。

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
- Worker secrets：`RESEND_API_KEY`、`TOKEN_SECRET`、operator-only `DELIVERY_V2_CUTOVER_TOKEN`（均通过 `wrangler secret put` 配置）。
- Worker vars：`PUBLIC_BASE_URL`、`FROM_EMAIL`、`FROM_NAME`、`DAILY_QUOTA`（默认 `"5"`）；`DELIVER_GATE` 是必需 Durable Object binding，`STORE` 用于验证设备、best-effort 验证限流与 cutover legacy evidence。

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
- **email-relay**：独立目录提供 `typecheck`、Vitest 与 Wrangler `deploy` 脚本；根 workspace/默认 GitHub Actions 不包含该服务。验证配置可使用 `wrangler deploy --dry-run`，仓库状态本身不证明 Worker 已外部部署。

### 验证面

core / node-runtime / plugin / CLI 工作区有 Vitest 覆盖；邮件定向面包含 Core claim/result/HTTP 分类、Node 与 Obsidian descriptor/权限/恢复，以及 CLI/Plugin consumer。email-relay 的独立 Vitest 覆盖认证路由、服务端 identity、Durable Object ledger/quota、provider 结果和 cutover proof。测试用于约束行为；生产路径以源码注册、构建入口和 Worker binding 为准。

## Security and Failure Behavior

### 密钥与脱敏

- LLM / Resend / hosted token 进入 logger 敏感值列表；CLI 对 stdout/stderr 做 `redactText`。  
- 插件密钥存于 Obsidian 数据（`ObsidianSettingsSecretProvider`）。  
- Worker 不向客户端暴露 Resend key；设备 token 只在验证完成页展示，KV key 使用 secret-scoped token hash。验证设备值仍保存规范化绑定邮箱以执行 recipient 校验；DO 名称、provider key、ledger、cutover/legacy sidecar 与公开成功响应使用哈希 identity 或无内容枚举，不保存 raw recipient、provider response body 或 provider ID。

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

邮件返回独立 `DeliverEmailResult`，**不**改写当日 pipeline run-state。`delivered_unrecorded` 表示 provider 已接受但兼容主状态未确认；`ambiguous` 表示 provider 或投递持久化结果不明；两类 automatic generation 都继续阻断自动重发，交由人工处理。`skipped` 包含已投递、活动 claim 或已开始 provider attempt。明确未调用 provider 的取消/失败可释放 generation；明确 provider 拒绝会返回 `failed`，但其终态 evidence 仍阻断对同一 automatic identity 的静默自动重试。Worker 对无效 token 返回 401、邮箱不匹配 403、quota/明确拒绝 429 或 422、identity/fingerprint 或 pending outcome 冲突 409、ledger/cutover/DO 不可用 503、provider 结果不明 502；验证限流保持统一对外成功形态。

### 运行时约束

- 插件 `isDesktopOnly: true`。  
- arXiv 请求有意限速与冷却。  
- LLM：默认温度 0.1；逻辑超时 300s；流空闲 120s；客户端最多 3 次瞬态重试。  
- 调度与流水线共享取消与锁，避免重叠写同一日产物。

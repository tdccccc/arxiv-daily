# Architecture, Code Organization, and UI Audit — 2026-08-10

> Branch: `review/architecture-code-ui-audit`
>
> Baseline: `75aaa88`
>
> Scope: current TypeScript monorepo, Obsidian plugin UI, Node CLI, shared Core, email relay, VS Code companion, build/test/CI organization

## Executive verdict

当前整体架构方向健康，不建议推倒重写。共享 `@arxiv-daily/core`、Plugin/CLI 两个 Host、独立 Node runtime、CLI 外部调度与 Plugin 进程内调度的边界均合理，现有 Obsidian UI 也已经较好遵循宿主主题变量和原生控件。

值得投入的工作集中在三类：

1. **先修正确性和持久化语义**：邮件投递幂等、非法过滤响应、Paper Index 恢复与并发 mutation、Scheduler 完成态提交。
2. **再补工程治理与结构边界**：PR 验证面、独立部署单元、失效 VS Code companion、Host 装配重复、Core 公开 API 和设置类型。
3. **前端做中等规模局部精修**：Dashboard 窄窗布局、操作层级、运行状态、设置行为统一、键盘/高对比度/日历状态。无需重做品牌、配色或改用前端框架。

| 维度 | 结论 | 建议强度 |
| --- | --- | --- |
| 总体架构 | 方向正确，少数事务边界和治理边界需要补强 | 定向改进 |
| Core 代码组织 | 正确性风险优先于文件拆分；随后提取应用阶段和仓储协议 | 中高 |
| Host / 工程组织 | 独立组件未进入统一验证，已有产品契约漂移 | 高 |
| 前端功能工程 | 设置双路径和 Dashboard 异步生命周期存在真实行为问题 | 高 |
| 前端美学 | 基础风格合适，信息密度和窄窗体验需要精修 | 中 |
| 全面视觉改版 | 收益不足，可能破坏 Obsidian 原生一致性 | 不建议 |

## Method and limitations

本次由多个只读审查 Agent 分别覆盖整体架构、Core、Host/工程、UI/视觉、可访问性与质量验证，再由主会话复核高优先级源码证据。技术报告用于定向，结论以当前源码、配置、测试和实际命令结果为准。

没有启动真实 Obsidian 宿主，因此主题最终计算样式、真实窗口断点、屏幕阅读器表现、Modal focus trap 和 declarative Settings 的宿主内部重绘时序仍需人工验证。未进行真实邮件投递或外部服务负载测试。

## Verification results

| 检查 | 结果 | 观察 |
| --- | --- | --- |
| `npm run check:boundaries` | 通过 | 当前四个受管工作区依赖方向通过 |
| `npm run lint` | 通过 | 0 errors、52 warnings；当前上限 60，其中 39 条 sentence-case、13 条 deprecated API |
| `npm run typecheck` | 通过 | Core、node-runtime、CLI、Plugin 均通过 |
| `npm test` | 失败 | Core 的 `pipeline.test.ts` 在默认约 4 GB V8 heap 下 OOM，继发 `ERR_IPC_CHANNEL_CLOSED` |
| 排除 `pipeline.test.ts` 的 Core 测试 | 通过 | 65 files、969 tests |
| `NODE_OPTIONS=--max-old-space-size=8192 npm test -- --maxWorkers=1` | 通过 | Core 1012、node-runtime 13、CLI 59、Plugin 288，共 1372 tests；峰值 RSS 约 4.94 GB |
| `npm run build` | 通过 | Plugin `main.js` 约 462.3 KiB；CLI bundle 约 800.3 KiB |
| email relay 单独测试 | 7/7 通过 | 不在根 workspace；当前根依赖环境下单独 typecheck 因缺少 `@cloudflare/workers-types` 失败 |
| VS Code companion 自检 | 通过 | 但测试固定了已移除 CLI 契约，不能证明当前集成可用 |

生产 TypeScript 约 33,025 行，测试 TypeScript 约 26,818 行。最大结构热点包括：

- `plugin/src/dashboard/view.ts`：2,443 行，主类约 2,232 行、93 个方法；
- `plugin/src/settings/tab.ts`：2,021 行，主类约 1,745 行；
- `apps/cli/src/init.ts`：1,279 行，`runStep` 约 512 行；
- `packages/core/src/pipeline/pipeline.ts`：758 行，`runForDateInner` 约 344 行；
- `plugin/styles.css`：1,582 行。

文件规模只作为职责混杂的线索，不单独构成缺陷。

## Immediate findings

### F1 — 非法 LLM 过滤响应被提交为“合法零结果”

**Priority: P0 · correctness**

`filterPapers` 在 JSON 解析失败或响应结构非法时返回空数组：

- `packages/core/src/pipeline/paper-filter.ts:142-158`

Pipeline 随后把任何空数组解释为当天完成且零篇：

- `packages/core/src/pipeline/pipeline.ts:225-234`

这使截断 JSON、未知 ID、未知分类、重复 ID 或 prompt 输出漂移都可能静默变成 `completed`。Scheduler 不会重试，用户看到的是“当天没有论文”，而非可诊断失败。

**建议**

将过滤结果改成判别联合或抛出专门的 validation error。只有严格验证通过的 `{"papers":[]}` 才能产生 completed-zero；解析或契约失败应映射为 `failed_transient`，并保留现有 cancellation 与 checkpoint 错误语义。

### F2 — 邮件 exactly-once 语义存在多个空洞

**Priority: P0 · correctness / duplicate side effect**

当前流程是“读状态 → 判断 → 发信 → 写状态”：

- `packages/core/src/delivery/deliver-email.ts:157-223`

它没有按 `date + recipient` 的事务或 claim。Plugin 与 CLI 同用一个 Vault 时可以并发读到“未投递”并各自发送。Hosted relay 有请求幂等键，但 BYOK Resend 请求没有等价 header：

- `packages/core/src/delivery/resend.ts:49-69`

此外，状态读取中的解析、权限和 I/O 错误全部被当作空状态：

- `packages/core/src/delivery/delivery-state.ts:127-140`

如果 provider 已接受邮件，而 `saveDeliveryState` 失败，代码会进入统一 catch、返回 `failed`，下一次自然重试可能重复发信：

- `packages/core/src/delivery/deliver-email.ts:211-263`

**建议**

1. 引入以日期和标准化收件人为 key 的 delivery transaction/lease；
2. 为 Resend 增加稳定 provider idempotency key，测试发送使用独立 key；
3. 将 provider outcome 与 bookkeeping outcome 分开，增加 `delivered_unrecorded` 一类结果；
4. 状态 `corrupt/unreadable` 对自动发送应 fail closed，不能退化为空记录；
5. 增加双并发调用、provider success + state save failure 的测试。

### F3 — Paper Index 原子写恢复和 mutation API 均有缺口

**Priority: P0 · data integrity**

`PaperIndexStore.writeAtomic` 先把主文件 rename 为 `.bak`，再安装 `.tmp`：

- `packages/core/src/services/paper-index.ts:505-527`

若进程在两次 rename 之间退出，下一次 `load()` 只检查主路径和 legacy 路径，不检查 `.bak`：

- `packages/core/src/services/paper-index.ts:170-193`
- `packages/core/src/services/paper-index.ts:472-480`

完整索引因此可能被解释为空索引。另一方面，专用 mutation 会进入同路径队列，但公开 `save(inbox)` 不进入队列：

- `packages/core/src/services/paper-index.ts:196-218`
- `packages/core/src/services/paper-index.ts:497-503`

现有调用在仓储外执行 `load → mutate → save`：

- `packages/core/src/services/daily-selection.ts:52-59`
- `packages/core/src/dashboard/history-sync.ts:114-126`

它们可以用旧快照覆盖并发写入的新状态。

**建议**

- 读取顺序改为 primary → valid backup → legacy；
- 优先复用 Host 的 `writeTextAtomic`，或抽出统一、受测试的 durable JSON replace primitive；
- 将裸 `save` 收为 internal/private，暴露 queued `mutate(callback)` 或明确的领域 mutation；
- 补充“仅剩 `.bak` 启动恢复”和“裸快照与专用 mutation 交错”的测试。

### F4 — Scheduler 结果提交失败后仍向调用方返回 `completed`

**Priority: P0 · state semantics**

Pipeline 返回 completed 后，Scheduler 再持久化状态、历史并执行投递 hook：

- `packages/core/src/services/scheduling/scheduler-driver.ts:418-474`

如果 `setCompleted` 或历史写入失败，外层 catch 仅记录错误，仍返回原始 completed result：

- `packages/core/src/services/scheduling/scheduler-driver.ts:475-484`

调用者可能收到“完成”，而 durable state 仍是 running；重启后又可能被 stale-running 恢复为失败。

**建议**

区分 pipeline work result 和 scheduler commit result。只有 durable state/history 提交成功后才对外返回 completed 并执行 post-commit hook；提交失败返回 `commit_failed` 或 `failed_transient`，同时 reload/回滚 StateStore 内存快照。

### F5 — Obsidian 1.13+ declarative Settings 绕过运行时事务

**Priority: P0 · product correctness**

Declarative controls 最终统一写嵌套值并保存：

- `plugin/src/settings/tab.ts:191-210`

旧版输出路径路径会执行 sibling 校验、重建 state/history store，并在失败时回滚：

- `plugin/src/settings/tab.ts:427-460`

Declarative 定义中，`dailyDir` 甚至没有传入 `papersDir` 做碰撞校验：

- `plugin/src/settings/definitions.ts:261-290`

这会让新版 Obsidian 用户看到设置已变更，而运行时 store 仍指向旧目录；log level 等需要立即应用的设置也有同类风险。

自定义时区又在每次 input 时未经 IANA 校验直接持久化：

- `plugin/src/settings/declarative-rows.ts:291-317`

中间输入如 `America/` 会成为有效配置对象中的无效值，并可能在 Dashboard 的时区计算中抛出 `RangeError`。

**建议**

建立统一 `SettingsChangeService`：`validate → apply runtime side effect → persist → rollback`。Declarative 与 legacy 只保留为渲染 adapter。时区使用 draft + blur/change 提交，并用 `Intl.DateTimeFormat` 验证；渲染边界提供安全 fallback。

## Engineering and architecture findings

### F6 — PR 验证面不足，独立产品和服务处于治理盲区

**Priority: P1 · engineering governance**

Push/PR workflow 只运行 lint 和 workspace typecheck：

- `.github/workflows/lint.yml:26-31`

Boundaries、tests、build 和 smoke 只在打 release tag 时执行，并通过 8 GB heap、单 worker 绕过测试内存问题：

- `.github/workflows/release.yml:60-72`

根 workspaces 仅含 `packages/*`、`apps/*`、`plugin`：

- `package.json:16-20`

因此 `services/email-relay` 和 `extensions/vscode-arxiv-daily` 不进入根 lint/typecheck/test/build。边界脚本也只扫描四个固定层：

- `scripts/check-boundaries.mjs:5-15`
- `scripts/check-boundaries.mjs:22-49`

**建议**

1. PR 至少运行 boundaries、test、build；先处理默认 `npm test` 的 OOM；
2. Worker 使用独立路径触发 job，执行自己的 install、typecheck、test、Wrangler dry-run；
3. VS Code companion 使用独立 job，并加入当前 CLI contract tests；
4. 将所有部署单元列入显式治理清单，新增目录不能静默绕过；
5. lint warning 不应继续依赖接近上限的全局额度，逐步清理 deprecated API。

### F7 — VS Code companion 的公开命令已与当前 CLI 不兼容

**Priority: P1 · broken product surface**

扩展仍注册 `runPending`、`summarizeById` 和 API key 配置：

- `extensions/vscode-arxiv-daily/package.json:14-20`
- `extensions/vscode-arxiv-daily/package.json:33-53`

调用层执行 `run-pending`、`summarize --id`，并向所有命令追加 `--config`、`--vault-root`：

- `extensions/vscode-arxiv-daily/src/pipeline-commands.js:8-15`
- `extensions/vscode-arxiv-daily/src/pipeline-commands.js:17-30`
- `extensions/vscode-arxiv-daily/src/pipeline-commands.js:33-73`

当前 CLI 明确拒绝这些 flags 和子命令：

- `apps/cli/src/main.ts:267-305`
- `apps/cli/src/main.ts:355-362`

扩展自测固定的是旧命令字符串，因此“测试通过”反而保护了失效契约。

**建议**

先确认该扩展是否已经发布及是否仍属于产品范围。若未正式支持，禁用 pipeline commands 并明确标记 Dashboard-only/experimental；若继续支持，只调用当前 `run --date` / `run --id`，要求用户先完成固定路径 TOML 配置，并加入跨仓库 CLI parser contract test。

### F8 — 两个 Host 重复装配共享 engine，已出现轻微分叉

**Priority: P2 · maintainability**

CLI 在 `apps/cli/src/runtime.ts:56-207` 手工创建 LLM、source、writer、index、checkpoints、pipeline、manual fetch、scheduler 和 delivery callback；Plugin 在以下位置重复同一对象图：

- `plugin/main.ts:157-169`
- `plugin/main.ts:317-353`
- `plugin/main.ts:373-424`
- `plugin/main.ts:443-462`

Host 生命周期差异合理，不应合并两个顶层 composition root；共享业务装配没有可比较契约，后续新增 dependency 容易只接到一端。

**建议**

抽取窄的 host-neutral engine assembler，接收已规范化设置、adapters、缓存/路径策略和 logger，返回 pipeline/manual-fetch/index/checkpoint 等共享服务。Plugin/CLI 继续各自负责 scheduler 启动方式、信号、UI、cron 和生命周期。

### F9 — Core 配置和公开 API 仍带有历史宿主形状

**Priority: P2 · boundary clarity**

共享设置仍名为 `PluginSettings`，Scheduler 接收整个产品对象：

- `packages/core/src/settings/types.ts:37-42`
- `packages/core/src/settings/types.ts:89-97`
- `packages/core/src/services/scheduler.ts:21-37`

CLI 因此必须构造永不启用的 plugin schedule，同时另存 `CliScheduleIntent`：

- `apps/cli/src/config.ts:23-49`
- `apps/cli/src/config.ts:269-270`

Core package 又只提供一个根 export，并 wildcard 暴露大量内部实现：

- `packages/core/package.json:18-20`
- `packages/core/src/index.ts:3-92`

审查统计显示入口约暴露 486 个符号，包含低层 parser、checkpoint helper 和测试 seam。

**建议**

- 拆出 `CoreGenerationSettings`、`EmailSettings`、`InProcessScheduleSettings`，由各产品组合；
- Scheduler 接收窄 `{ timezone, schedule }` 契约；
- 根入口改为显式稳定 facade，按真实宿主能力提供受控 subpath exports；
- 测试 reset seam 不从生产入口导出。

### F10 — 结构热点应按事务和行为边界拆，不按行数拆

**Priority: P2 · maintainability**

最值得提取的边界是：

- `ArxivPipeline.runForDateInner`：发现、过滤、内容、详情、日报 commit、checkpoint 与 index projection 混在一个应用流中，见 `packages/core/src/pipeline/pipeline.ts:146-489`；
- `ManualFetchService.fetchAndSummarize` 重复详情 note 工作流，见 `packages/core/src/services/manual-fetch.ts:50-302`；
- 两个 checkpoint store 复制 queue、primary/backup/tmp 和 hash/promotion 机制，并存在 pipeline contract 与具体 store 的双向层级编织；
- CLI init 的 15 个步骤集中在 `apps/cli/src/init.ts:329-840`；
- Dashboard 和 Settings 大类同时承担 controller、repository、renderer、action coordinator 和宿主 facade。

**建议顺序**

1. 先修 F1–F5 的事务语义；
2. 提取 `PaperNoteInspector`、`GeneratePaperDetailUseCase`、`DailyReportCommitter`；
3. 提取共享 `ValidatedJsonDocumentStore` / checkpoint persistence primitive；
4. Dashboard 拆成 session/controller、history gateway、renderer、host action coordinator；
5. Settings 拆成 change service、legacy/declarative adapters、topic/setup/sensitive-field components；
6. CLI init 按 step handler 拆分。

不建议只移动 helper、按 300 行切文件或重命名目录而不改变依赖方向。

## Frontend and visual findings

### F11 — Dashboard 异步 reload 缺少“只提交最新结果”的生命周期协议

**Priority: P1 · UI correctness**

`reloadIndex()` 可由打开、刷新、运行完成、下载、删除等入口并发调用；每次异步完成后都会无条件写状态并 render：

- `plugin/src/dashboard/view.ts:254-340`

月历翻页已有 sequence token：

- `plugin/src/dashboard/view.ts:417-441`

完整 reload 没有同类 guard。`onClose()` 只清 DOM，已在途的 reload 仍可完成并再次 render。另一个缓存问题是 history skip 只比较 Markdown 路径，命中后甚至不重新读取廉价的 Paper Index：

- `plugin/src/dashboard/view.ts:288-305`

因此 PDF path 或已有文件内容变化可能不会及时反映。

**建议**

- 完整 reload 使用 generation token 或 single-flight + dirty bit；
- close 时失效 token，每个 await 后检查 open/generation；
- 拆分“总是加载最新 index”和“路径/mtime 变化时才做昂贵 history reconciliation”；
- 增加后发先至、关闭期间完成、路径不变但 index 变化的测试。

### F12 — Dashboard 窄窗布局和操作层级需要针对性更新

**Priority: P1 · responsive UX / aesthetics**

表格固定最小宽度并依赖横向滚动：

- `plugin/styles.css:1164-1175`

每行有六个图标操作，排成两行三列：

- `plugin/src/dashboard/view.ts:1433-1493`
- `plugin/styles.css:1296-1306`

现有 container query 会重排工具栏、筛选器、概览和日历，但不改变结果表：

- `plugin/styles.css:1495-1544`

结果前还有 tabs、五个顶栏动作、筛选器、日历、四个 stats、批处理、排序与页大小：

- `plugin/src/dashboard/view.ts:726-820`
- `plugin/src/dashboard/view.ts:1201-1334`

**建议**

- 在约 680–720px 容器宽度以下切换为 compact card/list；
- 卡片保留标题、作者、主题、日期和 2–3 个高频动作，其余进入 overflow menu；
- 顶部保留 `Run today` 主 CTA，Refresh 为轻量图标，Summarize by ID 归入 More；
- 批处理条仅在有 selection 时出现，并可 sticky；
- stats 压为一行摘要或可折叠概览。

这属于局部信息架构与响应式精修，不需要改变 Obsidian-native 视觉语言。

### F13 — 运行状态、加载状态和动态焦点不够可靠

**Priority: P1 · interaction / accessibility**

Dashboard 标题只在 render 时读取 operation snapshot：

- `plugin/src/dashboard/view.ts:688-704`

其他入口启动任务时，已打开 Dashboard 不会实时更新。结果区域的 sort、pagination、clear selection 会触发子树清空与重建，除星标外没有统一 focus restoration。Hub modal 声明了 ARIA tabs，却没有 roving tabindex 或方向键协议：

- `plugin/src/dashboard/hub-modal.ts:30-53`
- `plugin/src/dashboard/hub-modal.ts:97-139`

`All` 页大小还允许一次渲染无界 table rows，并为每行创建多个监听器。

**建议**

- Dashboard 打开时订阅 operation registry，关闭时注销；状态行使用 `aria-live="polite"`；
- 长操作统一按钮文案与互斥 disabled 状态；
- 保留稳定 toolbar/pagination DOM，或以逻辑 focus key 恢复焦点；
- 结果摘要播报页码、数量和 selection count；
- Hub tabs 实现 roving tabindex、Arrow/Home/End；
- 删除 `All` 或设硬上限，真正需要全量时采用渐进加载/窗口化。

### F14 — 日历、高对比度和状态反馈过度依赖颜色、透明度与 `title`

**Priority: P2 · accessibility / visual semantics**

月历内部区分 future、not updated、report missing、permanent failure、runnable、report 和 zero-result，但多个不可操作日期都渲染为普通 span。永久失败详情主要依赖不可聚焦元素上的 `title`：

- `plugin/src/dashboard/view.ts:918-963`
- `plugin/src/dashboard/view.ts:1099-1117`
- `plugin/src/dashboard/calendar.ts:128-167`

样式大量通过 opacity 和绿色/accent 表达状态，未提供 `forced-colors` 或 `prefers-contrast` 规则；reduced-motion 也没有覆盖所有 Dashboard/progress 动画。

**建议**

- 给关键空状态独立 class、图标和可见 legend；
- 使用 `<time datetime>` 与 visually-hidden 状态文本，今天设置 `aria-current="date"`；
- permanent failure 提供可聚焦详情/修复入口；
- 增加 `forced-colors`、`prefers-contrast: more` 和完整 reduced-motion 规则；
- 正文状态不要依赖整体 opacity，颜色始终配合图标、边框样式或文字。

### F15 — Settings 与 Modal 需要统一行为规格，而非增加更多样式

**Priority: P1–P2 · consistency**

新版与旧版 Settings 在 Quick start、category 自定义、reasoning、术语和秘密字段行为上已有差异。Topic 编辑还可能在每次 input 后保存并触发 declarative update，导致重建自身和焦点不稳：

- `plugin/src/settings/tab.ts:1335-1350`
- `plugin/src/settings/tab.ts:1554-1570`
- `plugin/src/settings/tab.ts:1708-1756`

Modal 的 title、inline validation、默认焦点和 footer 布局也不一致；Hub 的日志级别过滤器位于 panel 内容之后。

**建议**

- 先定义统一 Settings 行为和 change service，再保留两种 renderer；
- Topic 使用本地 draft、防抖串行保存、blur/close flush；
- 秘密字段统一为不回填明文、显式 Replace/Clear；
- 建立轻量 modal shell：统一 `titleEl`、首个有效输入焦点、inline error 和 Cancel/CTA footer；
- 将日志过滤器置于内容前并 sticky。

## Visual design decision

### 应更新的部分

1. Dashboard 窄窗结果布局；
2. 顶部动作层级与批处理可见性；
3. 统一 loading / warning / error / partial 状态面板；
4. 实时运行阶段与可取消反馈；
5. 日历非颜色状态编码与图例；
6. Settings validation、秘密字段和两种 API 的行为一致性；
7. High contrast、forced colors、reduced motion；
8. Modal 与 tabs 键盘规格。

### 应保留的部分

- Obsidian 原生主题变量和控件；
- 克制的圆角、边框和背景层次；
- Dashboard container query 基础；
- 针对 setup/no index/no result/no starred 的可操作空状态；
- Settings setup guide 的任务型结构；
- 日历按钮、星标 pressed state、tabpanel 关联和 progress ARIA 的现有良好语义；
- Similar Papers 的信息卡片层级。

### 不建议的设计工作

- 引入 React/Vue 只为拆分当前 imperative DOM；
- 建立脱离 Obsidian 的品牌色、字体和阴影系统；
- 全面卡片化宽屏 Dashboard；
- 为视觉新鲜感添加渐变、大面积强调色或复杂动效；
- 在真实第二来源出现前做完整多源 UI；
- 仅为减小 bundle 做高风险替换；当前 Plugin bundle 约 462 KiB，不是首要问题。

## Recommended roadmap

### Phase A — Correctness first

- F1：过滤 validation error 语义；
- F2：投递幂等与 bookkeeping outcome；
- F3：Paper Index backup recovery 与 queued mutation；
- F4：Scheduler durable commit 语义；
- F5：统一 Settings change transaction 与时区校验。

### Phase B — Verification and product hygiene

- 修复默认测试 OOM，让普通 `npm test` 可在合理内存下运行；
- PR 加入 boundaries/test/build；
- Worker 独立 CI 与 client-worker contract；
- 决定 VS Code companion 的保留、降级或修复路径。

### Phase C — Structural refactoring

- 提取 shared engine assembler；
- 收窄 Core settings 和 package API；
- 提取详情 use case、日报 committer、checkpoint persistence primitive；
- 拆分 Dashboard session/gateway/renderer/actions；
- 拆分 Settings change service 与两种 renderer。

### Phase D — Targeted UI refinement

- Dashboard compact card/list breakpoint；
- Toolbar、stats、batch action 层级；
- operation subscription 与统一 state panels；
- focus restoration、Hub tabs、calendar semantics；
- high-contrast 与 modal consistency；
- 在真实 Obsidian 中做主题、窄 leaf、键盘和屏幕阅读器验证。

## Decisions that should remain unchanged

以下选择经当前源码复核后仍然合理：

1. 一个 host-neutral TypeScript Core，由 Plugin 和 CLI 直接消费；
2. Node runtime 与 Obsidian adapters 分离；
3. Plugin 与 CLI 配置独立，不做自动设置同步；
4. Plugin 进程内 scheduler 与 CLI 外部 cron intent 分离；
5. Email relay 保持独立部署和信任边界；
6. `paperKey = source:externalId` 的身份模型继续保留；
7. Host composition root 继续属于各 Host；只抽取共享 engine assembly；
8. 日报文件作为 durable commit、Paper Index 作为可修复 projection；
9. 日报逐篇结构化摘要保持顺序执行和 checkpoint 语义；
10. 现有 Obsidian-native 视觉方向继续保留。

## Final recommendation

架构无需重建，前端无需全面改版。下一步最有价值的是启动一个独立实施事项，先完成 Phase A 的五个正确性主题；其中邮件、Paper Index 和 Scheduler 应按事务语义一起设计，Settings change transaction 可并行推进。完成正确性与 PR 验证后，再进行 Dashboard/Settings 结构拆分和局部视觉精修。

# arXiv Daily 插件审查报告 — 2026-07-03

> 只读审查,不含代码改动。本文档为待办清单,供后续逐项处理。

## 方法

5 个独立 opus 子 agent 并行审查 `refactor/scheduler-hybrid` 分支的当前代码(含调度器 hybrid 重构 + dashboard run handler 的 refresh/notice 顺序修复),维度:性能与响应性、健壮性与错误处理、交互 UX 与可访问性、架构与可维护性、安全与数据完整性。每个 agent 独立读代码、带 `file:line` 证据。

**交叉验证信号**:被多个 agent 独立命中的问题标注 `[N 命中]`,可信度最高。行号相对本分支;部分位于重构前既有代码,`main` 上同样存在。

---

## P0 — 数据丢失(正常使用即触发,无需攻击者)

### A. 并发读改写 `papers.json` / `run-state.json` 无全局锁 → 静默丢失 star/read/状态 `[健壮性 + 安全 2 命中]`

- **证据**:`paper-index.ts:163-199`(load→mutate→save)、`state-store.ts:202-226`;`RunLock`(`run-lock.ts`)仅按日期加锁,不保护索引写;`scheduler-driver.ts:64-67` 的 `setInterval` 调 `tick()` 无重入保护。
- **问题**:dashboard 点星与后台 scheduler 运行同时发生时,两个 load-mutate-save 周期交错 → 后写覆盖先写。锁只按日期 → 能产生并发跨日期运行,读改写同一份 `papers.json`/`run-state.json`,且 `writeAtomic` 用固定 `.tmp`/`.bak` 路径,两个写者交错 rename 会损坏或丢数据。
- **影响**:正常操作即可丢失 star/read/status,无报错。Nextcloud 同步放大风险。
- **修复方向**:所有索引写经单一 async 队列/互斥串行化(参照 `run-history.ts:101-107` 的 `appendQueue`,或用 `RunLock.withLock("paper-index", …)`);`tick()` 加重入 flag + 进程级锁让手动与调度串行;`writeAtomic` 用唯一临时名。**改并发模型,落地前需确认设计方向。**

### B. LLM 过滤阶段失败被静默记成"完成、0 篇" `[健壮性 P1]`

- **证据**:`paper-filter.ts:69-73` catch 后 `return []`;`pipeline.ts:156-160` 把空过滤结果当 `{ kind: "completed", papersWritten: 0 }`;`scheduler-driver.ts:354-367` 据此标 `completed`,`isDone()` 返回 true。
- **问题**:LLM 临时故障 / 429 / 超时 / API key 被清空(调度器 tick 不重跑 `gateLlm`)都落进这个 catch。空 arXiv 列表反而正确 `pending` 重试(`pipeline.ts:111-114`)—— 过滤错误比空列表还糟。
- **影响**:有论文的一天被永久标记 0 篇、不再重试、不写日报。静默数据丢失。
- **修复方向**:`filterPapers` 对非取消错误改为 rethrow / 返回判别式错误,pipeline 映射为 `failed_transient`(与 `summarizeDaily` 失败处理一致,`pipeline.ts:238-244`)。**改结果语义,落地前需确认。**

### C. `run-state.json` 损坏使插件加载失败 `[健壮性 + 安全 2 命中]`

- **证据**:`state-store.ts:168-177` `JSON.parse(raw)` 无 try/catch;`main.ts:90` `onload` 中 `await load()` 无兜底;`writeAtomic` 写的 `.bak`(`state-store.ts:217-224`)读时从不使用。
- **问题**:Nextcloud 同步冲突 / 半写文件 / 中断的 `writeAtomic` → 无效 JSON → 解析抛错拒绝 `onload`。
- **影响**:插件硬崩溃,需手动修文件。
- **修复方向**:包 try/catch,失败回退 `.bak` → `{}`,记 notice;把坏文件 `.corrupt` 隔离而非丢弃。

---

## P1 — 响应性(改动小、收益大;与手动测试直接相关)

### D. 日历格子点击 = 已修 `runToday` 的同款 bug,漏了这处 `[性能 + UX 2 命中]`

- **证据**:`runDateFromCalendar`(`view.ts:1231-1256`)在 `view.ts:1240` 先 `await recentDates.refresh()` 再于 `:1248` 弹提示;由 `renderRunnableCell`(`view.ts:1195-1197`)直接 `void this.runDateFromCalendar(...)` 触发,未走 `runControlAction`(`view.ts:1882`)。
- **问题**:点日历 runnable 格子后数秒无反馈(refresh 是网络调用);且按钮不禁用,运行期可重复点击 → 并发运行 + 交错提示。这是 `runToday`/`runAllPending` 同款问题的第三处。
- **修复方向**:先弹 "running for {date}…" 再 refresh;经 `runControlAction` 或 `button.disabled = true` 防重入。

### E. 搜索框无防抖 `[性能 P1]`

- **证据**:`view.ts:1303-1307`,`input` 事件每次击键跑 `renderCurrentResults` → `queryDashboard`(全量 filter + sort + stats,~5 遍遍历)+ 重建 20 行表格(~120 次 `setIcon` SVG 注入)。
- **问题**:索引几千篇时打字明显卡顿。
- **修复方向**:防抖 200-300ms;复用现有防抖模式(`daily-selection.ts:110-127`)。

### F. `recentDates` 无 staleness TTL → 每次操作后重复网络请求 + 双重渲染 `[性能 P1]`

- **证据**:`recent-dates.ts:108` 存 `refreshedAt` 却从不读;`ensureRefresh` 仅去重并发中请求。`view.ts:445` `reloadIndex` 每次调 `refreshRecentDatesForForeground()`,而 `reloadIndex` 在每次 star/下载/总结/日历运行后触发。3 秒 race 超时后后台 promise 又调 `this.render()`(`view.ts:517/521`)→ 第二次全量渲染。
- **问题**:连点 5 次星 = 5 次 arXiv 请求;例行操作后可见 re-render 闪烁。
- **修复方向**:加 TTL(如 10-15 分钟),`Date.now() - refreshedAt < TTL` 时跳过 refresh。此改可一并吸收 P2 的 dashboard/scheduler 双重 refresh(见 I 之外的 P2.4)。

---

## P2 — 数据完整性 / 密钥 / 状态正确性

### G. API key 明文存 `data.json` + LLM base URL 无 HTTPS 校验 `[安全]`

- **证据**:`main.ts:237-240` `saveData({ settings })`;`settings/types.ts:2`;UI 标注 "Stored locally in data.json"(`tab.ts:129`)。`client.ts:125/137` 以 `Authorization: Bearer` 发送;`validateLlmConfig`(`validation.ts:9-14`)仅查非空;`client.ts:134` 的 `fetchModels` 有绕过代理的裸 `fetch` 仍带 key。
- **影响**:key 随 Nextcloud 同步到服务器和所有设备;填 `http://`(非 loopback)会明文传输 Bearer。
- **修复方向**:非 loopback host 强制 `https:` 并在设置 UI 警告;统一走 `requestUrl` 路径;文档提示把 `data.json` 加入 Nextcloud sync-ignore,或密钥移出同步设置(`hosts/obsidian/secrets.ts` 已有抽象但当前仍读写 `settings.llm.apiKey`)。

### H. 日报文件已写、明细阶段失败 → 永久丢明细 + 记 0 篇 `[健壮性 P2-1]`

- **证据**:`pipeline.ts:94-97` `dailyExists` 短路返回 `completed, 0`;日报写于 `:266`,先于明细循环 `:271-310`。明细阶段取消/抛错 → `failed_transient`,重试时 `dailyExists` 为真 → 永不再进明细循环。`preservedCompletedPaperCount`(`scheduler-driver.ts:411-423`)仅在前态为 `completed` 时保留计数,此处为 `failed_transient` → 记 0。
- **修复方向**:细粒度追踪"明细阶段完成"标记,或短路条件改为"是否仍需写明细"而非仅看日报文件存在;重完成时保留真实计数。

### I. 批量持久化块无 per-date 保护 → 首个写错误中止整批 `[健壮性 P2-2]`

- **证据**:`scheduler-driver.ts:353-388`,pipeline 执行错误已 catch 转结果,但 `setCompleted`/`setFailed` 的持久化块无保护,抛错传出 `withLock` → 拒绝 `tryRun` → `runAllPending`(`:270`)/`retryFailedInLookback`(`:222`)/`tick`(`:119`)的循环抛出,剩余日期静默跳过。
- **修复方向**:per-date 包裹持久化块,单日期写失败记录后继续循环;仅取消才中止批次。

### J. Nextcloud 跨设备 last-writer-wins `[健壮性 + 安全 2 命中]`

- **证据**:`state-store.ts:25-28` + `main.ts:90` state 仅 onload 读一次,此后写前不重读;`paper-index.ts:152` 写 `updatedAt` 但 load 时不比较;`state-store.ts:202-226` 的 `.tmp`/`.bak` 也会被同步。
- **影响**:设备 B 的更新被持 stale 内存的设备 A 覆盖;冲突副本文件重现。
- **修复方向**:写前重读 + 按 date/arxivId 的 `updatedAt` 合并(last-write-wins per entry);检测 `*conflicted copy*` 兄弟文件并警告;scratch 文件移出同步树。

### K. arXiv ID 未校验就拼路径 `[安全 P2-4]`

- **证据**:`arxiv-parser.ts:91` `id` 取自 listing HTML 无校验 → `markdown-writer.ts:32-36` `${papersDir}/${id}.md` 与 `paper-content.ts:232-244` 缓存路径。Obsidian `normalizePath`(`hosts/obsidian/storage-adapter.ts:7-9`)不拒 `..`;Node adapter(`hosts/node/storage-adapter.ts:72-79`)会拒。手动录入已校验(`manual-fetch.ts:36-49` `ID_RE`),自动 listing 路径未校验。
- **修复方向**:`arxiv-parser.parsePaper` 按 `^\d{4}\.\d{4,5}(v\d+)?$` 校验(复用 `ID_RE`),不匹配则丢弃。

### L. cancellation 全局状态跨日期泄漏 `[健壮性 P2-4]`

- **证据**:`cancellation.ts:13-42`,`cancellationRequested` 仅在 `controllers.size === 0` 时重置;`cancelAll` 中止所有 controller。并发跨日期运行下,取消一个中止全部;D1 结束后 D2 仍活时启动 D3,`prepareRun()` 见 `size === 1` 不重置 flag → `begin("D3")` 立即中止 D3。
- **修复方向**:cancellation 按 date 作用域(controller/flag map keyed by date)。

---

## P3 — 优化 / 一致性 / 架构

### 架构(以已完成的调度器重构为质量标杆)

- **M. 抽共享 UI run-layer** `[架构 "最该做的下一个"]`:`describeResult` / `describeManualResult` / `describeRunResults` 在 `view.ts:2287-2306` 与 `commands.ts:377-396` **逐字重复**且入参 `any`(丢弃现成的 `PipelineResult`/`SchedulerResult`/`ManualFetchResult` 联合)。gate(`commands.ts:31-47` vs `view.ts:1902-1930`)与 run handler(`commands.ts:49-80` vs `view.ts:1932-1968`)结构重复且文案已漂移(`•` vs `-`、`cannot run` vs `cannot summarize`)。→ 抽 `run-format.ts`(纯)+ `config-gate.ts` + `RunController`(deps 对象仿 `SchedulerDriverDeps`)。
- **N. god file 拆分**:`view.ts`(2727 行,~85 处 `this.plugin.*` 跨 12 子系统)、`commands.ts`(843 行)、`settings/tab.ts`(960 行)。仿调度器重构:thin facade(仅 Obsidian 生命周期)+ `DashboardController`(状态 + 数据加载/动作编排)+ `dashboard/calendar.ts`(把 `buildCalendarCells`/`isRunnable`/`getLookbackDates` 等困在 UI 里的纯逻辑抽出)+ `dashboard/render/*`(DOM builder)+ `dashboard/hub-modal.ts`。`getLookbackDates`(`view.ts:1077`)重复硬编码 `LOOKBACK_DAYS = 5`,应与调度器共享。
- **类型安全**:`view.ts:810` `(app as any).setting`、`main.ts:212` cast `tickToday()` 结果、`commands.ts:716` `app as any` + `:750/766` `as any` JSON 遍历 → 引入现成联合类型 + 类型化 helper。
- **DI 一致性**:`DailySelectionService`(`daily-selection.ts:104`)取 `buildPaperIndex` closure 但仅测试中构造,生产未接线 —— 要么接线要么标注为 test-only。

### 健壮性 P3

- **LLM 4xx 当 transient 重试**(`client.ts:216-220`)→ 400/401/404 重试 3× 再churn 到 `MAX_TRANSIENT_ATTEMPTS = 10`。修:非 429 的 4xx 分类为 `failed_permanent`。
- **单类目 fetch 失败拖垮整天**(`pipeline.ts:397-411`)→ 累积 per-category,全失败才失败该日。
- **无 `Retry-After` / 429 退避**(`arxiv-fetcher.ts:113-144` 固定 2s→4s)→ 尊重 `Retry-After` + jitter。
- **`respectDelay` per-instance,而 fetcher per-run**(`arxiv-fetcher.ts:183-188` + `main.ts:277-285`)→ 并发运行绕过节流。修:共享一个限流 fetcher。
- **隔夜窗口静默禁用调度器**(`time.ts:57-60`,`start > end` 恒 false)→ 设置校验 `runAtLocal <= runUntilLocal` 或支持跨夜。
- **LLM 流挂起使日期卡 `running`**(`client.ts:204-213`)→ 加 chunk 间空闲超时绑定 abort signal。
- **`run-history.jsonl` append 是全文件重写、非原子、无界**(`run-history.ts:78-88`)→ 真 append 模式 + 按大小轮转。

### UX / 可访问性 P3

- 错误态无恢复入口(`view.ts:604-610` 在渲染 toolbar 前 return)→ 保留 Refresh/Retry。
- 无"运行中"状态指示 / last-run(`view.ts:823-828` 仅标题)→ header 加状态行,统一 `isRunning` flag 门控所有 run 入口。
- 日历空格子是死 tab stop(`view.ts:1007-1011`)→ 禁用或给 aria-label。
- progressbar 缺 `aria-valuenow/min/max`(`status-bar.ts:229`)。
- 单字段 modal 不支持回车提交(`commands.ts:412-436` 等)。
- 提示文案标点不统一(`…` vs `...`、`•` vs `-`、`→` vs `->`)—— 随 M 的 `run-format.ts` 统一。
- 日历标记字体 9-10px + opacity 偏低(`styles.css:568-572`、`:630-637`)。
- 同一动作标签分歧("Run Today" vs "Run now (today)")。
- `<th>` 缺 `scope="col"`;HubModal tab 用 `aria-pressed` 而非 `role="tab"`;`styles.css:328-332` 对 input 加 `user-select: none`(疑似误加)。

### 性能 P3

- 勾选框切换触发全表重建(`view.ts:1441-1447`)→ 就地更新。
- 月历每次 render ~30-42 次 disk `exists()`(`view.ts:575-585`)→ 复用已加载的 `byDate` map。
- `queryDashboard` 每次 render 跑两遍(`view.ts:612` 与 `:646`)→ 算一次传入。
- tab 切换触发全量 `render()`(`view.ts:855-859`)→ 用 `renderCurrentResults` + tab count 更新。
- `cleanupCaches()` 每次启动扫描+解析全部缓存文件(`main.ts:129/353-370`,fire-and-forget 不阻塞 onload)→ 按天 gate。
- star 切换是读改写全文件 + 全量 reload 无乐观 UI(`view.ts:1985-1999`)。

---

## 已验证做得好(无需动)

- **prompt 注入加固**三处调用点(`paper-filter.ts:48-54`、`summarizer.ts:155-169`、`:451-462`)都正确落地,`<paper_data>` 分隔 + 单源 guard。
- **调度器 tick guard**(commit 35a3d5b):`tick()` 在 `store.isDone(today)` 时于 `scheduler-driver.ts:105` 早返回,先于 `refresh()`(`:110`),完成后不再发 arXiv 请求;`failed_transient` 不算 done,重试不受影响。
- **history-sync skip guard**(`view.ts:2355-2366`)避免未变时重扫所有 md。
- 分页封顶 20 行,无需虚拟化;DOM 用 `createEl({text})` textContent 无 XSS;`window.open` 带 `noopener`;`logger` buffer 有界(5000 行);无网络监听面、无 `eval`/动态 import 远程内容;arXiv URL 硬编码 `https` + `encodeURIComponent`;diagnostics/history 复制功能不含 key。

---

## 建议落地顺序

1. **P1 响应性三件套(D + E + F)** —— 局部、低风险、边界清晰,正是当前手动测试关注的响应性。可打包 codex task 委托。
2. **P0 数据丢失(A + B + C)** —— 最严重;但 A(全局锁,改并发模型)、B(改 pipeline 结果语义)动核心行为,落地前需确认设计方向。
3. **P2 完整性 / 密钥(G + H + I + J + K + L)**。
4. **P3 架构(M 优先,为 N 铺路)+ 其余优化**。

> 落地方式:有明确边界的实现任务走 codex 委托(写 `task.md` → 后台 Agent 跑 → 读报告);动核心行为的先确认设计。

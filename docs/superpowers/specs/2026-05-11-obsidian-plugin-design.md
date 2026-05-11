# arxiv-daily Obsidian 插件设计（MVP）

**日期：** 2026-05-11
**分支：** `obsidian-plugin`
**范围：** v1 (MVP) 单配置版；v2 多 profile 在后续迭代

---

## 1. 目标

把现有的 `arxiv_daily.py` 脚本重写为一个原生 TypeScript Obsidian 插件，提供：

- 设置 GUI（替代 `.env`）
- 在 Obsidian 内自动调度（catch-up loop）
- 手动触发 + 按指定日期补跑
- 跨平台（Win / macOS / Linux）

输出仍是 Markdown 文件，落入 vault，可被 Obsidian 直接打开。

---

## 2. 非目标 (v1)

以下功能 v1 不做，留给 v2：

- 多 profile（多研究方向并存）
- 跨机器同步 profile state
- 操作系统层 cron 后备 / 独立 CLI
- per-profile LLM 覆盖

v1 的"单配置"在 v2 升级时自然成为 default profile，迁移平滑。

---

## 3. 架构

```
┌─ Obsidian Plugin (main.ts) ─────────────────────────────────────┐
│                                                                  │
│  Settings Tab   Commands & Ribbon                                │
│       │              │                                           │
│       ▼              ▼                                           │
│  ┌──────────── SchedulerService ───────────┐                     │
│  │  - tick every N min                     │                     │
│  │  - compute pending dates (lookback)     │                     │
│  │  - dispatch to Pipeline via RunLock     │                     │
│  └──────────────────────────────────────────┘                     │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────── ArxivPipeline ──────────────┐                     │
│  │  1. ArxivFetcher   GET /list/<cat>/recent → parse by date    │
│  │  2. PaperFilter    LLM: relevant + category + detail flag    │
│  │  3. HtmlExtractor  GET full text for detail papers (cached)  │
│  │  4. Summarizer     LLM: daily + per-paper detailed reports   │
│  │  5. MarkdownWriter write to vault via VaultAdapter           │
│  │  6. StateStore     mark profile×date completed               │
│  └──────────────────────────────────────────┘                     │
│       │                                                          │
│       ▼                                                          │
│  Cross-cutting:                                                  │
│  - LlmClient (OpenAI-compat, openai npm SDK, stream + retry)     │
│  - HtmlCache  (Electron userData dir, TTL)                       │
│  - StateStore (plugin loadData/saveData, per-machine)            │
│  - RunLock    (per-key in-memory, prevent concurrent runs)       │
│  - Logger     (console + Obsidian Notice)                        │
└──────────────────────────────────────────────────────────────────┘
```

**职责边界：**
- **Settings/Commands**：纯 UI，不做 IO，触发 Pipeline 都经过 RunLock
- **SchedulerService**：唯一的"什么时候 tick"决策点；不直接知道 Pipeline 内部
- **ArxivPipeline**：无状态，输入 `(config, date)`，输出 `{success, errorKind?}`
- **StateStore**：唯一持久化"哪天跑过了"
- **RunLock**：保证同一 `(date)` 不会并发执行

---

## 4. 数据模型

### 4.1 设置 (saved via plugin `saveData()`)

```typescript
interface PluginSettings {
  llm: {
    apiKey: string;
    baseUrl: string;            // 默认 "https://api.deepseek.com/v1"
    model: string;              // 默认 "deepseek-v4-pro"
    temperature: number;        // 默认 0.3
    timeoutMs: number;          // 默认 300_000
    thinkingMode: boolean;      // 默认 true
    reasoningEffort: "low"|"medium"|"high"; // 默认 "high"
  };
  arxiv: {
    category: string;           // arXiv 分类（如 "astro-ph"、"cs.LG"），默认 "astro-ph"
    researchInterests: string;
    detailCriteria: string;
    // 注意：以下 detailCategories / categoryTagMap / categoryDisplayMap
    //   里的 key 是「LLM 输出的语义分类」（如 "photo-z"、"galaxy-cluster"），
    //   不是 arXiv 官方分类。Python 脚本里的 CATEGORY_* 同义。
    detailCategories: string[]; // 允许标 detail 的语义分类
    categoryTagMap: Record<string,string>;
    categoryDisplayMap: Record<string,string>;
    timezone: string;           // 默认 "Asia/Shanghai"
  };
  output: {
    dailyDir: string;           // vault 相对路径，默认 "arxiv-daily/daily"
    papersDir: string;          // vault 相对路径，默认 "arxiv-daily/papers"
  };
  schedule: {
    enabled: boolean;
    runAtLocal: string;         // "HH:MM"，默认 "09:30"
    tickIntervalMin: number;    // 默认 20
    lookbackDays: number;       // 默认 5，最大 5 (arXiv /recent 上限)
  };
  advanced: {
    requestDelayMs: number;     // 默认 3000
    cacheExpiryDays: number;    // 默认 7
    sectionCharLimit: number;
    paperCharLimit: number;
    dailyCharLimit: number;
    skipSections: string[];
    prioritySections: string[];
    logLevel: "debug"|"info"|"warn"|"error";
  };
}
```

### 4.2 运行状态 (plugin local, per-machine)

```typescript
interface RunState {
  // key = ISO date "YYYY-MM-DD"
  [date: string]: {
    status: "pending" | "running" | "completed"
          | "failed_transient" | "failed_permanent";
    lastAttempt: number;        // epoch ms
    attempts: number;
    error?: string;
  }
}
```

**重试规则：**
- `completed` / `failed_permanent`：catch-up 跳过
- `failed_transient`：catch-up 至少间隔 `tickIntervalMin` 后重试
- `attempts >= 10`：转为 `failed_permanent`

---

## 5. 关键流程

### 5.1 Catch-up tick

```
每 tickIntervalMin 分钟：
  if !schedule.enabled: return
  now = local time in timezone
  for d in [today, today-1, ..., today-(lookbackDays-1)]:
    state = StateStore.get(d)
    if state.status in {completed, failed_permanent}: continue
    if state.status == running: continue (RunLock 兜底)
    if state.status == failed_transient
       && (now - state.lastAttempt) < tickInterval: continue
    if d == today && now < schedule.runAtLocal: continue  # 不抢跑
    RunLock.acquire(d) ?: continue
    StateStore.set(d, running)
    result = await ArxivPipeline.run(d)
    StateStore.set(d, result.status, ...)
    RunLock.release(d)
```

### 5.2 ArxivPipeline.run(date)

1. **抓取**：`GET https://arxiv.org/list/<cat>/recent`（HtmlCache 命中则跳过）
2. **解析**：把页面按 announce date 分组，挑出 `date` 对应那段
   - 若 `date` 不在 `/recent` 里：返回 `failed_transient`（可能 arXiv 还没发布）
3. **过滤（LLM）**：单次调用送入所有 papers 的 `(arxiv_id, title, abstract)`，返回 JSON `[{id, relevant, category, detail}]`
4. **抓全文（仅 detail 论文）**：每篇 `GET https://arxiv.org/html/<id>`，HtmlCache 命中跳过；按 priority/skip sections 切片
5. **总结（LLM）**：
   - 日报：所有 relevant 论文一次性总结，按 category 分组
   - 详细报告：每篇 detail 论文单独总结
6. **写盘**：通过 Obsidian Vault API 写 `<dailyDir>/YYYY-MM-DD.md` 和 `<papersDir>/<id>.md`
7. **返回**：`{status: completed | failed_transient | failed_permanent, error?}`

错误分类：
- `failed_transient`：HTTP 5xx / 网络 / arXiv 当日未发布 / LLM 限流
- `failed_permanent`：HTML 解析失败 / LLM 返回非法 JSON（重试 N 次后）/ 配置缺失

### 5.3 手动触发

- **Run now**：等价于 SchedulerService 立刻跑一次，无视 `schedule.enabled` 和"不抢跑"判断
- **Run for date**：弹日期 picker，对该日期执行 Pipeline（即使不在 lookback 窗口内——但若不在 `/recent` 里会得到 `failed_transient`，UI 提示用户该日期已不可补）

---

## 6. UI 设计（Settings Tab）

按以下顺序排列：

1. **LLM 配置**
   - API Key (password input)
   - Base URL (默认 `https://api.deepseek.com/v1`)
   - Model (text，默认 `deepseek-v4-pro`，旁注提示其他可选：`deepseek-v4-flash` / `deepseek-chat`(弃用) / `deepseek-reasoner`(弃用))
   - Temperature, Timeout(s)
   - Thinking mode (toggle) + Reasoning effort (dropdown)

2. **arXiv 配置**
   - Category
   - Research interests (textarea)
   - Detail criteria (textarea)
   - Detail categories (textarea, 一行一个)
   - Category → Tag map (key-value editor)
   - Category → Display name map (key-value editor)
   - Timezone

3. **输出 & 调度**
   - Daily dir (vault 相对路径)
   - Papers dir (vault 相对路径)
   - Schedule enabled (toggle)
   - Run at (HH:MM)
   - Tick interval (min)
   - Lookback days

4. **高级**（折叠默认）
   - Request delay, cache expiry days
   - Section/paper/daily char limits
   - Skip / priority sections
   - Log level

### Commands

- `arxiv-daily: Run now` — 立即执行当天
- `arxiv-daily: Run for date...` — 弹日期 picker
- `arxiv-daily: Open today's daily` — 打开 `<dailyDir>/<today>.md`
- `arxiv-daily: Show run state` — 弹出最近 N 天的状态总览

### Ribbon icon

- 单击 = Run now
- 右键 / 长按（可选）= 弹命令菜单

---

## 7. 跨平台

- 全部 IO 走 Obsidian Vault Adapter，不直接调 Node `fs`
- HtmlCache 用 Electron `app.getPath('userData')`，跨平台自动隔离，不污染 vault
- 调度用 `setInterval` + Date 计算，三端一致
- 路径用 `normalizePath()`（Obsidian API）处理分隔符

---

## 8. 错误处理与可观察性

- **Notice**：每次运行（成功/失败）弹 Obsidian Notice，包含 `(date, status, papersWritten)`
- **状态视图**：命令 `Show run state` 显示最近 N 天的 RunState 表
- **日志**：`console.log/warn/error` + 可选写到 plugin data 下的 `arxiv-daily.log`（按 logLevel 过滤）
- **LLM 错误**：流式调用 + 自动重试（5s / 10s / 20s），与 Python 脚本一致
- **HTTP 重试**：指数退避，404 等不可恢复直接跳过

---

## 9. 测试策略

- **单元测试**（Jest 或 Vitest）：
  - HTML 解析器：拿真实 `/recent` 快照（fixture）测试按日期切片
  - State 状态机：覆盖所有 transition
  - Scheduler 决策逻辑：mock time，验证 tick 行为
- **集成测试**：
  - Pipeline end-to-end，mock LLM 客户端，验证写盘内容
- **手动测试**：
  - 在真实 Obsidian + 真实 DeepSeek API 上跑一次
  - 验证设置 UI 双向绑定
- 不写 E2E 自动化（Obsidian 插件 E2E 框架成熟度有限）

---

## 10. 构建与发布

- **模板**：基于官方 `obsidian-sample-plugin`（esbuild + TypeScript）
- **目录布局**：
  ```
  arxiv-daily/
    arxiv_daily.py            # 保留，不删
    plugin/                   # ← 新增
      main.ts
      manifest.json
      versions.json
      src/
        settings/
        services/
        pipeline/
        utils/
      tests/
      esbuild.config.mjs
      tsconfig.json
      package.json
  ```
- **打包产物**：`main.js` + `manifest.json` + `styles.css`（可选）
- **安装方式**（v1）：
  - 手动复制到 `<vault>/.obsidian/plugins/arxiv-daily/`
  - 或通过 BRAT 安装
- 提交到 Obsidian Community Plugins 留到 v1 稳定后

---

## 11. v2 预留

- `PluginSettings` 升级到 `{ profiles: ProfileSettings[]; activeProfileId: string }`
- 现有的 v1 `arxiv` / `output` / `schedule` 字段抽成 `ProfileSettings`
- `llm` / `advanced` 留全局
- 命令 `Run now` 等加 profile 参数
- 旧 settings 自动迁移成 default profile

---

## 12. 风险

| 风险 | 缓解 |
|---|---|
| arXiv 改版 `/recent` 页面结构 | 解析器写得宽松；保留 raw HTML fallback；日志详细记录 |
| Obsidian 关闭时正在跑 | Pipeline 按阶段原子写盘；state 是最后一步；下次 tick 重试 |
| LLM token 超额 | sectionCharLimit / paperCharLimit / dailyCharLimit 限制，已沿用 Python 配置 |
| 多端同时打开 Obsidian | v1 state 不同步，两台都会跑一遍（不致命，可接受） |
| DeepSeek 模型 ID 改名（2026/07/24 弃用 `deepseek-chat` 等） | 默认值用 `deepseek-v4-pro`；model 字段是自由文本，用户可改 |

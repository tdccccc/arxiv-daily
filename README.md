# arxiv-daily

arXiv 每日论文自动追踪，用 LLM **语义筛选**与你研究兴趣相关的新论文，生成中文日报和单篇详细报告。

提供两种运行方式：

| 方式 | 适合谁 | 特点 |
|---|---|---|
| **Obsidian 插件**（推荐） | 把笔记放在 Obsidian 里的研究者 | 设置 GUI、catch-up 调度、ribbon 一键触发、按 ID 单篇总结、跨平台 |
| **Python 脚本** | 用 cron / 服务器 / 命令行工作流的用户 | 单文件，靠 `.env` 和 crontab 跑，无 GUI |

两者输出的 Markdown 格式一致：`daily/YYYY-MM-DD.md`（日报）+ `papers/YYMM.NNNNN.md`（详细报告）。

---

## Obsidian 插件（v0.1.1）

### 功能

- **默认关闭**，首次安装不会自动跑，需手动启用
- **任意 arXiv 分类**：分组下拉选择（物理/计算机/数学/统计），也支持自定义输入
- **catch-up 调度**：Obsidian 打开时定时检查 `/list/<cat>/recent`，自动补跑过去 5 天内未完成的日期
- **跳过已有文件**：已有 daily 或 paper 文件时自动跳过，不消耗 LLM 调用
- **状态栏进度**：实时显示当前阶段和论文计数
- **Atom API 摘要补全**：listing 不再提供 abstract，插件二段抓 Atom API 拿全文摘要供 LLM 筛选
- **多厂商 LLM 预设**：DeepSeek / OpenAI / Anthropic / GLM 下拉选择，自动填充 URL 和模型，所有字段仍可手动修改
- **Ribbon 菜单**：Enable/Disable 开关 / Run for today / Run all pending / Run for specific date / **Summarize by arXiv ID**
- **状态可视化**：命令面板 `Show recent run state` 查看最近 N 天每天的运行状态
- **跨平台**：Windows / macOS / Linux

### 安装

**Option A — BRAT（推荐）：**

1. 在 Obsidian Community Plugins 装 [BRAT](https://github.com/TfTHacker/obsidian42-brat)
2. BRAT settings → Add Beta plugin → 输入 `tdccccc/arxiv-daily`
3. 启用 **arXiv Daily**
4. Settings → arXiv Daily → 填 API Key（默认 endpoint 已经是 DeepSeek）

**Option B — 手动：**

从 [latest release](https://github.com/tdccccc/arxiv-daily/releases) 下载 `manifest.json` / `main.js` / `styles.css`，扔进 `<vault>/.obsidian/plugins/arxiv-daily/`。

**Option C — 源码构建：**

```bash
git clone https://github.com/tdccccc/arxiv-daily.git
cd arxiv-daily/plugin
npm install
npm run build
# 然后照 Option B 把三个文件复制到 vault
```

### 设置概览

| Section | 字段 |
|---|---|
| Enable | 开关，显示 Running / Paused 状态 |
| LLM | Provider 下拉（DeepSeek/OpenAI/Anthropic/GLM/Custom）、API Key、Base URL、Model、Temperature、Timeout、Thinking mode、Reasoning effort |
| arXiv | 分类下拉（按领域分组）、研究兴趣、详细收录标准、详细分类（逗号分隔）、时区下拉 |
| Output & Schedule | Daily / Papers 路径、调度时间、tick 间隔、lookback 天数 (≤5) |
| Advanced | 请求间隔、缓存 TTL、字符限制、跳过/优先 sections、日志级别 |

### 命令 & Ribbon

| Command | 行为 |
|---|---|
| `arXiv Daily: Run now (today)` | 拉今日，写 daily + papers |
| `arXiv Daily: Run for date…` | 拉指定日期（5 天窗口内） |
| `arXiv Daily: Run all pending in lookback window` | 跑窗口内所有未完成日期 |
| `arXiv Daily: Summarize by arXiv ID…` | 用 arxiv id 单篇总结，写 papers/ |
| `arXiv Daily: Open today's daily report` | 打开 `<dailyDir>/<today>.md` |
| `arXiv Daily: Show recent run state` | 查看最近 20 天状态 |

ribbon 单击会弹菜单（Status + Enable/Disable 开关 / Run today / Run all pending / Run for date / Summarize by ID）。

### 调度模型

Catch-up 循环（默认每 20 分钟）只在 Obsidian 打开时运行：

- 插件默认**关闭**，需手动在设置或 ribbon 菜单中启用
- 每个 tick 走过 lookback 窗口（今天、昨天、…、4 天前）
- 跳过已完成 / 永久失败 / 正在运行的日期
- 今天若早于 `runAtLocal` 也跳过（避免抢跑）
- 周末（Sat/Sun）自动跳过，不消耗 LLM
- 已有 daily 文件的日期直接跳过（不抓取、不调 LLM）
- 失败_transient 会在 tick 间隔后重试
- 手动触发绕过时间门
- 启用时立即触发一次 today-only 总结
- 状态栏实时显示进度（日期、阶段、论文计数）

**含义：** Obsidian 必须每天打开至少一次（且过了 `runAtLocal`），插件才能跑当天。如果连续多天没开，超出 5 天窗口的日期 arXiv `/recent` 也拿不到了。需要离线/服务器跑的话用下面的 Python 脚本。

更多细节见 [`plugin/README.md`](./plugin/README.md)。

---

## Python 脚本（适合 cron / 服务器）

旧版本，单文件 `arxiv_daily.py`。Obsidian 插件没出现前的实现，仍然维护。

### 工作流程

1. 北京时间 9:30 起轮询 arXiv `<cat>/new`，等待当日更新
2. 解析所有新论文的标题 + 摘要
3. LLM 一次性筛选相关论文，标记分类 (category) 和是否详细收录 (detail)
4. 对筛选出的论文抓取 HTML 全文内容（带本地缓存）
5. 生成日报 → `daily/YYYY-MM-DD.md`
6. 对详细收录论文生成单独报告 → `papers/YYMM.NNNNN.md`

### 快速开始

```bash
pip install requests beautifulsoup4 pytz openai python-dotenv

cp .env.example .env
# 编辑 .env，填入 API Key 和研究兴趣

python arxiv_daily.py
```

### 配置

所有配置通过 `.env` 文件管理：

| 变量 | 说明 | 默认值 |
|---|---|---|
| `LLM_API_KEY` | LLM API Key | **必填** |
| `LLM_BASE_URL` | API 端点 | `https://api.openai.com/v1` |
| `LLM_MODEL` | 模型名称 | `gpt-4o` |
| `LLM_TEMPERATURE` | 生成温度 | `0.3` |
| `LLM_TIMEOUT` | LLM 请求超时（秒） | `300` |
| `WORK_DIR` | 输出目录 | `./output` |
| `RESEARCH_INTERESTS` | 研究兴趣描述 | 有默认值 |
| `DETAIL_CRITERIA` | 详细收录标准 | 有默认值 |
| `CATEGORY_TAG_MAP` | 分类→标签映射 (JSON) | 有默认值 |
| `CATEGORY_DISPLAY_MAP` | 分类→日报显示名称 (JSON) | 有默认值 |
| `REQUEST_DELAY` | arXiv 请求间隔（秒） | `3` |
| `POLL_INTERVAL` | 轮询间隔（秒） | `1800` |
| `MAX_RETRIES` | 最大轮询次数 | `16` |
| `LOG_LEVEL` | 日志级别 | `INFO` |
| `LOG_FILE` | 日志文件路径 | 脚本目录下 `arxiv_daily.log` |
| `CACHE_DIR` | 缓存目录 | 脚本目录下 `.cache/` |
| `CACHE_EXPIRY_DAYS` | 缓存过期天数 | `7` |

多行文本在 `.env` 中用双引号包裹即可直接换行，参见 `.env.example`。

### 自动运行 (crontab)

```bash
crontab -e
```

```cron
0 9 * * 1-5 /path/to/python /path/to/arxiv_daily.py >> /dev/null 2>&1
```

### 与插件版的差异

| | 插件 (v0.1.0) | Python 脚本 |
|---|---|---|
| 触发 | catch-up loop + 手动 | crontab + 9:30 轮询 |
| arXiv endpoint | `/recent` + Atom API | `/new` |
| Abstract 来源 | Atom API 富化 | listing 自带（部分）+ /abs fallback |
| 设置 | Obsidian GUI | `.env` |
| 状态持久化 | 插件 data | 每次启动从 0 |
| 多端协作 | 每台机器独立 state | 取决于你 cron 怎么布置 |
| Lookback / 补跑 | 5 天 rolling | 仅当天 |
| 按 ID 单篇 | ribbon 菜单 | 无 |

---

## 许可证

[MIT](./LICENSE)

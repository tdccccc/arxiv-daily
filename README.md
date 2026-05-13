# arxiv-daily

> 让每天最相关的 arXiv 论文，自动出现在你的 Obsidian 里。

不用再每天手动刷 arXiv 列表。设置好你的研究主题，Obsidian 打开时插件自动：

- 抓取当天新论文
- 用 LLM 筛掉跟你无关的
- 按主题分类生成中文日报
- 对核心相关论文额外生成一份详细解读

---

## 效果预览

每天打开 Obsidian，vault 里会多出一个文件 `arxiv-daily/daily/2026-05-13.md`：

```markdown
# arXiv astro-ph 每日追踪 2026-05-13
共 8 篇相关论文，其中 3 篇详细收录。

## Photo-z 相关

### Improved Photometric Redshifts from Multi-Survey Cross-Calibration → [[2605.12345]]
- **作者**: Y. Zhang et al.
- **arXiv**: [2605.12345](https://arxiv.org/abs/2605.12345)
- **一句话总结**: 利用 DES、HSC、LSST 三套巡天交叉标定，photo-z 精度提升约 15%
- **数据**: DES Y6 + HSC PDR3 + LSST DP0 模拟样本，约 200 万星系...
- **方法**: 改进的 SED 模板拟合 + 跨巡天系统误差联合标定...
- **主要结果**: σ_NMAD = 0.018（vs. 单巡天 0.022）；catastrophic outlier 比例降至 2.1%...
- **意义**: 为 LSST 第一年数据准备了系统误差校正流程...

## Galaxy Cluster 相关

### SZ-Selected Cluster Survey at z > 1.5 → [[2605.12350]]
- **作者**: A. Smith et al.
- ...

## ML in Astro
今日无相关论文更新。
```

带 `[[2605.12345]]` 链接的论文，会同时在 `arxiv-daily/papers/2605.12345.md` 生成完整解读（**背景与动机 / 数据 / 方法 / 结果 / 讨论 / 结论** 六个章节）。

---

## 主要功能

- **按你的研究主题筛选** —— 用一句自然语言描述每个研究方向（"photo-z 方法、目录、比较"），LLM 自动判断当天哪些论文属于哪个主题
- **日报 + 单篇深度报告** —— 日报里每个主题一节；标记为"详细收录"的主题，会对核心贡献论文额外生成完整解读
- **多 LLM 厂商内置预设** —— DeepSeek / OpenAI / Anthropic / GLM 一键切换，也支持任何 OpenAI 兼容的端点
- **catch-up 调度** —— 每次打开 Obsidian 自动补跑过去 5 天内漏掉的，不必每天都开着
- **省 token** —— 周末自动跳过、已生成的日报不重跑、不相关的论文不展开摘要
- **一键单篇** —— 知道某篇 arxiv ID，弹窗直接出详细解读
- **跨平台** —— Windows / macOS / Linux 都跑

---

## 安装

> 需要先装 [Obsidian](https://obsidian.md/download)（桌面版）。

### 1. 装 BRAT 插件

在 Obsidian 里 **Settings → Community plugins → Browse**，搜索 `BRAT`，Install → Enable。

（仓库主页：[obsidian42-brat](https://github.com/TfTHacker/obsidian42-brat)）

### 2. 通过 BRAT 装 arxiv-daily

**BRAT 设置面板 → Add Beta plugin →** 粘贴：

```
tdccccc/arxiv-daily
```

回 Community Plugins 启用 **arXiv Daily**。

> 不想用 BRAT 也行：从 [Releases](https://github.com/tdccccc/arxiv-daily/releases) 下最新版的 `manifest.json` / `main.js` / `styles.css` 三个文件，扔进 `<vault>/.obsidian/plugins/arxiv-daily/` 重启 Obsidian 即可。

---

## 第一次配置

打开 **Settings → arXiv Daily**，从上往下走：

### 1. 选 LLM 服务商 + 填 API Key

在 **LLM** 段：

- **Provider** 下拉选一个（DeepSeek / OpenAI / Anthropic / GLM / Custom）—— Base URL 和默认 Model 会自动填好
- **API Key** 粘贴你的 key

如果用自建/代理端点，选 **Custom**，手动改 Base URL 和 Model。

### 2. 选 arXiv 分类

在 **arXiv** 段，**arXiv Category** 下拉按领域分组：物理 / 计算机 / 数学 / 统计。比如：

- 天体物理 → `astro-ph`
- NLP → `cs.CL`
- 机器学习 → `cs.LG`

下拉里没有的，右边输入框可以手填。

### 3. 设置研究主题

最快的方式：**Load Template** 下拉选一个预设（Astrophysics + ML / NLP / Computer Vision / Bioinformatics），点选后自动填一组示例主题，你只需要按需删改。

也可以点 **+ Add Topic** 手动加。每张主题卡片里：

- **Name** —— 日报里这一节的标题（"Photo-z 相关"）
- **Tag** —— 写进每篇论文 YAML 的 Obsidian `#tag`，从 Name 自动派生
- **Description** —— **重要**：自然语言写清楚什么样的论文应该归到这个主题，LLM 按这个分类
- **Detail report** —— 打开则该主题的核心贡献论文会额外生成深度解读

> 默认 topics 为空。**至少加一个主题**，插件才会调用 LLM；否则 scheduler 就算开了也不会干活。

### 4. 打开 Enable 开关

页面顶部 **Enable** toggle。点开后会弹窗问你：

- **Run today** —— 立即跑今天的，几分钟后就能看到第一份日报
- **Skip today** —— 不跑今天，等明天定时
- **Cancel** —— 不启用

之后保持 Obsidian 开着，每天到 `Run time`（默认 09:30 上海时间）自动出报告。

---

## 怎么触发

| 方式 | 怎么操作 |
|---|---|
| 自动每日 | Enable 打开，每天到 Run time 自动跑 |
| 立即跑今天 | 左侧 ribbon 图标 → **Run for today** |
| 补跑过去 N 天 | Ribbon → **Run all pending** |
| 指定日期 | 命令面板 (`Cmd/Ctrl+P`) → **arXiv Daily: Run for date…** |
| 按 arXiv ID 单篇 | Ribbon → **Summarize by arXiv ID…** —— 弹窗粘贴 `2605.12345` 或完整 URL |
| 打开今日日报 | 命令面板 → **arXiv Daily: Open today's daily report** |

---

## 注意事项

- **要保持 Obsidian 开着** —— 插件只在 Obsidian 运行时跑。完全离开几天，超出 5 天窗口的就拿不到了（arXiv `/recent` 限制）
- **每个 vault 独立** —— 多台机器同步同一个 vault 时，可能两台机都会跑同一天（输出一致，但浪费一次 LLM 调用）
- **token 成本** —— 8-15 篇论文 + 3 篇深度报告大约一两毛钱（看模型），整体不算贵
- **手机暂不支持** —— 用了 Electron 文件系统，目前 desktop-only

---

## 高级选项

设置面板的 **Output & Schedule** 和 **Advanced** 段还有更多调参（输出路径、调度时间、字符限制、跳过/优先 section、日志级别等）。每个字段旁边的 `?` 图标鼠标悬停看说明。

---

## 命令行版本（cron / 服务器）

如果不想开 Obsidian，只想 cron 跑 + 文件夹里出报告，根目录的 `arxiv_daily.py` 单文件脚本也能用：

```bash
pip install requests beautifulsoup4 pytz openai python-dotenv
cp .env.example .env       # 填 API Key 和研究兴趣
python arxiv_daily.py
```

详细配置项见 `.env.example`。crontab 示例：

```cron
0 9 * * 1-5 /path/to/python /path/to/arxiv_daily.py
```

> 注：Python 版本是早期实现，功能不如插件完整（没有 topic 卡片、template、按 ID 单篇等），适合纯 headless 场景。

---

## 反馈 & 贡献

- Bug / 需求：[GitHub Issues](https://github.com/tdccccc/arxiv-daily/issues)
- 实现细节、开发文档、架构说明：见 [`plugin/README.md`](./plugin/README.md)

## License

[MIT](./LICENSE)

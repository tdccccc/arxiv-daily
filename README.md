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
> 信息来源：Abstract, Results, Conclusion
- **作者**: Y. Zhang et al.
- **arXiv**: [2605.12345](https://arxiv.org/abs/2605.12345)
- [ ] 关注 <!-- arxiv-daily:2605.12345:watch -->
- [ ] 重点 <!-- arxiv-daily:2605.12345:highlight -->
- **核心问题**: 多巡天 photo-z 训练集系统差异会把红移估计误差带入 LSST 早期样本。
- **关键方法**: 利用 DES、HSC、LSST 三套巡天交叉标定，联合约束模板拟合中的系统误差。
- **主要结果**: σ_NMAD = 0.018（vs. 单巡天 0.022）；catastrophic outlier 比例降至 2.1%...
- **为什么值得看**: 结果直接约束 LSST 第一年 photo-z 系统误差校正流程。
- **局限或边界**: 原文未说明。

## Galaxy Cluster 相关

### SZ-Selected Cluster Survey at z > 1.5 → [[2605.12350]]
- **作者**: A. Smith et al.
- ...

## ML in Astro
今日无相关论文更新。
```

带 `[[2605.12345]]` 链接的论文，会同时在 `arxiv-daily/papers/2605.12345.md` 生成完整解读（**研究问题 / 方法设计 / 关键证据 / 主要结论 / 适用边界 / 一句话价值判断** 六个章节）。

---

## 主要功能

- **按你的研究主题筛选** —— 用一句自然语言描述每个研究方向（"photo-z 方法、目录、比较"），LLM 自动判断当天哪些论文属于哪个主题
- **日报 + 单篇深度报告** —— 日报里每个主题一节；标记为"详细收录"的主题，会对核心贡献论文额外生成完整解读
- **日报内直接挑选** —— 在日报中勾选"关注"或"重点"，插件会自动同步到 `papers.json`，不用再去 inbox 里二次整理
- **Reading Dashboard** —— 在 Obsidian 里跨日期查看关注、重点、阅读中、已收藏、已读、忽略论文，支持搜索、筛选、汇总、Zotero 字段维护和打开、手动下载 PDF、追加到项目笔记和批量改状态
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
| 标记论文状态 | 论文详情页里用命令面板 → **arXiv Daily: Mark current paper as saved/read/ignored** |
| 回看论文列表 | 命令面板或 ribbon → **Open reading dashboard** |
| 打开今日日报 | 命令面板 → **arXiv Daily: Open today's daily report** |
| 复制引用片段 | 命令面板 → **arXiv Daily: Copy citation snippet for current paper…** 或 **Copy citation snippet by arXiv ID…** |

---

## 注意事项

- **日报仍是 markdown** —— 每天继续生成 `arxiv-daily/daily/YYYY-MM-DD.md`。在日报里勾选"关注"/"重点"会自动同步论文状态；论文级状态和去重记录在隐藏的 `arxiv-daily/.index/papers.json`，日常回看用 Reading Dashboard，只有 detail / saved / 手动创建的论文会有单篇 md。
- **不替代 Zotero 或 PDF 阅读器** —— v0.1.8 只维护 `citationKey`、`zoteroKey`、`zoteroUri`、`pdfPath` 和项目笔记链接，并从 Dashboard 打开这些外部资源；Better BibTeX / Zotero local API 自动同步留作后续增强。
- **要保持 Obsidian 开着** —— 插件只在 Obsidian 运行时自动跑；手动指定超出 `/recent` 5 天窗口的日期时，会用 arXiv export API 的 submittedDate 单日窗口近似补跑，并在日报中标注。
- **每个 vault 独立** —— 多台机器同步同一个 vault 时，可能两台机都会跑同一天（输出一致，但浪费一次 LLM 调用）
- **token 成本** —— 8-15 篇论文 + 3 篇深度报告大约一两毛钱（看模型），整体不算贵
- **手机暂不支持** —— 用了 Electron 文件系统，目前 desktop-only

---

## 高级选项

设置面板的 **Output & Schedule** 和 **Advanced** 段还有更多调参（输出路径、调度时间、字符限制、跳过/优先 section、日志级别等）。每个字段旁边的 `?` 图标鼠标悬停看说明。

---

## 命令行版本（cron / 服务器）

如果不想开 Obsidian，只想 cron 跑 + 文件夹里出报告，使用 `plugin/` 里的 Node CLI。它复用 Obsidian 插件的同一套 core pipeline、配置 schema 和 `arxiv-daily/.index/run-state.json`：

```bash
cd plugin
npm install
npm run build

ARXIV_DAILY_API_KEY=sk-... npm run cli -- run-pending --vault-root /path/to/vault
```

也可以放一个 `arxiv-daily.config.json`，再运行：

```bash
npm run cli -- run --date 2026-06-13 --config arxiv-daily.config.json --vault-root /path/to/vault
npm run cli -- summarize --id 2606.12345 --config arxiv-daily.config.json --vault-root /path/to/vault
```

crontab 示例：

```cron
0 9 * * 1-5 cd /path/to/arxiv-daily/plugin && ARXIV_DAILY_API_KEY=sk-... npm run cli -- run-pending --vault-root /path/to/vault
```

> 注：根目录 `arxiv_daily.py` 已退役为兼容 shim，只转发到 Node CLI，不再维护独立 Python pipeline。

---

## 反馈 & 贡献

- Bug / 需求：[GitHub Issues](https://github.com/tdccccc/arxiv-daily/issues)
- 实现细节、开发文档、架构说明：见 [`plugin/README.md`](./plugin/README.md)

## License

[MIT](./LICENSE)

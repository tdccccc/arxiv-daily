# 新手教程

在 Obsidian 里跑通第一份 **日报（Daily report）**，再按需打开定时与邮件。

产品总览见 [中文说明](README.zh-CN.md)。下面的分步教程以 **Obsidian 插件**为主；不使用 Obsidian 时也能从同一套日报与阅读流程开始。

## 选择使用入口

| 入口 | 开始方式 |
|---|---|
| Obsidian 桌面插件 | 安装并启用插件，按下方步骤设置 |
| 独立浏览器工作台 | 从源码构建 CLI，运行 `npm run cli -- ui`；见 [CLI 文档](../apps/cli/README.md#local-reading-workbench) |
| DSH | 安装本地 `0.1.18` 包，在左栏“设置”上方点击 `arxiv-daily`；见 [构建、安装与升级](../extensions/dsh-arxiv-daily/README.md) |

独立工作台首次打开时，配置保存目录、模型、arXiv 分类和研究主题。**设置修改后自动保存**：开关和下拉选择立即保存，文本输入稍停后保存，底部显示保存进度或错误。“完成”、× 和 Esc 会等待保存完成再关闭；失败时可重试，或明确放弃未保存的修改。关闭“每日发现”后，再打开设置仍保持关闭。完成首份日报后，是否开启自动日报由你决定。

左侧是日历与筛选，右侧是论文列表和 Markdown 阅读器，支持公式、前进/后退、待读与收藏。点击“生成”开始生成日报；“设置 → 外观”可切换主题和中英文界面，界面语言与报告语言独立。Agent 对话不是使用这些功能的前提。

DSH 包目前通过本地源码构建分发，未发布到 npm。工作台与 Obsidian 共用核心流程和设置规则，各宿主配置值、密钥独立保存。先在设置中连接文献目录并建立索引，再从左侧“个人文献库”浏览、检索及打开 PDF；“方向审核”可查看依据、编辑候选、预览并接受到研究主题。更完整的运行管理仍在后续计划中。

## 你需要准备

- Obsidian **桌面版**
- LLM 的 API key（以及需要时的 Base URL / 模型）
- 一个或多个 arXiv 分类（如 `astro-ph`、`cs.LG`、`hep-th`）
- 每个研究主题的简短描述

生成内容默认写在 vault 的 `arxiv-daily/` 下。API key 保存在本机插件数据里；保存后显示 **Configured**，修改用 **Replace** / **Clear**。

## 1. 打开设置

安装并启用 **arXiv Daily** 后打开：

```text
Settings → arXiv Daily
```

顶部有五步引导：

1. **Connect AI（连接 AI）**  
2. **Choose paper sources（选择论文来源）**  
3. **Describe your research interests（描述研究兴趣）**  
4. **Generate your first report（生成第一份报告）**  
5. **Turn on daily reports（开启定时日报）**

按钮会跳到对应表单；五步全部完成后，Obsidian 会记住引导已完成，即使后来关闭定时也不会重新弹出。不开启定时也可以手动生成日报。

## 2. 连接 AI

选择 provider，填写并保存 API key。按需改 Base URL 和模型。第一次可先用默认高级选项。

## 3. 选择 arXiv 分类

勾选要抓取的分类，可多选；同一篇论文只会保留一份。

例子：`astro-ph`、`astro-ph.CO`、`cs.LG`。

## 4. 添加研究主题

每个 topic 对应 **日报里的一个章节**。

每个 topic 需要：

- **Name** — 章节标题  
- **Directions** — 每行一个具体研究方向，筛选时逐项匹配

例子：

```text
Name: Photometric Redshift
Directions:
Uncertainty calibration for photometric redshift estimation
Benchmarks and systematics of photometric redshift catalogs
```

可先用模板，再改成你的方向。文献库提出的方向需要审核接受才会加入普通主题；未接受候选不参与筛选。

**论文总结（可选加深）：**  
每个 topic 的 **Detail report** 表示：该主题下的论文是否有机会生成 `papers/` 里更长的 **论文总结**（不只是日报里的短条目）。列表下方的 **Automatic detail notes**（Fewer / Recommended / More）控制自动写总结的频率。之后仍可手动生成（例如 **Summarize by arXiv ID**）。

## 5. 生成第一份日报

在引导里点 **Generate first report**，或打开 **Dashboard** 点 **Run Today**。

插件会：

1. 按分类抓取近期论文  
2. 按主题筛出相关论文  
3. 写入 **日报**：每篇入选论文一段结构化短摘要  
4. 在允许时为少量论文生成 **论文总结**  
5. 更新 Dashboard  

日报路径：

```text
arxiv-daily/daily/YYYY-MM-DD.md
```

| 产出 | 路径 | 作用 |
|---|---|---|
| **日报** | `daily/YYYY-MM-DD.md` | 当天的阅读列表（主要结果） |
| **论文总结** | `papers/<arxiv_id>.md` | 单篇更长的总结 |

自动论文总结失败或跳过时，日报仍可成功。等待 arXiv 公告、当日无更新、筛选无匹配和真正失败会分别展示；等待公告不消耗普通失败重试额度。

## 6. 使用 Dashboard

设置完成后，日常从 Dashboard 进入：

- **Starred** / **All** — 关注你标星的论文  
- **日历** — 按日期打开日报  
- **搜索与筛选** — 在本地索引里找论文  
- **行操作** — 打开日报、论文总结、arXiv、PDF；加星  

需要进正式文献库时，可从行内打开 arXiv，再用 Zotero 等工具导入。

## 7. 打开定时

手动跑通后，在 **Settings → arXiv Daily** 启用 scheduler。

只在 **Obsidian 打开时**、按你配置的工作日时间窗口运行；漏掉的工作日之后还可能补上。

## 8. 可选：邮件

邮件是可选的。发信失败**不会**导致日报失败。

| 模式 | 你要做什么 |
|---|---|
| **自己发送**（默认） | 自备 [Resend](https://resend.com) API Key，无项目配额 |
| **官方代发 (Beta)** | 验证邮箱后由项目代发；共享免费额度，仅适合轻度个人使用 |

### 自己发送（快速）

1. 注册 Resend 并创建 API Key（`re_…`）。  
2. **Settings → arXiv Daily → Email delivery**  
   - How to send：**Send yourself**  
   - **Your email**：一般与 Resend **账号邮箱相同**  
   - 粘贴 API Key；**From email 留空**最简单  
3. **Send test**，检查收件箱/垃圾箱。  
4. 测试成功后再打开 **Daily auto-send**。

From 留空时，测试发件地址通常**只能发到 Resend 账号邮箱**（GitHub 登录多为 GitHub **主邮箱**）。要发给其它地址，需在 Resend 验证域名并填写自定义 From。

### 官方代发 (Beta)

1. 选择 **Official delivery (Beta)**。  
2. 填写邮箱 → **Send verification email**。  
3. 打开链接，粘贴网页上的**长验证码**（不是链接里的短参数）。  
4. **Send test**，成功后再开 **Daily auto-send**。  
5. 触达当日额度则等下一个 UTC 日，或改用自己发送。

Beta 额度偏小（每个已验证邮箱每个 UTC 日仅少量消息，测试也计入）。量大请用自己发送。

### 真·日报之后

打开自动发送后，某日 run **completed** 才可能发一封摘要；默认同一天不重发。**Send test** 不会挡住当天的正式日报邮件。

## 常见问题

| 情况 | 可尝试 |
|---|---|
| **Run Today** 不可用 | 完成 Settings → arXiv Daily 顶部清单 |
| Dashboard 没有论文 | 先成功跑一次今天（或其它日期） |
| 运行失败 | Dashboard → More → Show diagnostics |
| 入选论文太多 | 减少分类，或把 topic 描述写得更具体 |
| 测试邮件 HTTP **403** | **Your email** 改成报错里的 Resend 账号邮箱；验证域名前 From 留空 |

## CLI（可选）

若希望不打开 Obsidian 也能出日报（Node.js 20.19+）：

```bash
npm install -g arxiv-daily
arxiv-daily init          # 交互向导；Enter 保留默认值
arxiv-daily run --today
```

配置在 `~/.config/arxiv-daily/config.toml`。卸载：`npm uninstall -g arxiv-daily`（配置和 vault 不会自动删）。Windows 上定时建议用 **WSL**，或继续用插件。详见 [CLI README](../apps/cli/README.md) · [0.3.4 说明](releases/0.3.4.md)。

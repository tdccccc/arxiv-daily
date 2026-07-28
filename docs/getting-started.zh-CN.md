# 新手教程

在 Obsidian 里跑通第一份 **日报（Daily report）**，再按需打开定时与邮件。

产品总览（插件 + CLI）见 [中文说明](README.zh-CN.md)。本页只讲 **Obsidian 插件**。

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

顶部有四步引导：

1. **Connect AI（连接 AI）**  
2. **Choose paper sources（选择论文来源）**  
3. **Describe your research interests（描述研究兴趣）**  
4. **Generate your first report（生成第一份报告）**  

按钮会跳到对应表单。第一份报告完成前会显示完整引导；完成后变成简短的「设置完成」（配置失效时完整引导会回来）。

## 2. 连接 AI

选择 provider，填写并保存 API key。按需改 Base URL 和模型。第一次可先用默认高级选项。

## 3. 选择 arXiv 分类

勾选要抓取的分类，可多选；同一篇论文只会保留一份。

例子：`astro-ph`、`astro-ph.CO`、`cs.LG`。

## 4. 添加研究主题

每个 topic 对应 **日报里的一个章节**。

每个 topic 需要：

- **Name** — 章节标题  
- **Tag** — 短标签  
- **Description** — 用自然语言说明哪些论文算这个主题  

例子：

```text
Name: Photometric Redshift
Tag: photo-z
Description: Methods, benchmarks, uncertainty calibration, catalog construction,
and systematics for photometric redshift estimation.
```

可先用模板，再改成你的方向。

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

自动论文总结失败或跳过时，日报仍可成功。

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

若希望不打开 Obsidian 也能出日报，见 [中文说明里的 CLI 一节](README.zh-CN.md#cli)：先 `init`，再 `run --today`，配置在 `~/.config/arxiv-daily/config.toml`。Windows 上定时建议用 **WSL**，或继续用插件。

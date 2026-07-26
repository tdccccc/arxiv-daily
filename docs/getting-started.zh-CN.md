# 新手教程

这份教程用于完成 arXiv Daily 在 Obsidian 里的第一次成功运行。

## 开始前

你需要准备：

- Obsidian 桌面端。
- 一个 LLM provider 的 API key。
- 一个或多个 arXiv 分类，例如 `astro-ph`、`cs.LG`、`hep-th`。
- 你希望跟踪的研究主题描述。

arXiv Daily 会把生成内容写入你的 vault。为保持兼容，API key 以明文保存在 `<你的-vault>/.obsidian/plugins/arxiv-daily/data.json`，不是 keyring 或加密存储；vault 同步或备份工具可能复制该文件。保存后设置页只显示 **Configured**，修改或删除需显式使用 **Replace** / **Clear**；日志、诊断和展示给用户的错误会做脱敏。抓取的 source 内容会按设置的保留时间（默认 7 天）缓存在相邻的 `.cache/` 目录；如需清除，请先禁用插件再删除该目录。

## 1. 打开插件设置

安装并启用插件后，打开：

```text
Settings -> arXiv Daily
```

设置页顶部有一个无障碍的四步 **Getting started** 引导：

1. **Connect AI（连接 AI）**
2. **Choose paper sources（选择论文来源）**
3. **Describe your research interests（描述研究兴趣）**
4. **Generate your first report（生成第一份报告）**

引导会显示四步中已完成的数量。只有待完成且可操作的步骤才显示按钮；按钮会跳转到现有的设置表单，不会重复创建 provider、分类或 topic 输入。配置有效后，最后一步会调用现有的 run-now 命令。详细校验原因保留在可展开的 **Configuration details** 中。

在第一份报告完成前，完整引导会一直显示。报告完成且配置有效后，设置页改为紧凑的 **Setup complete** 摘要，显示最近完成的报告日期和 **Open dashboard**。如果之后配置失效，完整引导会重新出现。

## 2. 配置 LLM

先选择 provider，然后填写并保存 API key。保存后的 key 不会重新渲染到页面，而是显示 **Configured** sentinel；需要修改或删除时使用 **Replace** 或 **Clear**。Provider preset 会自动填入 base URL 和 model，但这两个字段仍然可以手动修改。

第一次运行时，temperature、timeout、reasoning 等高级设置可以先保持默认，除非你的 provider 明确要求修改。

## 3. 选择 arXiv 分类

选择你要抓取的 arXiv 分类。可以选多个分类，重复论文会按 arXiv ID 合并。

例子：

- `astro-ph`：天体物理。
- `astro-ph.CO`：宇宙学。
- `cs.LG`：机器学习。

## 4. 添加研究主题

每个 topic 会变成日报里的一个章节。

一个 topic 需要：

- **Name**：日报里的章节标题。
- **Tag**：短的 Obsidian tag slug。
- **Description**：自然语言描述，说明什么论文应该归到这个 topic。

例子：

```text
Name: Photometric Redshift
Tag: photo-z
Description: Methods, benchmarks, uncertainty calibration, catalog construction, and systematics for photometric redshift estimation.
```

如果模板里有接近你方向的配置，可以先加载模板，再按自己的研究方向修改。

每个 topic 还有一个 **Detail report** toggle。它不影响相关论文是否进入日报并获得结构化总结，只决定该 topic 下的论文是否有资格自动生成 `papers/` 中的独立 deep dive。

Topic 列表下方的 **Automatic detail notes** 用于控制自动生成独立详细笔记的严格程度，只需选择 **Fewer（更少）**、**Recommended（推荐）** 或 **More（更多）**；默认是 **Recommended**。已有 Custom 策略会继续生效，直到你主动选择其他选项；高级自定义阈值仍可通过持久化配置或 CLI 设置。

## 5. 第一次运行

在设置引导中点击 **Generate first report**。也可以从左侧 ribbon 图标或命令面板打开 **arXiv Daily Dashboard**，再点击 **Run Today**。插件会依次做这些事：

1. 按配置的分类抓取 arXiv 近期论文。
2. 根据你的 topic 筛选相关论文。
3. 抓取可用全文；仅在存在 eligible deep-dive candidates 时，额外调用一次 LLM 对它们统一评分。
4. 为所有入选论文生成结构化 Markdown 日报总结。
5. 为自动选中的论文创建独立的 `papers/<arxiv_id>.md` deep dive。
6. 更新 Dashboard 索引。

生成的日报默认在：

```text
arxiv-daily/daily/YYYY-MM-DD.md
```

需要区分两种输出：

- `daily/YYYY-MM-DD.md` 包含当天每篇入选论文的结构化总结，是完整的每日阅读清单。
- `papers/<arxiv_id>.md` 是可选的单篇全文 deep dive，与日报短总结相互独立。

Deep-dive 评分发生在全文抓取之后。论文只有在所属 topic 启用了 **Detail report**、存在可用全文且尚无 paper 文件时才 eligible。没有候选时不会多出 selector 调用。如果评分调用失败或返回无效结果，系统不会创建新的自动 deep dive，但会继续日报总结，daily run 仍可成功。手动 **Summarize by arXiv ID** 不受影响。

## 6. 使用 Dashboard

设置完成后，Dashboard 就是主要入口。

- **Starred**：显示你标记为重点的论文。
- **All**：显示所有未忽略的历史论文。
- Search 完全在本地进行，按相关度检索 arXiv ID、标题、作者、topic、分类和结构化摘要字段；支持精确现代 arXiv ID、英文技术词和中文切词。有搜索词时默认按相关度排序，显式选择星标/发表日期/topic/标题排序后则保持该主排序。
- **Similar Papers**（论文行的 **Find similar papers** 操作）使用已持久化的 abstract 和从历史日报恢复的结构化总结，进行加权的多概念、跨字段词法匹配，并抑制弱匹配和仅作者匹配。结果会显示匹配原因、元数据、资源可用性和相应打开操作；查询本身完全在本地进行，不使用网络、LLM、embedding 或数据库。
- 右侧日历可以按日期打开日报。
- 每行操作可以打开/创建论文笔记、查找相似论文、打开来源日报、打开 arXiv、打开 PDF、下载 PDF；相似论文结果可打开 detail、日报、arXiv 页面或 PDF。
- **Dashboard -> More -> Cancel active tasks** 会协作式取消自动/手动日报运行、手动 detail 总结和 PDF 下载。**Get Models** 不在范围内；已经发出的 Obsidian `requestUrl` 请求可能先完成，后续工作才停止。

如果某篇论文要进入正式文献库，建议从 Dashboard 打开 arXiv 页面，然后用 Zotero 浏览器插件导入。

生成的日报和 detail 笔记末尾会有折叠的 **Generation metrics** callout，显示可用的 pipeline 总耗时、LLM 耗时、逻辑调用数、HTTP attempts 和 provider 报告的 tokens。缺失或因重试而不完整的 usage 会显示 unavailable/incomplete，不会记为 0；插件不估算费用。

Paper Index schema 3 会持久化 abstract，并可读取已有 schema 1/2 文件。旧条目会在后续日报再次遇到相应论文时惰性补齐 abstract，不会为了迁移而联网，也无需批量重写；已有 Markdown 仍可继续使用。

## 7. 启用自动运行

确认第一次手动运行成功后，回到 **Settings → arXiv Daily**，启用 scheduler。

Scheduler 只会在 Obsidian 打开时运行。lookback 窗口内漏掉的工作日会在之后补跑。

## 8. 可选：邮件日报

邮件是**可选**功能。产品规划为**双模式**（详见 helm `email-dual-mode.md`）：

| 模式 | 状态 | 你要做什么 |
|---|---|---|
| **自己发送** | **现已可用** | 自备 [Resend](https://resend.com) API Key，发到个人邮箱 |
| **官方代发 (Beta)** | **规划中 / 未上线** | 验证邮箱后由项目代发，用户无需 API Key — **尚未上线** |

默认是 **自己发送**，不依赖项目服务器。发信失败**不会**把当天日报标成失败。

### 自己发送（Resend，自备 API Key）

### 重要限制（快速配置 / 测试发件）

**From email 留空**时，插件使用 Resend 测试发件地址 `onboarding@resend.dev`：

- **To 几乎只能填 Resend 账号绑定的那个邮箱**（「本人邮箱」）。
- 若用 **GitHub 登录** Resend，通常是 GitHub 的**主邮箱（Primary）**，不是 GitHub 上挂的每一个邮箱。
- 发给其它地址会 403，直到你在 Resend **验证自己的域名**并填写自定义 From。

请把邮件当成**给自己的提醒通道**，不要默认当成群发/通知多人，除非完成域名验证。

### 快速配置（推荐）

1. 打开 [resend.com](https://resend.com) 注册（可用 GitHub）。
2. **API Keys → Create**，复制一次性显示的 `re_…`。
3. Obsidian：**Settings → arXiv Daily → Email delivery**：
   - **To**：填与 Resend 账号**相同**的邮箱（见上限制）。
   - **Resend API key**：粘贴并保存。
   - **From email**：**留空**。
4. 点 **Send test**，在该账号邮箱的收件箱/垃圾箱查看。
5. **测试成功后**，再打开 **Daily auto-send（每日自动发送）**。

API key 与 LLM key 一样保存在本地 `data.json`（磁盘明文；保存后设置页不再回显完整 key）。

### 真·日报跑完之后

开启 **Daily auto-send** 后，某日 run **completed** 才可能发一封真日报；默认同一天不重复发（见 vault 内 `arxiv-daily/.index/delivery-state.json`）。仅 repair 索引的完成**不会**发信。

### 进阶：发给其它收件地址

1. 在 Resend 添加并验证你拥有的域名（DNS 配置 SPF/DKIM）。
2. **From email** 填该域名下的地址。
3. 之后 **To** 才可改为任意邮箱（仍受 Resend 规则与配额约束）。

## 常见问题

如果 **Run Today** 是 disabled，先完成 **Settings → arXiv Daily** 顶部 checklist。

如果 Dashboard 显示还没有 indexed papers，先运行今天或运行 pending dates。

如果运行失败，用 **Dashboard → More → Show diagnostics** 查看设置、日期上下文和最近运行状态。

如果入选论文太多，缩小 arXiv 分类，或者把 topic description 写得更具体。

如果 **Send test** 报 HTTP **403**，且提示只能发给自己的邮箱：把 **To** 改成报错信息里写出的那个地址（Resend 账号邮箱）；不要用未绑定的次要 GitHub 邮箱；在验证域名之前 **From 保持留空**。


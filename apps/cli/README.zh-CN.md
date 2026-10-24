[English](https://github.com/tdccccc/arxiv-daily/blob/main/apps/cli/README.md) · 简体中文

# arXiv Daily CLI

按分类抓取 arXiv 论文，用 LLM 按你的研究主题筛选，生成 Markdown **日报** 和 **论文详细总结** —— 可在终端、本地浏览器或服务器/定时任务上运行。

属于 [arXiv Daily](https://github.com/tdccccc/arxiv-daily) 项目。Obsidian 插件是可选的独立组件，两者可以共用同一个 vault 目录。

## 功能

- **每日筛选** —— 只保留匹配你研究主题中描述方向的论文
- **本地浏览器工作台** —— 运行在 `127.0.0.1` 上的阅读器，不需要 Obsidian（`arxiv-daily ui`）
- **定时运行** —— Linux/macOS/WSL 上用 cron，或在工作台打开期间使用内置的定时检查
- **个人文献库** —— 索引你自己的 PDF，审核后接受为正式研究主题
- **可恢复的运行** —— 运行被中断后重试不会重复已付费的模型调用
- **邮件摘要（可选）** —— 用你自己的 Resend key，或使用官方托管发送

## 环境要求

- Node.js **20.19.0** 及以上

## 安装

```bash
npm install -g arxiv-daily
```

或不做全局安装，直接用 npx：

```bash
npx arxiv-daily@latest help
```

npm 包主页：https://www.npmjs.com/package/arxiv-daily

## 快速开始

```bash
arxiv-daily init        # 引导式设置：vault、模型、分类、研究主题……
arxiv-daily run --today
arxiv-daily ui           # 打开本地阅读工作台
```

`init` 是一个 TUI 向导——**↑/↓** 移动，**Space** 多选，**Enter** 确认/接受默认值，**Ctrl+C**/**Esc** 退出。流程依次是：vault 位置 → LLM 提供商/API key/模型 → 可选邮件 → arXiv 分类 → 时区 → 报告语言 → 研究主题 → 可选的详细总结与定时参数。如果配置文件已经存在，`init` 会先询问：**Keep existing**（默认；不改动文件并直接退出）、**Overwrite**（先把旧文件另存为 `config.toml.bak-<时间戳>`）或取消。

## 配置

| | |
|---|---|
| Linux/macOS | `$XDG_CONFIG_HOME/arxiv-daily/config.toml`（默认 `~/.config/arxiv-daily/config.toml`） |
| Windows | `%APPDATA%\arxiv-daily\config.toml` |

`init` 的默认 vault 路径是 `~/arxiv-daily`。没有配置用的环境变量，也没有 `--config`/`--vault-root` 参数。你可以直接手改这个文件；工作台设置或 `library connect` 修改它时，只会改写它们改动的值，注释和排版都会保留。配置文件以明文保存 API key，建议加权限限制：

```bash
chmod 600 ~/.config/arxiv-daily/config.toml
```

**每日论文数量上限** —— 新生成的日报里所有主题合计默认最多 20 篇。可在已有的 `[output]` 表中调整：

```toml
[output]
max_daily_papers = 20
```

0、负数、小数和带引号的字符串都会被拒绝；缺省时按 20 处理。改动这个值不会重写已经生成过的报告。

## 命令

| 命令 | 作用 |
|---|---|
| `init` | 引导式设置向导 |
| `status` | 以 JSON 显示输出路径、研究主题、模型就绪状态、论文数量和最近一次运行状态 |
| `ui [--port PORT] [--no-open]` | 打开本地阅读[工作台](https://github.com/tdccccc/arxiv-daily/blob/main/docs/workbench.zh-CN.md) |
| `papers [--query TEXT] [--offset N] [--limit N]` | 分页返回本地论文索引（默认 limit 30，最大 100） |
| `library …` | 连接并索引个人 PDF 文献库——见[下方](#可选的个人文献库) |
| `update [--check] [--yes]` | 检查 npm 上是否有新版本，可选择安装；`--check` 只报告不安装 |
| `run --today` | 生成今天的日报（典型的 cron 用法） |
| `run --date YYYY-MM-DD` | 生成（或重试）某一天的日报 |
| `run --id ARXIV_ID [--date YYYY-MM-DD]` | 为单篇论文生成详细总结 |
| `email test\|status\|verify-start` | 发送测试邮件、查看发送配置，或启动托管邮件验证 |
| `schedule show\|install\|uninstall` | 查看或管理 cron 任务（仅限 Linux/macOS/WSL） |
| `data export --out PATH.zip` / `data import PATH.zip [--yes]` | 备份或恢复 vault |
| `help` | 打印完整用法 |

输错的命令不会被当成别的操作执行：未知命令或选项、重复的选项、多余的参数，以及不存在的日期或 arXiv ID（`run --id` 需要形如 `2606.12345` 的 ID），都会打印用法并以退出码 2 结束。`data export` 不会写出 `data import` 因体积限制会拒绝的压缩包。

筛选时如果有候选论文没有摘要、只依据标题判断，`run` 完成时输出的那一行会写出数量（例如 `3/40 candidates judged by title only`）；日报页头和运行历史里也有同样的说明。

运行被中断后重试不会重复已付费的模型调用；如果需要强制重新生成，删除 `<output-root>/.index/*-checkpoints/<date>*`。

### 托管邮件

`email verify-start` 会向 `email.to` 发送验证链接。打开链接后，把页面上的长验证码粘贴到 `config.toml` 的 `[email]` 表的 `hosted_token`，并设置 `mode = "hosted"`。持续发送时验证码会自动续期，自签发或上次续期起一年后失效；如果 `email test` 报告 `hosted_token_rejected`，重新运行 `email verify-start` 并粘贴新的验证码。`email test` 对其他拒绝原因（`hosted_recipient_mismatch`、`provider_quota_exceeded`、`resend_api_key_rejected`、`resend_access_denied`）也会指出原因，并给出一句修复建议。

## 本地阅读工作台

```bash
arxiv-daily ui                   # 在默认浏览器中打开
arxiv-daily ui --port 8123       # 使用固定端口
arxiv-daily ui --no-open         # 只打印地址，不自动打开浏览器
```

只在 `127.0.0.1` 上提供服务。完整功能说明见 [docs/workbench.zh-CN.md](https://github.com/tdccccc/arxiv-daily/blob/main/docs/workbench.zh-CN.md)。

## 定时运行

```cron
30 9 * * 1-5  arxiv-daily run --today
```

或者在 `config.toml` 中配置 `[schedule]`，再让 CLI 接管：

```bash
arxiv-daily schedule install
```

`schedule install`/`uninstall` 会在 Linux/macOS/WSL 上写入受管理的 crontab 条目。条目使用执行 `install` 时的 Node 可执行文件运行 `<node> <脚本> run --today`，输出追加到配置文件旁的 `schedule.log`；升级或切换 Node 后请重新运行 `schedule install`。cron 不会读取 shell 启动文件，所以配置位于自定义 `XDG_CONFIG_HOME` 下时，条目会自带这个变量（移动配置后请重新运行 `schedule install`）。原生 Windows 的任务计划程序不支持；请使用 WSL，或改用 Obsidian 插件/工作台自带的定时功能。

## 可选的个人文献库

索引你自己的 PDF，匹配到的论文会和手动输入的研究主题一起参与日报筛选。访问权限是只读的，并且只限于你连接的那一个目录。

```bash
arxiv-daily library connect /path/to/pdfs
arxiv-daily library status                        # 查看范围和授权指纹
arxiv-daily library authorize --fingerprint HASH   # 对该指纹对应的确切范围授权
arxiv-daily library prepare                        # 安装可选的 PDF/嵌入运行时组件
arxiv-daily library scan
arxiv-daily library index                          # 若自上次扫描后增删或改名了 PDF，先自动重新扫描（只为新论文联网查询 arXiv）；目录没变则不扫描、不联网
arxiv-daily library propose                        # 生成方向候选
arxiv-daily library propose --new                  # 新增论文后：只把新论文连同现有方向发给模型
arxiv-daily library directions                     # 查看候选
arxiv-daily library confirm --candidate ID --proposal-revision N
# 也可以一次性查看/接受多个：
arxiv-daily library review --input REQUEST.json
arxiv-daily library revoke                         # 撤销模型处理授权
```

被接受的方向会变成普通的研究主题设置。`library search --query TEXT [--mode hybrid|lexical|dense] [--limit N]` 可以直接在已索引的文献库中检索。[工作台](https://github.com/tdccccc/arxiv-daily/blob/main/docs/workbench.zh-CN.md)也支持在浏览器里连接、索引、浏览文献库和审核方向。

### 回收旧的搜索索引

多次修改文献库后，可以回收不再使用的搜索索引。先关闭使用同一 vault 的其他所有 Obsidian 窗口、CLI 任务和工作台客户端，并在清理结束前保持关闭，然后运行：

```bash
arxiv-daily library maintain --other-clients-closed
```

（在 Obsidian 里改用命令面板中的 **Clean up old personal library search indexes…**。）清理会保留当前索引和备份索引，以及仍被读取方或待写入占用的版本。它不会自动运行，因为单个进程无法看到其他客户端的读取状态。早期嵌入模型留下的单篇论文文件会保留；只回收旧的搜索版本。

## 更新

```bash
arxiv-daily update
```

检查 npm 上是否有新版本的 `arxiv-daily`，可选择安装；不会改动你的配置。

## 卸载

```bash
npm uninstall -g arxiv-daily
```

这只会移除全局命令，**不会**删除 `~/.config/arxiv-daily/`（配置和密钥）或你的 vault/输出目录（例如 `~/arxiv-daily`）。如果需要彻底清理，请手动删除它们。

## 从 monorepo 开发

这个包由 [arxiv-daily](https://github.com/tdccccc/arxiv-daily) 工作区构建：

```bash
git clone https://github.com/tdccccc/arxiv-daily.git
cd arxiv-daily
npm ci
npm run build
npm run cli -- run --today    # 或构建后直接：npx arxiv-daily run --today
```

发布到 npm 的产物是单个打包好的二进制文件（`dist/arxiv-daily-cli.cjs`）。

## 许可证

MIT —— 见 [LICENSE](./LICENSE)。

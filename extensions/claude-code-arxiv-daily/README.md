# arXiv Daily for Claude Code CLI — 本地阅读工作台

独立运行 arXiv Daily 原有的论文筛选、日报与详细总结，再通过 Claude Code 辅助操作。**没有个人文献库也能开始使用。** 模型调用、筛选、总结、索引、运行状态和断点恢复由原有业务核心管理。

## 先设置产品，再用任意入口操作

此 worktree 中构建：

```sh
node extensions/claude-code-arxiv-daily/build.mjs
```

构建使用原有 CLI 构建脚本，包含本机原生存储模块。源码构建需要仓库开发依赖、CMake、C++ 编译器和 Node-API 头文件；多平台分发沿用原有 native release 构建要求。生成的 `dist/plugin/` 可整体复制，运行端只需 Node.js 20.19+。

首次使用，在交互式终端运行：

```sh
node /absolute/plugin/path/dist/arxiv-daily-cli.cjs init
```

设置输出目录、模型 API、arXiv 分类和研究主题。已经使用现有 arXiv Daily CLI 的用户直接复用其配置。这里不要求选择文献库，也不要求 Obsidian。

模型 API 使用产品自己的配置；Claude Code 的对话模型与它分开。API 密钥在本地设置，不要粘贴到对话中。

## 打开阅读界面

初始化后直接运行：

```sh
node /absolute/plugin/path/dist/arxiv-daily-cli.cjs ui
```

命令启动本地服务并尝试打开默认浏览器。也可点击终端输出的完整 `Workbench:` 链接。保持进程运行，结束阅读后按 Ctrl+C 停止。`ui --no-open` 只显示链接；`ui --port 8123` 指定端口，默认自动选择空闲端口。服务只监听 `127.0.0.1`。

- 左侧切换日报 / 论文总结，按标题、作者、arXiv ID 或日期搜索。
- 日报标签提供月历，可切换月份、回到今天，点击已有日报的日期直接阅读。其他日期展示运行状态；生成或重试会先打开已填好日期的确认表单。手机端可折叠月历。
- 月历结合实际文件与完整运行记录区分未生成、生成中、无匹配、失败、已跳过和文件缺失；日期按配置时区显示。查看日期不触发模型，也不会把没有文件直接判断为尚未发布。
- 右侧阅读 Markdown 渲染的正文：标题、表格、代码、图片和科学公式；目录可跳转章节，日报链接可打开已有论文总结。
- “Markdown”打开原始文本；原文 / PDF 按钮打开论文来源，存在产品输出目录下的本地 PDF 时优先读取本地文件。阅读不会修改原文件。
- 页面可以发起日报和单篇详细总结，显示进度、结果和取消操作。它调用同一 CLI 流程；启用的邮件投递配置也会生效。
- 设置界面显示当前主题、模型和输出目录；修改仍通过终端 `init` 向导或 CLI TOML 完成，之后重启工作台。

界面、样式与公式字体随插件打包，读取本地文档不调用模型，也不依赖 CDN。外链图片与论文原文仍需要相应网络连接。首版提供阅读，不提供 Markdown 编辑、双向链接图谱或完整 Obsidian 插件能力；个人文献库的索引和方向审核继续使用已有命令。

## 直接运行核心功能

```sh
node /absolute/plugin/path/dist/arxiv-daily-cli.cjs status
node /absolute/plugin/path/dist/arxiv-daily-cli.cjs run --today
node /absolute/plugin/path/dist/arxiv-daily-cli.cjs run --date 2026-09-30
node /absolute/plugin/path/dist/arxiv-daily-cli.cjs run --id 2609.12345
node /absolute/plugin/path/dist/arxiv-daily-cli.cjs papers --query "inference"
```

- 日报：抓取候选 → 按配置主题筛选 → 结构化总结 → 按策略生成自动详细总结 → 保存日报和索引。
- 单篇：按 ID 获取正文 → 详细总结 → 保存文件并关联 Paper Index。
- 重复运行、暂未发布、零命中、取消和恢复均沿用原有规则。
- `status` 提供不含密钥的 JSON 概览；`papers` 查询真实 Paper Index，使用与 Dashboard 相同的词法检索。
- 可通过既有 `schedule` 命令安装系统调度；运行不依赖 Claude 会话持续打开。

## 通过 Claude Code 操作

```sh
claude --plugin-dir /absolute/path/to/extensions/claude-code-arxiv-daily
```

进入后输入 `/arxiv-daily:research`，再说：

- “查看当前主题和最近日报的运行状态。”
- “生成今天的日报。”
- “为 arXiv:2609.12345 生成详细总结。”
- “在已保存论文里找 inference 相关内容，解释其中两篇的区别。”

Claude 调用同一个产品命令，并读取实际结果。它不会临时选几篇论文代替产品筛选，也不另写一套报告、研究方向或索引。

想用界面时，输入 `/arxiv-daily:open`，或说“打开论文阅读工作台”。Claude 会在后台启动本地服务并返回实际链接。此时“已启动”表示服务就绪，后台进程会持续运行；生成任务仍以页面或命令的实际完成结果为准。

## 数据位置与兼容

配置仅来自平台 CLI 配置目录：Linux/macOS 默认 `~/.config/arxiv-daily/config.toml`，Windows 为 `%APPDATA%/arxiv-daily/config.toml`。其中 `vault_root` 决定输出位置；在不同目录启动 Claude 不会切换资料库。

默认产物是配置 Vault 下的 `arxiv-daily/daily/`、`arxiv-daily/papers/`、`arxiv-daily/.index/`。与原 CLI/Obsidian 使用同一业务格式，但各产品设置不会自动同步。用户原有论文和笔记保护沿用共享核心。

早期 P1 的临场研究代码和 `arxiv-daily-agent/` 独立档案接口已移除。已生成的试用文件保留，未自动删除或导入正式索引；需要时可手动查阅。

## 可选：接入个人文献库

先完成基本设置，再按需要启用：

```sh
ARXIV_DAILY_CLI="/absolute/plugin/path/dist/arxiv-daily-cli.cjs"
node "$ARXIV_DAILY_CLI" library connect "/absolute/path/to/pdfs"
node "$ARXIV_DAILY_CLI" library status
# 阅读上一步显示的目录、处理深度和模型端点后，使用其指纹授权：
node "$ARXIV_DAILY_CLI" library authorize --fingerprint "sha256:上一步的指纹"
node "$ARXIV_DAILY_CLI" library prepare
node "$ARXIV_DAILY_CLI" library scan
node "$ARXIV_DAILY_CLI" library index
node "$ARXIV_DAILY_CLI" library propose
node "$ARXIV_DAILY_CLI" library directions
```

审核候选后，用显示的 ID 和版本确认：

```sh
node "$ARXIV_DAILY_CLI" library confirm --candidate "候选ID" --proposal-revision 0 --profile-revision 0
node "$ARXIV_DAILY_CLI" run --today
node "$ARXIV_DAILY_CLI" library search --query "你关心的具体问题"
```

确认后的有效方向会进入原来的日报筛选，与手动主题共同工作；日报解释发现来源和相对于已有文献的新意。新增论文后再次 `scan`、`index`，有变化的索引会按原规则生成增量建议，建议仍须人工审核。修改方向、启停、锁定、应用/忽略建议见 [命令参考](references/commands.md)。

本地索引不把全文发送到模型服务，也可先索引、后为方向生成授权。远程 embedding 会发送全文片段，必须取得显示范围的授权。切换端点或处理范围会使旧授权失效；`library revoke` 可撤销。库连接保存进同一 TOML，连接/授权操作保留其他设置值，但会规范化 TOML 格式。

`prepare` 将锁定的 PDF.js 和本地推理依赖装到产品缓存；选择远程 embedding 时不安装本地 CPU 组件。本地 e5 模型权重在首次使用时下载，后续复用缓存。已在 Linux Node 20.19 和 Node 22 验证实际 PDF 解析和 CPU 推理；Windows/macOS CPU运行尚待验收。

Markdown 可以在工作台、Obsidian 或其他阅读工具中查看。同一输出目录的文献库重建/审核请单宿主执行；本次没有建立 Obsidian 与 CLI 同时重建同一库的并发保证。

## 验证

```sh
node extensions/claude-code-arxiv-daily/build.mjs
node --test extensions/claude-code-arxiv-daily/tests/*.test.cjs
claude plugin validate --strict --json extensions/claude-code-arxiv-daily
npm run typecheck
npm run check:boundaries
npm run check:product-units
```

基础端到端测试只替换 HTTP，实际运行打包 CLI、scheduler、pipeline、writer 和 Paper Index，覆盖日报、自动/手动详报、离线重跑、零命中、未发布和用户文件保护。库流程另有受控端口的集成测试，以及真实 PDF.js/e5 CPU 索引与检索验证。配置与研究输出隔离在试验目录；原生模块使用产品正常的本机缓存。

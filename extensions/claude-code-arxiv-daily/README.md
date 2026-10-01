# arXiv Daily for Claude Code CLI — 核心流程试验版

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

## 数据位置与兼容

配置仅来自平台 CLI 配置目录：Linux/macOS 默认 `~/.config/arxiv-daily/config.toml`，Windows 为 `%APPDATA%/arxiv-daily/config.toml`。其中 `vault_root` 决定输出位置；在不同目录启动 Claude 不会切换资料库。

默认产物是配置 Vault 下的 `arxiv-daily/daily/`、`arxiv-daily/papers/`、`arxiv-daily/.index/`。与原 CLI/Obsidian 使用同一业务格式，但各产品设置不会自动同步。用户原有论文和笔记保护沿用共享核心。

早期 P1 的临场研究代码和 `arxiv-daily-agent/` 独立档案接口已移除。已生成的试用文件保留，未自动删除或导入正式索引；需要时可手动查阅。

## 当前范围

本阶段先完整保留基础筛选、日报和详细总结。个人文献库的完整索引、聚类方向与个性化发现正在按共享核心边界逐步接入，不能将当前状态查询当作已支持这些操作。阅读界面仍待试用研究；Markdown 可用任意阅读工具打开。

## 验证

```sh
node extensions/claude-code-arxiv-daily/build.mjs
node --test extensions/claude-code-arxiv-daily/tests/*.test.cjs
claude plugin validate --strict --json extensions/claude-code-arxiv-daily
npm run typecheck
npm run check:boundaries
npm run check:product-units
```

端到端测试只替换 HTTP，实际运行打包 CLI、scheduler、pipeline、writer 和 Paper Index，覆盖日报、自动/手动详报、离线重跑、零命中、未发布和用户文件保护。配置与研究输出隔离在临时目录；原生模块使用产品正常的本机缓存。

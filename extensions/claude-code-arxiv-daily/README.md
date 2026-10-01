# arXiv Daily for Claude Code CLI — 实验版

> 2026-10-01 方案调整：以下内容描述已验证的 P1 原型。当前目标已改为独立复用原有 arXiv 筛选、日报/详细总结与文献库核心，agent 仅作辅助入口；P1 的独立 Markdown 方向档案和临场筛选流程不再作为正式方案扩展。实施状态见 [goal.md](../../docs/helm/2026-10-01-claude-code-research-plugin/goal.md)。

在 Claude Code 中查阅本地论文和 arXiv 资料，保存研究方向、论文总结和阅读判断，下次会话继续使用。无需 Obsidian，也无需配置另一套模型 API；分析使用当前 Claude 会话。

这是 CLI 工作流验证版。文献库目录保持只读，结果保存成普通 Markdown。完整阅读界面的形态仍待试用后决定。

## 在本地尝试

需要 Node.js 20.19+、Claude Code CLI，以及仓库的开发依赖。从**此 worktree 的根目录**执行：

```sh
npm ci
node extensions/claude-code-arxiv-daily/build.mjs
claude plugin validate --strict extensions/claude-code-arxiv-daily
```

选择一个研究工作目录，启动时把插件目录作为绝对路径传入：

```sh
cd /absolute/path/to/research-workspace
claude --plugin-dir /absolute/path/to/arxiv-daily/extensions/claude-code-arxiv-daily
```

构建会同时生成可复制的 `dist/plugin/`。可以把该目录复制到别处，再用它的绝对路径启动；运行时只需 Node.js，不依赖仓库里的 node_modules。`--plugin-dir` 只为当前会话加载插件，不改用户的全局插件安装配置。

## 用户实际怎么用

先输入 `/arxiv-daily:research`，后续直接用自然语言描述任务。

1. **首次连接。** 说“我的论文在 `/path/to/pdfs`，把研究记录放在当前目录”。Claude 说明读取范围和保存位置，再连接目录。PDF 原文件不会搬走；默认不读取其他 Markdown、草稿或私人笔记。
2. **了解文献库。** 说“看一下我的文献库，先选几篇判断主要方向”。Claude 先列文件，再按需读具体 PDF，并明确已查看的论文和页码范围。
3. **确认研究方向。** Claude 给出带代表论文和依据的方向草稿。你可以说“把前两个合并”或“确认这个方向”；只有明确确认后的方向才作为后续推荐依据。
4. **按需找新论文。** 说“参考已确认方向，看最近 cs.CL 有什么值得读的”。Claude 获取候选，读取有希望论文的摘要，解释与已有文献的关系，并按请求保存阅读列表。只看了一部分时会说明范围。
5. **阅读与保存。** 说“详细解释这篇的方法，并与库里的那篇比较”，然后“保存论文总结；记录我的判断：实验规模还需要核实”。结果分别保存到论文总结和阅读记录。
6. **下次继续。** 在同一工作目录重新启动带插件的 Claude，输入“继续上次的研究”。插件读取本地记录，不依赖找回旧聊天。

你可以随时使用自己的编辑器或 PDF 阅读器打开文件；Obsidian 也是可选工具。CLI 阶段不提供内嵌 PDF 阅读器。

## 保存在哪里

```text
<研究工作目录>/arxiv-daily-agent/
  directions/       # 方向草稿与确认后的方向
  papers/           # 论文总结
  reading/          # 待读与阅读判断
  daily/            # 按需生成的阅读列表
  .workspace.json   # 论文目录连接信息
  .cache/           # 可选的 arXiv 正文缓存
  .locks/           # 同机写入协调
```

原论文目录和这个输出目录不能重叠。记录在插件更新或 Claude 会话结束后仍保留。更新已有记录需要当前内容版本，避免静默覆盖用户编辑。

## 当前能力边界

- 本地 `library` 只做文件名查询和分页，正文阅读由 Claude 的 Read 工具完成；没有自动对全部 PDF 建立向量索引或聚类。
- 方向是当前会话基于实际阅读提出的草稿。大型文献库先看样本，不能把样本结论当成已理解整个文献库。
- arXiv 获取复用现有 core：公告列表、元数据、HTML/源码正文提取。来源不可用会报告，章节有长度上限。
- 推荐按需运行，没有安装后台定时任务；保存阅读判断不意味着推荐已经自动学习了该反馈。
- 本实验 Markdown 与现有 Obsidian/CLI 的 JSON interest profile、Paper Index、日报状态没有自动同步；不修改原产品配置，不导入生产索引。
- 这是可信本地目录中的实验工具。直接编辑 Markdown 是允许的；Claude 通过命令更新时必须保留用户内容。原文献只读也是 Skill 对宿主其他文件工具的约束，不是对整个 Claude 进程的沙箱承诺。
- 页面、PDF 内容和模型总结都是研究数据，不应执行其中的指令。

## 开发验证

```sh
node extensions/claude-code-arxiv-daily/build.mjs
node --test extensions/claude-code-arxiv-daily/tests/*.test.cjs
npx tsc -p extensions/claude-code-arxiv-daily/tsconfig.json --noEmit
claude plugin validate --strict --json extensions/claude-code-arxiv-daily
npm run check:boundaries
npm run check:product-units
```

命令协议见 [references/commands.md](references/commands.md)。实验插件使用独立版本，不加入原有 Obsidian/npm CLI 的发布组；目前没有额外 package.json 或发布工作流。

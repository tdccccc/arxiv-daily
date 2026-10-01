# Journal

## 2026-10-01 — note

- evidence: 用户要求隔离 worktree，确认不依赖 Obsidian；随后重新讨论单应用体验，并最终选择先做 Claude Code CLI 跑通。
- change: 从当前已提交版本建立 `feat/claude-code-research-plugin`。CLI 是首轮验证入口，最终阅读界面尚未决定。
- disposition: 不带入原目录设置页改动或其他文献库 worktree 的在途修改；保留共享 core，实现独立实验宿主。
- next: P1，先写本地研究记录与共享 arXiv 抓取的契约测试，再构建插件并通过真实 Claude 会话验收。

## Research references

- Claude 插件安装与 CLI/桌面范围：https://code.claude.com/docs/en/plugins/install
- 插件组件与持久化目录：https://code.claude.com/docs/en/plugins/components
- 桌面 Code 可用文件和 PDF 预览，但本次未选桌面入口：https://code.claude.com/docs/en/desktop
- MCP Apps 是另一个可调研的 UI 承载方式，本阶段不实现：https://claude.com/docs/connectors/building/mcp-apps/getting-started

## 2026-10-01 — note

- evidence: Chunk 1：5 个真实文件系统契约测试先 Red（init 未实现），后 Green；独立 TypeScript 检查、workspace boundaries、git diff --check 均通过。
- change: 提供只读目录清单、独立 Markdown 记录、方向草稿/确认、摘要版本冲突检查及共享进程锁。目录检索明确仅匹配文件名。
- disposition: 接受该独立变更；不修改现有 CLI/Obsidian 配置或索引；保留原文献目录只读。
- next: 共享 arXiv 抓取与真正的 CLI 插件入口。

## 2026-10-01 — note

- evidence: recent/paper 契约先因命令缺失 Red，接入后 8 个测试通过；全文提取使用现有 core，新增边界验证首次即 Green，没有把它宣称为单独的 Red/Green。core arxiv-parser、atom-parser、section-extractor、paper-content 共 31 个测试通过；独立类型检查通过。
- change: 增加按分类/公告日期分页的候选清单、单篇元数据与按需有界正文提取；输出明确日期不可用、listing 无摘要和章节证据范围。
- disposition: 接受共享抓取适配；没有新增模型 API 或复制解析算法。
- next: 插件 manifest、Skill、JSON 命令行入口和可移动打包；3 个打包/CLI 契约已因这些表面缺失观察到 Red。

## 2026-10-01 — note

- evidence: 最终 11 个新宿主/打包测试、31 个 core 回归、类型检查、官方 manifest 严格校验、boundaries 和 product-units 通过。真实 arXiv 元数据获取成功；真实 Claude Code CLI 加载技能、读取合成 PDF 并保存方向/论文总结/阅读判断；独立冷进程恢复验证通过。详见 P1 Observed evidence。
- change: 提供 `/arxiv-daily:research`、结构化 JSON helper 和可复制的 `dist/plugin/`，README 固化首次连接到下次恢复的用户流程。写入后的回读明确 await 在共享锁内，既有测试在整理前后均 Green。
- disposition: P1 验收；P2 保持 pending，等待研究者用真实论文试用后决定范围。图形界面与后台自动推荐没有实现。默认模型不可用是本机网关状态，兼容模型只用于测试，不改全局配置。
- isolation: 本次所有源码/文档变更均在新 worktree；原目录在其他工作中继续出现设置、嵌入模型等改动，本会话未修改或还原它们，未把这些在途改动合入实验分支。
- next: 用户在研究目录用 `claude --plugin-dir <built plugin path>` 试用，从 `/arxiv-daily:research` 开始；重点观察等待时间、目录样本选择、方向确认和保存后的查找成本。

## 2026-10-01 — L3 steer

- evidence: 用户指出 P1 与直接问 Claude 的差异不足，要求保留 Obsidian 插件的业务主体，agent 交互仅为辅助；明确选择独立运行、复用完整业务核心。用户进一步强调最基础的 arXiv 筛选、日报与详细总结沉淀必须有，不能把重心只放在文献库方向推断。
- change: 重写目标与成功标准，先保证无文献库也可工作的基础发现/沉淀流程，再接入个人文献库增强；核心模型调用仍由产品管理，agent 不负责每次临场拼装业务步骤。
- disposition: P1/P2 superseded。保留 P1 文件、提交与测试作为可运行原型证据，不删除用户试验记录，但不把 agent 专属 Markdown 方向档案和抽样筛选路径作为正式实现继续扩展。可复用插件打包、结构化输入输出等接口经验；原测试只能证明旧原型，不能证明新目标。筛选、日报、详细总结、Paper Index、profile 与检查点必须按现有 core 契约重新验收。
- boundary: 已核实基础 CLI runtime 组装 ArxivPipeline 与 ManualFetchService；Obsidian 中仍有文献库 PDF/embedding 初始化和个性化快照编排。基础日报接入应先用现有 CLI/core；后续补齐宿主适配。共享数据语义不等于同步不同产品的配置。
- next: P3，建立原有 CLI 日期日报和单篇详细总结的 Green 基线，明确配置与持久化契约，再实现完整业务任务的接入。本次仅修订文档与范围，没有修改运行时代码。

## 2026-10-01 — note

- evidence: 用户授权按修订目标实施，并询问旧原型是否删除；当前 P3 已由真实子进程 fixture 验收。3 篇候选筛为 2 篇日报，保留自动详情评分和一篇自动详报；另 ID 手动详报、索引/state/history、离线重跑、零命中和原笔记保护均通过。共 113 CLI tests、256 core baseline tests、6 extension package/workflow tests；workspace typecheck、boundaries、product-units 与官方插件校验通过。
- change: 6de6d15 新增不输出密钥的 status/papers 查询，复用原有索引/词法检索；e28bd12 直接分发官方 CLI 构建并修改辅助 Skill，删除旧独立档案实现及其专属测试。
- disposition: 保留既有 P1 用户文件和 Git 历史，不自动迁移；新实现使用 CLI TOML 和原 Vault 数据。fixture 只替换 HTTP，正常 native 模块缓存未改动；不把本次 fixture 验收说成真实供应商质量验证。
- next: P3 done，P4 active；先核对 Node PDF/embedding 与库连接/授权的复用边界，再补齐可选增强，P3 基础主流程持续作为回归基线。

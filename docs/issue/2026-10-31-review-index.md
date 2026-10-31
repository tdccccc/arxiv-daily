# 2026-10-31 代码审查：问题索引

> 审查对象：main `e0498c94`（2026-10-31）· 范围：core 日报流水线、运行状态与调度、文献库核心、CLI（init / 配置 / 定时 / 工作台服务端与前端）、插件主入口；**未含**插件设置页与 Dashboard、邮件投递与中继、DSH / VS Code 扩展、node-runtime、scripts
> 方式：会话内直接通读（未用 subagent），每条对照周边代码与现有测试；发现时全部为代码推断，其中 PIPE-02、PIPE-03、STATE-01、CLI-01 随后用离线测试复现
> 结果：14 条（中 6、低 8），其中功能 bug 3、逻辑优化 4、体验 7；已修复 7、按既有 ADR 不改行为只补文档 1、未修复 6（分支 `fix/2026-10-31-review`）

每条问题一个文件，结构与 `archive/` 相同：问题与依据 / 原因 / 建议和验收 / 修复后应补的测试。修复后按惯例移入 `archive/`，并在文件末尾补"修复情况"。

编号前缀：PIPE 日报流水线（core）· STATE 设置与 LLM 客户端 · LIB 个人文献库 · CLI 命令行与阅读工作台。编号在本索引内独立，与 2026-10-21 索引的同名编号无关。

## 中

| 编号 | 类型 | 状态 | 置信度 | 问题 | 文件 |
|---|---|---|---|---|---|
| PIPE-01 | 逻辑 | 已修复 | 代码推断 | 日报摘要和详细笔记一篇做完再做下一篇，20 篇日报耗时 ≈ 20 × 单次模型延迟；正文抓取却是 6 路并行 | [daily-generation-is-sequential](archive/2026-10-31-daily-generation-is-sequential.md) |
| PIPE-02 | bug | 已修复 | 已复现 | 筛选模型一次回复格式不对，当天直接记为临时失败：等一个 tick、消耗一次重试配额、下次整批重发；摘要路径有 3 次纠错重试，筛选一次都没有 | [filter-response-validation-not-retried](archive/2026-10-31-filter-response-validation-not-retried.md) |
| PIPE-03 | bug | 已修复 | 已复现 | 详细笔记选择器遇到不合格回复（含 ```json 围栏）或一次网络失败就静默选零篇，日报照常写出、没有任何提示 | [detail-selector-silently-selects-nothing](archive/2026-10-31-detail-selector-silently-selects-nothing.md) |
| STATE-01 | bug | 已修复 | 已复现 | 插件设置页、CLI init、工作台设置里的"加载模型列表"对 Anthropic 原生端点必定失败：请求头用 Bearer 而不是 x-api-key | [model-list-request-uses-bearer-for-anthropic](archive/2026-10-31-model-list-request-uses-bearer-for-anthropic.md) |
| CLI-01 | 体验 | 已修复 | 已复现 | init 向导允许把占位说明文字当成研究方向写入配置，并把"之后再改"说成可选；第一份日报因此为空或乱 | [init-accepts-placeholder-topic-direction](archive/2026-10-31-init-accepts-placeholder-topic-direction.md) |
| CLI-02 | 逻辑 | 不改行为（ADR 0003）；README 已说明 | 代码推断 | cron 定时只跑 `run --today`，机器关机错过的日子不补；复核发现 ADR 0003 与 0.3.4 发布说明明确把"CLI 不做多日回看"记为产品决定，本次只补文档 | [cron-schedule-never-catches-up-missed-days](2026-10-31-cron-schedule-never-catches-up-missed-days.md) |

## 低

| 编号 | 类型 | 状态 | 置信度 | 问题 | 文件 |
|---|---|---|---|---|---|
| PIPE-04 | 体验 | 未修复 | 代码推断 | 被每日上限截掉的论文也进了索引：Dashboard "All" 页和工作台 inbox 里当普通条目出现，无摘要、无日报链接、无标记，还算进"本周新增" | [cut-by-daily-limit-papers-in-index](2026-10-31-cut-by-daily-limit-papers-in-index.md) |
| PIPE-05 | 逻辑 | 未修复 | 代码推断 | 被服务商内容安全拒绝的论文在筛选决策记录里记成"模型跳过"，污染筛选质量评估的样本 | [refused-papers-recorded-as-model-skips](2026-10-31-refused-papers-recorded-as-model-skips.md) |
| PIPE-06 | 体验 | 已修复 | 代码推断 | "抓取正文"进度在请求开始时就计数，6 路并发下一开始就显示 6/N | [content-fetch-progress-counts-starts](archive/2026-10-31-content-fetch-progress-counts-starts.md) |
| PIPE-07 | 逻辑 | 已修复 | 代码推断 | 主题设置校验在抓完 arXiv 列表之后才做，配置错误时白费一次抓取 | [topic-validation-after-listing-fetch](archive/2026-10-31-topic-validation-after-listing-fetch.md) |
| LIB-01 | 体验 | 未修复 | 代码推断 | 元数据拉取失败的 PDF 不会被"索引 / Retry preparation"重试，部分失败时也不提示；恢复路径（重新扫描）没人指 | [metadata-failed-pdfs-not-retried-by-index](2026-10-31-metadata-failed-pdfs-not-retried-by-index.md) |
| CLI-03 | 体验 | 未修复 | 代码推断 | 工作台页面只在"已知有任务在跑"时轮询，服务端自动调度启动的任务在空闲页面上看不到，跑完也不刷新 | [workbench-misses-server-scheduled-runs](2026-10-31-workbench-misses-server-scheduled-runs.md) |
| CLI-04 | 体验 | 未修复 | 代码推断 | config.toml 里类型写错的值（如 `enabled = "true"`）被静默忽略，功能悄悄回到默认 | [config-type-mismatches-silently-ignored](2026-10-31-config-type-mismatches-silently-ignored.md) |
| CLI-05 | 体验 | 未修复 | 代码推断 | init 对所有服务商默认开启思考模式，而连接测试只请求 /models，参数不兼容到第一次日报才以永久失败暴露 | [init-defaults-thinking-without-chat-test](2026-10-31-init-defaults-thinking-without-chat-test.md) |

## 相互关联

- PIPE-02、PIPE-03 与摘要路径是同一类问题（"模型输出格式不对怎么办"）的三种不同处理，本次统一成"容忍单层围栏 + 一次纠错重问 + 失败可见"。
- STATE-01、CLI-01、CLI-05 都落在新用户的首次配置路径上；前两条已修，CLI-05 待办。
- CLI-02 与 CLI-03 都涉及 CLI / 工作台的定时运行；若将来修订 ADR 0003 让 CLI 补跑，可一并处理。
- PIPE-06 是 PIPE-01 并行化后更明显的问题，已一起修。

## 本次没有覆盖或没做完的部分

- Obsidian 插件的设置页（`plugin/src/settings/tab.ts`）、Dashboard 视图、commands 只看了主入口与 Force run 文案；邮件投递与中继、DSH / VS Code 扩展、node-runtime、工作台前端的设置 / 文献库 / 方向审核页、scripts / CI 没有看。
- 没有在 Windows / macOS 上验证。审查阶段没有运行代码；修复阶段运行了 core 与 CLI 的单元测试（见各文件"修复情况"）。

## 相关但不在本索引中

- 摘要质量评估的后续问题 SQE-01~04 见 [2026-10-30-summary-eval-followups-index.md](2026-10-30-summary-eval-followups-index.md)。
- 测试与评测缺口 T03~T10 见 [2026-10-29-tests-and-llm-evaluation-review.md](2026-10-29-tests-and-llm-evaluation-review.md)（T01 / T02 已在 #146 修复）。

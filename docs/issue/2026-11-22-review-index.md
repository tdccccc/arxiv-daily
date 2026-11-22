# 2026-11-22 代码审查：问题索引

> 审查对象：main `2557b90a`（v0.6.6，2026-11-22）· 范围：`packages/core/` 全模块（LLM 客户端、设置与校验、日报流水线、数据源、运行状态/历史/检查点/调度、Dashboard 核心与同步、PDF/文档解析、邮件投递、个人文献库目录/提案/全文检索与代索引）及 `plugin/` 全模块（主入口、命令、Obsidian 宿主适配器、Reading Dashboard、个人文献库 UI、Obsidian 1.13+ 声明式与 `<1.13` 旧版设置页、`styles.css`、`manifest.json`）；**未含** CLI（`apps/cli`）、邮件中继服务端（`services/`）、DSH / VS Code 扩展（`extensions/`）、`scripts/`
> 方式：会话内直接通读源码并对照周边测试与 ADR，排除了已存在且未关闭的历史 issue 与已由 PIN 测试固化的刻意行为；共对 10 项核心缺陷执行了离线脚本实测复现
> 结果：22 条（高 6、中 11、低 5），其中功能 bug 18、逻辑优化 2、体验 2；已复现 8、部分已复现 2、代码推断 12；全部状态为未修复

每条问题一个文件，结构与 `archive/` 相同：问题与依据 / 原因 / 建议和验收 / 修复后应补的测试。修复后按惯例移入 `archive/`，并在文件末尾补“修复情况”。

编号前缀：OBS Obsidian 插件（设置页 / Dashboard / 命令）· STORE 存储适配器与运行历史 · LIB 个人文献库（core + 插件）· PIPE 日报流水线与服务 · LLM 模型客户端 · MAIL 邮件投递 · DOC 文档与 PDF 解析。编号在本索引内独立，与历史索引的同名编号无关。

## 高

| 编号 | 类型 | 状态 | 置信度 | 问题 | 文件 |
|---|---|---|---|---|---|
| OBS-01 | bug | 未修复 | 代码推断 | 设置页修改 API Key 或 Base URL 后点击 "Get models"，因 `cacheKey` 在渲染时冻结，请求成功但下拉模型列表被静默丢弃 | [settings-get-models-stale-cache-key](2026-11-22-settings-get-models-stale-cache-key.md) |
| STORE-01 | bug | 未修复 | 已复现 | `ObsidianStorageAdapter.remove` 仅调用文件级 `vault.adapter.remove`，导致 Obsidian 下旧搜索索引清理必然失败（目录删不掉）且 `run-history.jsonl` 超 512 KB 轮转后永久停写（`ENOENT`） | [obsidian-storage-remove-breaks-maintenance-and-history-rotation](2026-11-22-obsidian-storage-remove-breaks-maintenance-and-history-rotation.md) |
| LIB-01 | bug | 未修复 | 代码推断 | 文献库索引在 `replaceManifest` 提交之前提前调用 `store.removePaper` 删除旧 manifest 下的论文文档，切换模型全失败或中途取消会破坏已提交索引 | [fulltext-index-deletes-committed-papers-before-manifest-commit](2026-11-22-fulltext-index-deletes-committed-papers-before-manifest-commit.md) |
| LIB-02 | bug | 未修复 | 代码推断 | 撤销文献库远程处理授权（`revokeLibraryProcessing`）未取消正在进行的远程全文索引，剩余论文继续外发远程嵌入 API 并提交 manifest | [revoke-library-processing-does-not-cancel-remote-indexing](2026-11-22-revoke-library-processing-does-not-cancel-remote-indexing.md) |
| LIB-03 | bug | 未修复 | 代码推断 | 远程嵌入配置尚未填完时被 `refreshLibraryIndexTrace` 误报为“索引文件读取失败”，且补填 `embedding.apiKey` 漏在监听列表中导致永久卡死在该错误态 | [remote-embedding-config-traps-library-index-in-read-error](2026-11-22-remote-embedding-config-traps-library-index-in-read-error.md) |
| PIPE-01 | bug | 未修复 | 已复现 | 损坏的 `daily-summary-checkpoints/<date>.json`（无有效 `.bak`）会让当天陷入重试循环、消耗 10 次 LLM 调用变永久失败，且 Force run / Clear run state 均无法自愈 | [corrupt-summary-checkpoint-wedges-date-and-burns-llm](2026-11-22-corrupt-summary-checkpoint-wedges-date-and-burns-llm.md) |

## 中

| 编号 | 类型 | 状态 | 置信度 | 问题 | 文件 |
|---|---|---|---|---|---|
| OBS-02 | bug | 未修复 | 代码推断 | Dashboard 打开时 `recentDates` 在 3 秒内正常加载完成后直接 `return` 未触发 `render()`，回看窗工作日一直显示不可运行且部分失败/超时提示不渲染 | [dashboard-recent-dates-missing-rerender-on-open](2026-11-22-dashboard-recent-dates-missing-rerender-on-open.md) |
| LLM-01 | bug | 未修复 | 已复现 | 非原生 Anthropic 代理/网关（`provider: "anthropic"` + 非 `api.anthropic.com`）仍向 OpenAI `/chat/completions` 发送原生 `thinking`/`output_config` 且跳过参数降级回退 | [anthropic-proxy-sends-native-thinking-and-skips-fallback](2026-11-22-anthropic-proxy-sends-native-thinking-and-skips-fallback.md) |
| LLM-02 | bug | 未修复 | 已复现 | 非自适应思考 Claude 模型传入 `maxCompletionTokens: 4096`（方向提议/预览）时 `budget_tokens` 占掉 4095 token，仅留 1 个输出 token 必然截断 | [anthropic-thinking-budget-leaves-one-output-token](2026-11-22-anthropic-thinking-budget-leaves-one-output-token.md) |
| LIB-04 | bug | 未修复 | 部分已复现 | 插件每次本地文献库搜索都重建 ~130 MB ONNX WASM 实例且从未释放；同时 core 中 `pipelinePromise` 在 `load()` 失败后永不清空导致实例永久毒化 | [local-embedding-rebuilt-every-search-and-load-failure-poisons-cache](2026-11-22-local-embedding-rebuilt-every-search-and-load-failure-poisons-cache.md) |
| OBS-03 | bug | 未修复 | 代码推断 | `syncDashboardHistory` 忽略 `MarkdownWriter` 写入的 `published:` frontmatter，重建索引后单篇论文笔记丢失日期与 `seenDates` | [dashboard-history-sync-ignores-published-frontmatter](2026-11-22-dashboard-history-sync-ignores-published-frontmatter.md) |
| PIPE-02 | bug | 未修复 | 已复现 | `PdfService.downloadById` 未校验 `%PDF-` 文件头，将 arXiv 返回的 HTTP 200 HTML 错误/验证页原样保存为 `.pdf` 并记入 `papers.json` | [pdf-service-saves-html-error-page-as-pdf](2026-11-22-pdf-service-saves-html-error-page-as-pdf.md) |
| OBS-04 | bug | 未修复 | 代码推断 | 设置页两处保存/回显缺陷：Self-Send（Resend）点 "Send test" 未先 flush 刚粘贴的 API Key；自定义 IANA 时区重开设置页后从下拉框消失且旧版漏同步 `<select>` | [settings-resend-send-test-races-api-key-and-custom-timezone-dropped](2026-11-22-settings-resend-send-test-races-api-key-and-custom-timezone-dropped.md) |
| OBS-05 | bug | 未修复 | 代码推断 | Dashboard 三处交互状态 bug：未连接文献库打开相似论文默认选中 Library 页签报错、Starred 空态点 "Show All" 未更新页签高亮/计数/分页、文献库搜索失败永久写入缓存无法点刷新重试 | [dashboard-similar-modal-and-ui-state-bugs](2026-11-22-dashboard-similar-modal-and-ui-state-bugs.md) |
| PIPE-03 | bug | 未修复 | 部分已复现 | 日报流水线先写精读笔记再写日报且不回填 frontmatter，导致精读笔记的 `published` 日报双向链接永远缺失；且非 `YYYY-MM-DD` 日期字符串会让写笔记抛 `Invalid date` 崩溃 | [paper-detail-wikilink-never-created-in-daily-run-and-bad-date-crash](2026-11-22-paper-detail-wikilink-never-created-in-daily-run-and-bad-date-crash.md) |
| LIB-05 | bug | 未修复 | 代码推断 | `mutatePersonalLibraryProposal` 遇到版本冲突重读磁盘提案却丢弃返回值导致内存状态保持过期；且 `selectLibraryRoot` 加载 catalog 失败未记录 `libraryCatalogLoadError` | [library-proposal-stale-conflict-discards-reloaded-doc](2026-11-22-library-proposal-stale-conflict-discards-reloaded-doc.md) |
| MAIL-01 | bug | 未修复 | 已复现 | `sendViaHosted` 把 `RunCancelledError` 和前置校验异常吞为 `HostedDeliveryError(..., ambiguous: true, notSent: false)`；`formatResendFrom` 未给含 RFC 5322 特殊字符的发件人名加引号 | [hosted-email-swallows-cancellation-into-delivery-error](2026-11-22-hosted-email-swallows-cancellation-into-delivery-error.md) |

## 低

| 编号 | 类型 | 状态 | 置信度 | 问题 | 文件 |
|---|---|---|---|---|---|
| LIB-06 | 逻辑 | 未修复 | 代码推断 | 个人文献库方向预览与主题整理未剥离 Markdown JSON 代码围栏，且方向预览缺少格式纠错重试 | [library-direction-preview-and-organization-reject-json-fences](2026-11-22-library-direction-preview-and-organization-reject-json-fences.md) |
| DOC-01 | bug | 未修复 | 已复现 | `PdfJsDocumentParser` 对大写 UTF-16LE BOM（`FFFE`）大小写敏感误按大端解码成乱码且无长度上限；`title-extraction` 未校验 HTML 数字实体码点上限可能抛 `RangeError` | [pdfjs-uppercase-utf16le-bom-and-unchecked-codepoint](2026-11-22-pdfjs-uppercase-utf16le-bom-and-unchecked-codepoint.md) |
| OBS-06 | 体验 | 未修复 | 代码推断 | 命令与诊断三处遗漏：`clear-run-state` 与 `set-paper-status`/`mark-current-*` 不刷新 Dashboard（前者也不检查活跃任务）、`diagnoseEmbedding` 忽略远程模式强下本地模型、`openTodayDaily` 未规范化路径 | [obsidian-commands-missing-dashboard-refresh-and-diagnostics-mode](2026-11-22-obsidian-commands-missing-dashboard-refresh-and-diagnostics-mode.md) |
| PIPE-04 | 逻辑 | 未修复 | 代码推断 | 零命中日报完成遗漏 `cleanupCommittedCheckpoints` 残留检查点文件；且流水线在摘要成功前就写入 `papers.json` 导致失败/取消残留无摘要孤立条目 | [zero-match-day-leaks-checkpoints-and-failed-run-leaves-orphan-index](2026-11-22-zero-match-day-leaks-checkpoints-and-failed-run-leaves-orphan-index.md) |
| OBS-07 | 体验 | 未修复 | 代码推断 | `<1.13` 旧版设置页仍展示已被 ADR 0013 废弃的 `"Better PDF parser"` sidecar 配置；反向 1.13+ 声明式设置缺少主题模板下拉与自定义 arXiv 分类输入 | [settings-legacy-vs-declarative-parity-gaps](2026-11-22-settings-legacy-vs-declarative-parity-gaps.md) |

## 相互关联

- **OBS-01、OBS-04、LIB-03、OBS-07** 同属 Obsidian 设置页路径，且多集中在首次配置（填 API Key、切远程嵌入、配置邮件/时区）的关键入口，建议统一排查并补齐声明式/旧版双路径回归测试。
- **STORE-01** 同时暴露了测试替身失真问题（`plugin/tests/obsidian-adapters.test.ts` 的 `realFilesystemAdapter` 错用 `fs.rm(..., { recursive: true, force: true })` 模拟 Obsidian 的文件级 `vault.adapter.remove`），建议先修正测试替身契约再修适配器和调用方。
- **LIB-01、LIB-02、LIB-03、LIB-04、LIB-05** 同属个人文献库与嵌入生命周期管理，修复 LIB-04 复用本地嵌入实例时需先修复 core 中 `pipelinePromise` 失败毒化，否则复用反而会放大首次下载失败的副作用。
- **LLM-01、LLM-02** 同属 `packages/core/src/llm/client.ts` 的 Anthropic 参数生成逻辑，可一并修复。
- **OBS-03 与 PIPE-03** 共同涉及单篇论文笔记 `published:` frontmatter 契约（写出与回扫重建两端）。

## 本次审查排除项与未覆盖部分

- 已排除既有未关闭 issue（数学公式校验、筛选评估集、CLI cron 补跑等，见同目录 `2026-10-30-*` 与 `2026-10-31-*`）及既有行为固化测试（如 `scheduler.test.ts` 中 PIN 测试明确规定当天完成后 `tick()` 不再回看历史日期；`daily-paper-summary.test.ts` 明确规定摘要首轮拒绝围栏交由 3 轮纠错重试处理）。
- 未在真实 Obsidian 桌面端（Windows / macOS / Linux Electron 运行时）启动 UI 实测；基线运行了 `typecheck`、`lint:obsidian`、`check:boundaries`、`check:obsidian-submission`、`lint:review` 以及 core / plugin 单测（plugin 仅 2 条涉及写 `$HOME/.arxiv-daily` 的用例受沙箱限制，非代码缺陷）。

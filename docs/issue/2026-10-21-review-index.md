# 2026-10-21 深度审查与模型探索：问题索引

> 审查对象：main `d999037`（2026-10-21）· 不含 VS Code 扩展与 Claude Code mod
> 方式：7 个分区代码审查 + 4 轮模型驱动的模拟使用探索（gpt-6.1-sol，reasoning effort high）与逐条复核
> 更正：OBS-08 修复时对照 Obsidian 源码发现实际后果是设置被静默覆盖，严重程度由低调为中，以下计数已按新分级。
> 结果：54 条（高 2、中 34、低 18），其中产品问题 43 条、测试/发版工具问题 11 条；49 条已复现，5 条为代码推断；已修复 36 条、部分修复 0 条、未修复 18 条

每条问题一个文件，结构与 `archive/` 相同：问题 / 原因 / 可能的解决方案 / 修复后应补的测试。修复后按惯例移入 `archive/`，并在文件末尾补"修复情况"。

编号前缀：PIPE 日报流水线（core）· STATE 运行状态、调度、设置与 Node 运行时 · LIB 个人文献库 · CLI 命令行与阅读工作台 · OBS Obsidian 插件 · MAIL 邮件投递与中继 · DSH DSH 插件 · SCR 脚本、CI、发版与验收工具 · EXP 模型探索发现、经复核确认。

## 高

| 编号 | 状态 | 置信度 | 问题 | 文件 |
|---|---|---|---|---|
| LIB-01 | 已修复 | 已复现 | 按标题识别文献库 PDF 时，把 arXiv 条目摘要或评论里引用的另一篇编号当成本篇编号，目录静默记成别的论文 | [title-search-takes-cited-arxiv-id](archive/2026-10-21-title-search-takes-cited-arxiv-id.md) |
| LIB-02 | 已修复 | 已复现 | 两份相同的未识别 PDF 的路径用了两种不一致的排序，索引在最后一步提交失败，每次重试都失败 | [index-fails-on-duplicate-pdfs-locale-path-order](archive/2026-10-21-index-fails-on-duplicate-pdfs-locale-path-order.md) |

## 中：产品

| 编号 | 状态 | 置信度 | 问题 | 文件 |
|---|---|---|---|---|
| PIPE-01 | 已修复 | 已复现 | 模型回复被拦截、被服务端打断或中途断流时，残缺文本被当成完整回答，残缺笔记永久落盘 | [llm-interrupted-stream-accepted-as-complete](archive/2026-10-21-llm-interrupted-stream-accepted-as-complete.md) |
| PIPE-02 | 已修复 | 已复现 | arXiv 列表只取第一页（最多 2000 条），大分类较早的日期被判永久失败或静默缺论文 | [recent-listing-capped-at-2000-entries](archive/2026-10-21-recent-listing-capped-at-2000-entries.md) |
| PIPE-03 | 已修复 | 已复现 | 正文和摘要页都取不到时，用"[获取失败]"占位符生成摘要，并当成正常结果写进日报 | [content-fetch-failure-summarized-from-placeholder](archive/2026-10-21-content-fetch-failure-summarized-from-placeholder.md) |
| PIPE-04 | 未修复 | 已复现 | LaTeX 源码回退删掉公式命令（`\frac{1}{3}` 变成 `13`、`\pm`/`\leq` 消失） | [source-fallback-strips-math-commands](2026-10-21-source-fallback-strips-math-commands.md) |
| PIPE-05 | 未修复 | 已复现 | 源码清洗正则是二次复杂度，构造的论文源码可让插件或 CLI 卡死数分钟 | [source-extractor-quadratic-regex-freeze](2026-10-21-source-extractor-quadratic-regex-freeze.md) |
| PIPE-06 | 已修复 | 已复现 | 设置页允许保存日报预检会拒绝的值（重名主题、超长方向），日报整天反复失败直到永久失败 | [settings-accept-values-the-daily-preflight-rejects](archive/2026-10-21-settings-accept-values-the-daily-preflight-rejects.md) |
| PIPE-07 | 已修复 | 已复现 | 服务商内容安全拦截只要命中一篇论文，当天日报就永久失败 | [provider-content-block-fails-whole-day-permanently](archive/2026-10-21-provider-content-block-fails-whole-day-permanently.md) |
| EXP-01 | 已修复 | 已复现 | 给已入库论文生成详细总结后，研究主题被换成 arXiv 分类，主题筛选与列表对不上 | [manual-detail-summary-overwrites-paper-topic](archive/2026-10-21-manual-detail-summary-overwrites-paper-topic.md) |
| STATE-01 | 已修复 | 已复现 | 进程被强杀或崩溃后残留"运行中"：一小时内不能重跑，长期开着的 Obsidian 不会自动恢复 | [crash-left-running-state-blocks-rerun](archive/2026-10-21-crash-left-running-state-blocks-rerun.md) |
| STATE-02 | 已修复 | 已复现 | `run-state.json` 损坏后所有运行都以含糊的 JSON 错误失败，"清除运行状态"也失败 | [corrupt-run-state-wedges-scheduler-and-clear](archive/2026-10-21-corrupt-run-state-wedges-scheduler-and-clear.md) |
| CLI-01 | 已修复 | 已复现 | 在运行工作台的终端按 Ctrl+C，正在生成的日期卡在"运行中"，一小时内无法重跑 | [workbench-ctrl-c-leaves-date-running](archive/2026-10-21-workbench-ctrl-c-leaves-date-running.md) |
| CLI-02 | 已修复 | 已复现 | API 密钥是 `local`/`none`/`test` 这类短词时，输出里的同一个词被替换成 `[REDACTED]`，还会被自动保存写进研究方向 | [short-api-key-redaction-corrupts-text](archive/2026-10-21-short-api-key-redaction-corrupts-text.md) |
| CLI-03 | 已修复 | 已复现 | 自定义 `XDG_CONFIG_HOME` 的用户，`schedule install` 装的 cron 任务找不到配置，日报永不生成 | [cron-job-ignores-custom-xdg-config-home](archive/2026-10-21-cron-job-ignores-custom-xdg-config-home.md) |
| EXP-02 | 已修复 | 已复现 | 工作台设置里点"生成第一份日报"看不到任何反馈，运行进度被设置弹窗挡住 | [workbench-first-report-button-gives-no-feedback](archive/2026-10-21-workbench-first-report-button-gives-no-feedback.md) |
| EXP-03 | 已修复 | 已复现 | 工作台首次生成固定用"今天"，今天未公告时卡在"等待公告发布"，引导完不成 | [workbench-first-report-stalls-on-unannounced-today](archive/2026-10-21-workbench-first-report-stalls-on-unannounced-today.md) |
| LIB-03 | 已修复 | 已复现 | 文献库里只要有一个文件名带反斜杠，整库扫描和索引都失败，报错不指出是哪个文件 | [library-scan-fails-on-backslash-file-names](archive/2026-10-21-library-scan-fails-on-backslash-file-names.md) |
| LIB-04 | 已修复 | 已复现 | 标题检索临时失败（限流、503、断网）后，PDF 被永久记为未识别，不再重试 | [unresolved-pdfs-never-retried-after-arxiv-failure](archive/2026-10-21-unresolved-pdfs-never-retried-after-arxiv-failure.md) |
| LIB-05 | 已修复 | 已复现 | 识别 PDF 时先把压缩流完整解压再检查大小，一个很小的异常 PDF 就能耗尽内存 | [pdf-identification-inflates-streams-without-output-cap](archive/2026-10-21-pdf-identification-inflates-streams-without-output-cap.md) |
| OBS-01 | 已修复 | 已复现 | 开启调度器时选"Skip today"，把今天已完成的运行记录改成"已跳过"，向导退回 | [skip-today-overwrites-completed-run](archive/2026-10-21-skip-today-overwrites-completed-run.md) |
| OBS-02 | 已修复 | 已复现 | 日历把"日报被删的已完成日期"画成可运行，点击只回"skipped: already done" | [calendar-rerun-of-deleted-report-refused](archive/2026-10-21-calendar-rerun-of-deleted-report-refused.md) |
| OBS-04 | 已修复 | 已复现 | 取消文献库扫描、索引、方向生成、PDF 下载、详情摘要后，状态栏一直停在"Cancelling…" | [cancelled-task-leaves-status-bar-stuck](archive/2026-10-21-cancelled-task-leaves-status-bar-stuck.md) |
| OBS-05 | 已修复 | 已复现 | 调度器后台跑完日报后，已打开的 Dashboard 不刷新，一直显示"Running…" | [dashboard-not-refreshed-after-scheduled-run](archive/2026-10-21-dashboard-not-refreshed-after-scheduled-run.md) |
| MAIL-01 | 已修复 | 已复现 | 崩溃留下的空 claim 文件，让之后每天的自动日报邮件都发不出去 | [delivery-claim-zero-length-sidecar-wedges-email](archive/2026-10-21-delivery-claim-zero-length-sidecar-wedges-email.md) |
| MAIL-02 | 已修复 | 已复现 | 确定没发出去的失败（断网、DNS 失败、中继维护）被记成"结果不明"，当天邮件永不再发 | [email-known-unsent-failures-block-digest-forever](archive/2026-10-21-email-known-unsent-failures-block-digest-forever.md) |
| MAIL-03 | 已修复 | 已复现 | 中继文档里的部署命令丢掉 `BUILD_IDENTITY`，部署后自动邮件和验证链接全部失效，测试发送却仍成功 | [relay-documented-deploy-command-locks-relay](archive/2026-10-21-relay-documented-deploy-command-locks-relay.md) |
| MAIL-04 | 已修复 | 代码推断 | 官方代发的 device token 签发满一年即失效，日报邮件静默停发 | [hosted-token-expires-after-one-year-silently](archive/2026-10-21-hosted-token-expires-after-one-year-silently.md) |
| OBS-08 | 已修复 | 代码推断 | `data.json` 不是合法 JSON 时，插件悄悄按默认设置启动，下次保存就覆盖原文件，设置和密钥无声丢失（最初推断为"插件加载失败"、低，修复时对照 Obsidian 源码更正） | [corrupt-data-json-blocks-plugin-load](archive/2026-10-21-corrupt-data-json-blocks-plugin-load.md) |

## 中：测试与发版工具

| 编号 | 状态 | 置信度 | 问题 | 文件 |
|---|---|---|---|---|
| SCR-01 | 未修复 | 已复现 | 文档里的发版探索命令不构建 Obsidian 插件，会悄悄探索一份过期插件 | [exploration-obsidian-host-skips-plugin-build](2026-10-21-exploration-obsidian-host-skips-plugin-build.md) |
| SCR-02 | 未修复 | 已复现 | Obsidian 探索任务 `library-edit-candidate` 永远阻塞（插件返回格式已变，探索驱动没跟上） | [exploration-edit-candidate-task-reads-stale-plugin-api](2026-10-21-exploration-edit-candidate-task-reads-stale-plugin-api.md) |
| SCR-05 | 未修复 | 代码推断 | Windows 上的验收/探索夹具不隔离 `%APPDATA%`，会读写用户真实的 CLI 配置 | [acceptance-fixture-isolation-ignores-windows-appdata](2026-10-21-acceptance-fixture-isolation-ignores-windows-appdata.md) |
| SCR-06 | 未修复 | 代码推断 | 随 CLI 安装到用户机器的 PDF/向量运行时锁文件，没有依赖更新提醒，也不在漏洞审计里 | [shipped-runtime-lockfiles-not-audited-or-updated](2026-10-21-shipped-runtime-lockfiles-not-audited-or-updated.md) |
| SCR-09 | 未修复 | 已复现 | Obsidian 探索里点一次"Download PDF"，当前任务和整次运行都被判失败 | [exploration-download-pdf-click-fails-whole-run](2026-10-21-exploration-download-pdf-click-fails-whole-run.md) |
| EXP-04 | 未修复 | 已复现 | Obsidian 探索看不到插件之外的变化（打开笔记、PDF），"Open PDF"也没被列为受限控件 | [exploration-obsidian-observer-blind-outside-plugin-surface](2026-10-21-exploration-obsidian-observer-blind-outside-plugin-surface.md) |
| EXP-05 | 未修复 | 已复现 | "只用键盘"的探索角色看不到焦点，在 Obsidian 里被带出插件后无路可回 | [exploration-keyboard-only-persona-cannot-see-focus](2026-10-21-exploration-keyboard-only-persona-cannot-see-focus.md) |

## 低

| 编号 | 状态 | 置信度 | 问题 | 文件 |
|---|---|---|---|---|
| PIPE-08 | 已修复 | 已复现 | 日报写出后索引步骤失败，重试走修复分支，当天邮件永不发送 | [repair-path-never-delivers-email](archive/2026-10-21-repair-path-never-delivers-email.md) |
| PIPE-09 | 未修复 | 已复现 | 标题和摘要里的 Markdown 远程图片、嵌入原样写进日报，打开日报就向外部网址发请求 | [report-keeps-remote-images-and-embeds-from-titles](2026-10-21-report-keeps-remote-images-and-embeds-from-titles.md) |
| STATE-03 | 未修复 | 已复现 | 设置的日报文件夹大小写与磁盘不一致时，面板同步删除未标记论文的索引条目 | [dashboard-sync-wipes-index-when-folder-case-differs](2026-10-21-dashboard-sync-wipes-index-when-folder-case-differs.md) |
| LIB-06 | 未修复 | 已复现 | 索引中途取消、崩溃或最后一步失败，已嵌入的论文全部作废，下次从头来 | [index-cancel-discards-all-progress](2026-10-21-index-cancel-discards-all-progress.md) |
| LIB-07 | 未修复 | 已复现 | 每次重建索引都在 Vault 里留下一整份新搜索索引，旧的从不清理 | [old-search-index-generations-never-deleted](2026-10-21-old-search-index-generations-never-deleted.md) |
| CLI-04 | 已修复 | 已复现 | 工作台自动保存、`library connect`、接受方向都重写整个 `config.toml`，注释全部丢失 | [config-writes-erase-toml-comments](archive/2026-10-21-config-writes-erase-toml-comments.md) |
| OBS-03 | 已修复 | 已复现 | 状态栏的 disabled/idle 文字不跟调度器开关走 | [status-bar-disabled-idle-text-wrong](archive/2026-10-21-status-bar-disabled-idle-text-wrong.md) |
| OBS-06 | 未修复 | 已复现 | 改"Embedding dimension"绕过设置事务：不重检索引、失败不回滚、非法值静默丢弃 | [embedding-dimension-edit-bypasses-settings-transaction](2026-10-21-embedding-dimension-edit-bypasses-settings-transaction.md) |
| OBS-07 | 已修复 | 代码推断 | 插件 README 和新手教程描述已不存在的功能 | [plugin-readme-describes-removed-features](archive/2026-10-21-plugin-readme-describes-removed-features.md) |
| MAIL-05 | 已修复 | 已复现 | 各种自动邮件失败原因都只显示同一句 `provider_definitive_rejection` | [email-failures-collapse-into-one-opaque-reason](archive/2026-10-21-email-failures-collapse-into-one-opaque-reason.md) |
| MAIL-06 | 已修复 | 已复现 | 中继 README 的 `/v1/deliver` 协议说明和 secret 清单已过时 | [relay-readme-describes-old-protocol](archive/2026-10-21-relay-readme-describes-old-protocol.md) |
| MAIL-07 | 已修复 | 已复现 | 邮件摘要里常见 LaTeX 记号被拼成乱码（`M_\odot` 变成 `M_odot`） | [email-latex-simplifier-garbles-common-symbols](archive/2026-10-21-email-latex-simplifier-garbles-common-symbols.md) |
| MAIL-08 | 已修复 | 已复现 | 无相关论文那天的邮件页脚指向不存在的日报，英文写成"1 papers" | [email-zero-day-footer-points-to-missing-report](archive/2026-10-21-email-zero-day-footer-points-to-missing-report.md) |
| DSH-01 | 已修复 | 已复现 | DSH 被强制结束后，它启动的工作台进程（含自动调度）继续在后台运行 | [dsh-workbench-outlives-crashed-host](archive/2026-10-21-dsh-workbench-outlives-crashed-host.md) |
| SCR-03 | 未修复 | 已复现 | 分层边界检查对相对路径的跨层导入一律放行 | [boundary-check-allows-relative-cross-layer-imports](2026-10-21-boundary-check-allows-relative-cross-layer-imports.md) |
| SCR-04 | 未修复 | 已复现 | 探索的"禁止调度开关"拦不住首次使用清单里的"Turn on daily reports"按钮 | [exploration-setup-guide-turn-on-daily-reports-not-fenced](2026-10-21-exploration-setup-guide-turn-on-daily-reports-not-fenced.md) |
| SCR-07 | 未修复 | 已复现 | `scripts/personalization` 的四个诊断脚本已无法按注释运行 | [personalization-scripts-cannot-run-as-documented](2026-10-21-personalization-scripts-cannot-run-as-documented.md) |
| SCR-08 | 未修复 | 代码推断 | 探索中模型单次回复超过 45 秒即判任务阻塞，超时无法从命令行调整 | [exploration-single-slow-model-reply-blocks-task](2026-10-21-exploration-single-slow-model-reply-blocks-task.md) |

## 相互关联

- CLI-01 与 STATE-01 是同一症状（日期卡在"运行中"）。CLI-01 是工作台 Ctrl+C 这一个触发点，STATE-01 是 core 侧的恢复策略，适合一起修。
- MAIL-02、MAIL-04、MAIL-05 都需要把自动邮件失败的原因告诉用户，可以共用一个展示入口。PIPE-08 是修复分支不发邮件，与它们不是同一个问题。
- EXP-04（"Open PDF"未受限）、SCR-04（"Turn on daily reports"未受限）、SCR-09（"Download PDF"让整次运行失败）都是探索工具受限控件分类的缺口，可以一起修。
- SCR-02 是今天 Obsidian 发版探索里唯一一个"阻塞"的原因，与产品无关。
- OBS-04、CLI-03 不是回归。它们是已归档修复（`archive/2026-10-18-status-bar-stuck-complete.md`、`archive/2026-10-18-schedule-install-cron-unrunnable.md`）没有覆盖到的另一条路径。

## 模型探索记录

四轮都使用隔离的临时环境，外部请求走固定响应，产品是当天从 `d999037` 构建的版本。证据在本地 `output/playwright/acceptance/2026-10-21-review-*`，不入库。

| 运行 | 宿主 | 内容 | 结果 | 模型调用 | 输入 token | 耗时 |
|---|---|---|---|---|---|---|
| 第 1 轮 | Reading workbench | `--release`：12 个目标任务 + 3 个用户角色 | 12 个通过，1 条疑点 | 103 | 约 74 万 | 8.2 分钟 |
| 第 1 轮 | Obsidian | `--release`：21 个目标任务 + 3 个用户角色 | 20 个通过，1 个阻塞（SCR-02），4 条疑点 | 119 | 约 63 万 | 12.3 分钟 |
| 第 2 轮 | Reading workbench | `library-revoke` + 3 个用户角色 | 1 个通过，3 条疑点 | 48 | 约 33 万 | 4.3 分钟 |
| 第 2 轮 | Obsidian | `daily-paper-limit` + 3 个用户角色 | 1 个通过，2 条疑点 | 48 | 约 24 万 | 4.4 分钟 |

复核了 10 条疑点和 1 个阻塞任务，结论如下：
- 产品问题 3 条，即 EXP-01、EXP-02、EXP-03。
- 探索工具局限 5 条，合并写成 EXP-04 和 EXP-05。
- 误报 2 条。日报正文里没有站内单篇阅读入口，是已确认的分栏设计：日报和单篇阅读分开打开。
- 阻塞任务的原因是 SCR-02。

## 本次没有覆盖或没做完的部分

- 没有在 Windows 和 macOS 上实测。CLI-01 的 Windows 部分、SCR-05、STATE-03 的相关判断来自文档或代码。
- core 的 `documents/**`（PDF.js 与 sidecar 解析）只看了接口。`services/pdf.ts` 只读了代码，没有写测试。
- 只粗读了 `plugin/src/library/interest-profile-modal.ts` 和中继的 `cutover-control.ts`。
- 没有测跨进程并发写 `PaperIndexStore`，也没有测 `run-history` 的跨进程轮转。
- 以下流程没有实际运行：CLI 的 `init` 交互向导、`update` 的联网路径、文献库的 build/propose/preview，以及质量评估的 export/evaluate 端到端流程。

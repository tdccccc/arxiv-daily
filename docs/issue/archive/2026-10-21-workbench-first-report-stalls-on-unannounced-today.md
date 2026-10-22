# 阅读工作台首次生成日报固定用“今天”：今天还没公告时停在“等待公告发布”，引导完不成，也没有下一步提示

> 发现日期：2026-10-21 · 来源：模型探索（gpt-6.1-sol，high）+ 人工复核 · 编号 EXP-03
> 范围：阅读工作台“开始使用”清单第 4 步、运行状态条、日期卡片 · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-21-batch3`，提交 281941e、d24940d） · 置信度：已复现

## 问题

arXiv 每天晚上（美东）才公布当天的新论文列表。对大多数时区的用户来说，一天里有相当长的时间、以及每个周末，“今天”还没有公告。此时在阅读工作台里：

1. 设置里的“开始使用”清单第 4 步“生成第一份日报”固定去生成“今天”。运行立刻停在“等待公告发布”（0 篇论文），清单仍写着“已完成 3 / 5 步”，第 4 步永远不会变成已完成，除非用户自己想到去换日期。
2. 弹窗关掉后看到的运行状态条只有“等待公告发布 / 生成今日日报”和折叠的“查看运行详情”；展开后是英文命令行日志，例如 `date 2026-10-21 is newer than newest cs.AI /recent bucket 2026-10-09; arXiv announce page may not be available yet`。日期卡片上唯一的动作是“重新检查公告”。界面没有用中文说明“最近一次公告是 10 月 9 日”，也没有一键改成生成那一天。
3. 对首次使用的人，这等于不知道该等一等还是换日期。2026-10-21 的探索里 first-time-user 因此记录了 finding；它之后靠自己读英文日志、再到日历里点 10 月 9 日才继续下去。

Obsidian 插件在同一步专门处理了这个情况：“第一份日报要在任何一天都能成功：周末和 arXiv 公告之前的几个小时今天没有列表，所以使用不晚于今天的最近一个公告日”（`plugin/src/settings/tab.ts:2728-2781` 的 `latestAnnouncedDate`），并弹出通知说明“今天的论文还没公告，所以第一份日报用最近公告日 2026-10-09”。工作台没有对应逻辑，两个宿主的首次体验不一致。

## 证据

- 运行目录 `output/playwright/acceptance/2026-10-21-review-workbench-personas2/exploration/`（本地，不入库）：
  - `exploration-events.jsonl`：first-time-user 步骤 8-9 的观察里出现“2026-10-21 · 等待公告发布 · 重新检查公告”，状态条文字为“等待公告发布 / 生成今日日报 / 查看运行详情”，展开后的日志含上面那句英文；步骤 10-16 里模型自己点日历 10 月 9 日、点“生成日报”后才得到“已有日报 · 2 篇”。
  - `exploration-first-time-user-step-9.png`：状态条展开后的样子。
- 同一次运行的 Obsidian 对照 `output/playwright/acceptance/2026-10-21-review-obsidian-personas2/exploration/exploration-events.jsonl` 步骤 5：同样的夹具和日期，插件点“Generate first report”后弹出 “today's papers are not announced yet, so the first report uses the latest announced day, 2026-10-09…”，步骤 6 得到 “2 papers written”。

## 原因

- `apps/cli/src/workbench/web/settings.ts:445`：清单的 `generate` 动作只发 `{kind:'daily'}`，不带日期。
- `apps/cli/src/workbench/server.ts:383-386`：`parseTask` 对没有日期的日报请求固定走 `run --today`，标签“生成今日日报”，日期是今天。
- `apps/cli/src/workbench/server.ts:102-104`：运行结果为“等待公告发布”时状态记为 `pending`、`outcome` 为 `awaiting_announcement`。
- `apps/cli/src/workbench/web/settings-setup.ts:16,22` 与 `settings.ts:405-407`：第 4 步只在出现 `completed` 的发现运行后才算完成，`pending` 不算。
- `apps/cli/src/workbench/web/app.ts:677-696`：状态条对 `pending` 只显示标签和原始运行输出；`apps/cli/src/workbench/calendar.ts:88` 的日期卡片对“等待公告发布”只给“重新检查公告”。
- 可复用的数据源已经在核心里：`packages/core/src/services/recent-dates.ts` 的 `RecentDatesCache`（插件和调度器都在用），工作台目前没用它。

## 可能的解决方案

推荐：

1. 清单第 4 步的 `generate` 动作先取“不晚于今天的最近一个公告日”（用核心的 `RecentDatesCache`，取不到时退回今天，与插件一致），再用该日期发起运行，并在步骤下写一句“今天的论文还没公告，改为生成 2026-10-09 的日报”。
2. 无论从哪里发起（清单、“生成”对话框、日历），只要运行以 `awaiting_announcement` 结束，状态条和日期卡片都用中文写出“最近一次公告是 {日期}”，并给一个“改为生成 {日期} 的日报”按钮；原始英文日志仍保留在“查看运行详情”里。

备选：只做第 2 点。改动面小，但首次引导仍要多一次手动换日期。

## 修复后应补的测试

- `apps/cli/tests/workbench-settings-setup.test.ts`：今天晚于最近公告日时，点击 `[data-setup-action="generate"]` 发出的请求带的是最近公告日；取不到公告日时发今天。
- `apps/cli/tests/workbench-server.test.ts`：`api/runs` 对带日期的请求按该日期运行（现有行为不变），以及新增的“取最近公告日”分支。
- `apps/cli/tests/workbench-ui.test.ts`：运行以 `pending`/`awaiting_announcement` 结束时，状态条出现中文说明和“改为生成”按钮，点击后发起该日期的运行。

## 修复情况

首次日报不再固定生成“今天”。设置里的“生成第一份日报”请求 `api/runs` 时带 `latest: true`，工作台服务端用 core 的 `RecentDatesCache`（与 Obsidian 首次日报同一个数据源）读取 arXiv 的 /recent 列表：今天已公告就生成今天，否则生成最近一次公告日，并让这一步写明“今天的论文还没公告，第一份日报改用最近一次公告日 …”；读不到列表（离线或 10 秒内没有结果）时仍用今天，既不阻塞也不会让运行失败。任何日报运行以“等待公告”结束时，服务端会在标记结束之前查出最近一次公告日，任务面板、日历里该日期的卡片和阅读区标题用中文写出“{日期} 的论文还没公告，最近一次公告是 {日期}”，并给出一键“改为生成 {日期} 的日报”；原始英文日志仍在“查看运行详情”里。查不到最近公告日、或等待状态来自更早的运行时，界面与以前相同（只有“重新检查公告”）。代价：点击后服务端要先读一次 arXiv 的列表，通常几秒，最长 10 秒。

测试：`apps/cli/tests/workbench-server.test.ts`（带 `latest` 的请求按最近公告日运行并返回 `fellBackFrom`；今天已公告、查不到、查询出错时仍运行今天；不带 `latest` 的请求不查询；运行以等待公告结束时附上最近公告日，且在状态变为结束之前；查询失败不影响运行结束）；`apps/cli/tests/workbench-announced-day.test.ts`（用桩 arXiv 列表测真实的查询实现）；`apps/cli/tests/workbench-settings-setup.test.ts`（请求体与说明文字）；`apps/cli/tests/workbench-ui.test.ts`、`workbench-calendar-ui.test.ts`、`workbench-calendar.test.ts`、`workbench-reading-language.test.ts`（任务面板、日历卡片、阅读区标题的中英文提示与按钮，点击后发起该日期的运行）。

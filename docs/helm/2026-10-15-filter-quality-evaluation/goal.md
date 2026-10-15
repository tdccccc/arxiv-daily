# Filter quality evaluation

status: active
created: 2026-10-15T10:35:00+08:00
updated: 2026-10-15T10:50:00+08:00
revision: 2
owner: claude-code-session-2026-10-15

## Intent

让日报筛选"选得准不准"可以被反复、可比地测量：每天的全部筛选决定（包括被判不相关的论文）在日报完成后保留下来，研究者在一份单独的标注表上给一批论文写下"该选 / 不该选 / 属于哪个方向"，开发者用一条命令以真实模型重跑筛选并得到准确率，改提示词或换模型前后可以直接对比。

## Success criteria

- [ ] 日报成功写完后，当天全部候选论文的筛选决定（含被判不相关的）仍保留在 Vault data 中，记录每篇的方向、分数、是否进入日报，以及当时的方向列表、提示词版本和模型；CLI 和 Obsidian 插件都如此；固定测试覆盖。
- [ ] 保留的筛选决定有上限，旧记录按日期自动清理，不会无限增长；固定测试覆盖。
- [ ] 一条开发者命令能从保留的筛选决定中按日期抽样导出一份标注表（含标题、摘要、当时的模型判定），研究者可以直接在表里填写判断。
- [ ] 一条开发者命令能读取填好的标注表，用真实模型重跑筛选，输出"该选的选中了多少（召回）""选中的有多少该选（精确）""方向判对了多少"，结果连同提示词版本、模型写成一份可留存的报告；同一命令也能只对保留的历史判定打分，不调用模型。
- [ ] 至少一次真实模型基线运行：研究者标注不少于 30 篇论文，得到第一份基线报告（运行前征得用户同意费用）。

## Non-goals

- 在产品界面上增加"选对了 / 不该选"之类的判断按钮，或让研究者在产品里浏览被筛掉的论文。
- 把现有阅读标记（待读、已读、收藏、忽略）当作评估标注。ADR 0011 仍然成立：阅读行为不是研究信号。
- 用评估结果自动调整筛选（自动改提示词、自动设分数门槛）。
- 摘要质量评估（大模型评审摘要）：留到筛选评估跑顺之后另立计划。
- 详细解读选择（detail selector）的评估。
- 作为 CI 或发版门槛；普通测试和 CI 不调用模型 API。

## Constraints

- 分支 `feat/quality-evaluation`，worktree `/home/tiandc/Documents/code/arxiv-daily-quality-eval`。
- 现有 Filter checkpoint 的语义不变：它仍是日报完成前的临时记录，完成后照旧删除；保留的筛选决定是另一份独立记录，不参与重跑时的缓存复用。
- 保留记录要小：不存摘要全文和完整请求，导出标注表时按论文编号从 arXiv 取回摘要。不存 API 密钥或接口地址明文。
- 写入保留记录失败只记警告，不得让已经成功的日报变成失败。
- 评估工具放在 `scripts/` 下，作为开发者工具，不进入 CLI 或插件的用户命令；复用 core 里现有的筛选请求构造和结果校验，不复制一套平行实现。
- 标注表记录标注时的方向列表；评估时方向列表若已变化，报告明确标出，不静默混算。
- 真实模型运行会产生费用，每次运行前先征得用户同意。
- 实现块尽量交给 subagent；subagent 不再派生 subagent，结论写进报告文件；owner 验收。
- Commits: checkpoint commits authorized（用户 2026-10-15 授权：每个验收通过的块单独提交，阶段切换时提交 Helm 文件）。不推送，推送与 PR 需用户逐次同意。

## Phases

<!-- Single source of truth for phase status. PN ↔ filename NN. Integer IDs are permanent and never reused; replacement phases take the next unused number. Outcomes only — no steps. The active line is the current focus. -->
1. P1 — 日报完成后保留当天全部筛选决定，有日期上限，CLI 与插件一致 — status: active
2. P2 — 开发者评估工具：导出标注表、按标注用真实模型重跑或给历史判定打分、输出可对比的报告 — status: pending
3. P3 — 研究者标注不少于 30 篇并完成第一次真实模型基线运行 — status: pending

## Open questions

- 保留天数默认值：暂定 180 天，P1 中按一天候选数量估算体积后确认。
- P3 的候选从哪里来：P1 上线后需要先跑几天日报积累记录；若等不及，P2 可加"按日期直接取 arXiv 当天新论文"作为补充来源。

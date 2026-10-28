# P2 — llm-judge

goal_ref: ../goal.md
created: 2026-10-28T23:50:00+08:00
updated: 2026-10-28T23:50:00+08:00
revision: 1

## Outcome

一条开发者命令对按固定种子抽样的日报条目和详细笔记运行大模型评审：评审看到与生成时同口径的原文和被评摘要，按准确、完整、清楚三项给出判定和带原文引用的问题，结果可续跑、可留存为中文报告；评审模型单独配置且必须与生成模型不同；已用真实评审模型试跑过少量条目。

## Assumptions

- P1 的原文缓存（`~/quality-eval/summary-sources`）和 `rebuildDailyInput` / `rebuildDetailInput` 能还原生成时的模型输入；评审把同一份 `<paper_data>` 当作"模型当时能看到的全部原文"。
- 让评审逐字引用摘要原句和原文依据，并由程序核对引用确实存在，可以显著减少评审自己编造问题；引用对不上的判定作废或重试一次。
- 评审不需要看到 P1 的"疑似无出处数字"；保持独立，事后交叉对照，顺带检验评审。
- 研究者会提供评审模型的 `[llm]` 配置文件（0600）；真实试跑前征得费用同意。

## Approach

在 `scripts/quality-eval/` 加 summary-judge 系列模块：评审提示词（`.md`，构建时内联）与严格 JSON 合同 → 抽样与输入构造 → 配置、预览/确认、逐条缓存的运行循环 → 中文报告。全部用假模型测试；最后用真实评审模型试跑 2–3 条。

## Chunks

### Chunk 1 — 评审提示词与输出合同

- change: 评审系统提示词（中文输出）说明三项维度和"小问题 / 大问题"的界线，附上生成时的要求（直接引用当前生成提示词文本，说明格式规则已由离线检查负责、不要评格式与长度），原文放在 `<paper_data>` 中并沿用注入防护。输出严格 JSON：三项各一个判定（ok / minor / major），问题列表每项含维度、严重程度、位置（日报字段键或笔记章节标题）、摘要原句、原文依据（准确性问题中"原文未提及"可为 null 并须说明，完整性问题必须给出被漏掉内容的原文）、说明。程序校验：键与枚举、位置合法、摘要原句必须是摘要的子串、原文依据必须是原文的子串（空白规范化后）、判定与问题一致（有 major 问题则该维度为 major）。校验失败给出具体原因并重试一次，仍失败记为无效判定。
- kind: behavior
- first signal: Red — `scripts/tests/quality-eval-summary-judge-contract.test.mjs` 的合法/非法样例因模块不存在而失败。
- checks: 同一测试 Green；quality-eval 全部测试 Green。
- [ ] accepted on observed evidence

### Chunk 2 — 抽样与评审输入

- change: 从语料和原文缓存中按种子确定性抽样（`--daily N --notes M --seed`，可 `--ids` 指定）；无原文或无法还原输入的条目单列不评；每条构造评审消息（系统提示词 + 生成时的 `<paper_data>` + 被评摘要：日报五字段或笔记正文）。
- kind: behavior
- first signal: Red — 抽样确定性、跳过无原文、消息内容与 `rebuildDailyInput` 一致等测试失败。
- checks: 同一测试 Green；全部测试 Green。
- [ ] accepted on observed evidence

### Chunk 3 — 评审配置、预览/确认与可续跑运行

- change: `node scripts/quality-eval/summary-judge.mjs --judge-config <toml> [--config <cli toml>] --sources <dir> --out <dir> ...`。评审配置只读 `[llm]`（与 acceptance 的 `--model-config` 同格式，尽量复用其读取函数）；评审模型名与生成模型（CLI 配置）相同则拒绝。不带 `--yes` 只打印条目数、请求数、输入字符总量和模型名后退出；带 `--yes` 逐条调用，每条结果写入 `--out/judgments/` 并可续跑；单条失败记录原因不中断；报告与日志不含接口地址和密钥。
- kind: behavior
- first signal: Red — 假模型下的运行测试（预览不发请求、续跑跳过已评、同模型拒绝、失败单列、输出不含密钥）失败。
- checks: 同一测试 Green；全部测试 Green。
- [ ] accepted on observed evidence

### Chunk 4 — 评审报告

- change: `report.md`（中文）：评审模型与时间、抽样参数、三项判定分布（日报 / 笔记分开）、按字段或章节的问题数、问题清单（摘要原句、原文依据、说明）、无效判定数；可选 `--offline <findings.json>` 时交叉对照 P1 的"疑似无出处数字"与评审的准确性问题（同一条目同一字段是否都被指出）。`judgments.json` 存全部数据。README 增补用法。
- kind: behavior
- first signal: Red — 报告内容测试失败。
- checks: 同一测试 Green；`npm run test:release-tools` Green。
- [ ] accepted on observed evidence

## Phase verification

- 用研究者提供的评审模型试跑 2–3 条（日报 + 笔记各至少 1 条，先征得同意），报告生成，引用校验通过率和判定内容由 owner 读一遍确认合理。
- `npm run test:release-tools` Green。

## Abort / reshape triggers

- 如果真实评审模型的引用校验大量失败（试跑中超过一半判定无效），先调整提示词与规范化（L1）；仍不行则放宽为"引用必须能模糊匹配"并在报告注明（L2）。
- 如果评审模型的上下文装不下详细笔记的原文（约 10 万字符）加笔记本身，L1 改为截断说明或换更大上下文的模型，需与研究者商量。

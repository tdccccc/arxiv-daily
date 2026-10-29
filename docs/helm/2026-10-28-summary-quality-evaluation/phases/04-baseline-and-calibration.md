# P4 — baseline-and-calibration

goal_ref: ../goal.md
created: 2026-10-29T01:50:00+08:00
updated: 2026-10-29T23:00:00+08:00
revision: 7

## Outcome

40 条日报条目和 10 篇详细笔记已在缓存原文上用当前配置重新生成并由评审模型评审，报告留存在 `~/quality-eval/summary-baseline-2026-10-29/`；研究者在中文核对网页上对不少于 10 条评审判定给出同意 / 不同意，一致率写进核对报告。

## Assumptions

- 研究者选择了 40 + 10 的规模（估算：重新生成约 175 万输入 token，评审约 250 万输入 token），视为同意这个量级的费用；实际预览超出估算时再问。
- 当前配置下问题可能很少；不足 10 条时，用 P2 试跑（历史摘要）中评审指出的问题补足——它们同样能检验评审判得对不对。
- 研究者要求评审用中文：说明必须是中文，程序校验。

## Approach

先补中文校验（会改变评审提示词或合同，必须在评审基线之前合入），并行在后台重新生成；再做中文核对网页与导入计算；然后评审基线、生成核对页给研究者；最后导入核对结果并收尾。

## Chunks

### Chunk 1 — 评审说明必须是中文

- change: 合同校验要求每个问题的 "explanation" 含中文（汉字占比达到一个下限，公式与引号内英文不计），不满足时给出具体原因走一次纠正重试；提示词里再强调一次"说明一律用中文，英文术语可保留"。
- kind: behavior
- first signal: Red — 英文说明的判定被接受（测试期望拒绝）。
- checks: 同一测试 Green；quality-eval 全部测试 Green。
- [x] accepted on observed evidence (256/256 tests, Red 3/4 first; all 37 real trial explanations have Han share ≥ 0.60 against the 0.3 threshold)

### Chunk 2 — 中文核对网页与一致率

- change: `summary-review.mjs --judgments <评审输出目录> [--extra <另一个评审输出目录>] --count N --seed --out <dir>` 生成一个自包含的中文网页 `review.html`（离线打开、无外部依赖）：逐条显示论文标题与编号、位置、维度、严重程度、摘要原句、原文依据（并附原文中该句前后的上下文）、评审说明，研究者点"同意 / 不同意 / 拿不准"并可写备注；进度保存在浏览器本地，可导出结果 JSON。`--import <结果 JSON>` 计算一致率（总体、按维度、按严重程度），写 `calibration.md`。优先从 `--judgments` 抽，不够 N 条用 `--extra` 补，来源在页面和报告中标明。
- kind: behavior
- first signal: Red — 抽题、页面内容、导入计算的测试失败。
- checks: 同一测试 Green；`npm run test:release-tools` Green。
- [x] accepted on observed evidence (272/272 quality-eval tests, Red 14/14 first; test:release-tools 1180/1180; owner opened the sample page in headless Chrome at 1200px and 390px: no errors, no horizontal scroll, buttons and progress work)

### Chunk 3 — 基线评审

- change: 用评审模型评审后台重新生成的 50 条（`--summaries`），报告写到 `~/quality-eval/summary-baseline-2026-10-29/judged/`；owner 读报告，抽查 3 条判定。
- kind: non-behavioral（真实运行）
- first signal: n/a
- checks: 评审完成数、无效数、错误数；预览规模不超过估算。
- [x] accepted on observed evidence (50/50 regenerated with 0 fallbacks; 50/50 judged valid, 47 first try and 3 after retry; 11 transient proxy cut-offs at ~24 s succeeded on resume; owner read the report and checked the major and accuracy issues)

### Chunk 4 — 第二模型复核全部评审问题

- change: `summary-meta-review.mjs --judged <基线评审目录> --sources <dir> --config <cli toml> --reviewer-model <name> --reviewer-effort <effort> --out <dir> [--yes]`：复核模型用 CLI 配置的接口，覆盖模型名与推理强度（研究者指定 gpt-6-astra、xhigh），必须与评审模型不同。每篇有问题的条目一次请求：给出同一份 `<paper_data>`、被评摘要和评审的问题列表，复核模型逐条给出 同意 / 不同意 / 拿不准、严重程度是否合适（合适 / 偏重 / 偏轻）、中文理由，可附原文引用（程序核对为原文子串）。严格 JSON 合同、一次纠正重试、预览/确认、逐条可续跑、无密钥泄漏；写 `meta-review.md`（复核一致率：总体、按维度、按严重程度；严重程度意见；分歧清单）和 `meta-review.json`。
- kind: behavior
- first signal: Red — 假模型测试（合同校验、逐条覆盖、续跑、同模型拒绝、报告计算）失败。
- checks: 同一测试 Green；`npm run test:release-tools` Green。
- [x] accepted on observed evidence (289/289 quality-eval tests, Red 17/17 first; test:release-tools 1197/1197; owner read the reviewer prompt)

### Chunk 5 — 只把分歧条目交给研究者

- change: `summary-review.mjs` 新增 `--meta <复核目录>`：卡片只取复核模型不同意或拿不准的问题，卡片上同时显示评审说明与复核理由；`--import` 的核对报告合并复核结果：复核同意的条数、分歧条目中研究者站在评审一边 / 复核一边 / 拿不准的条数。
- kind: behavior
- first signal: Red — 分歧筛选、卡片内容、合并报告的测试失败。
- checks: 同一测试 Green；`npm run test:release-tools` Green。
- [x] accepted on observed evidence (300/300 quality-eval tests, Red 10/11 first; test:release-tools 1208/1208)

### Chunk 6 — 真实复核与研究者裁定

- change: 用 gpt-6-astra（xhigh）复核基线（研究者已同意，约 45 次请求、约 180 万输入 token；预览超出时再问），生成分歧核对页，研究者裁定并导入。
- kind: non-behavioral（真实运行 + 研究者操作）
- first signal: n/a
- checks: `meta-review.md` 与 `calibration.md` 存在。
- [ ] accepted on observed evidence

## Phase verification

- 基线报告与核对报告都在评估目录；goal 的成功标准逐条打勾或记录豁免。
- `npm run test:release-tools` Green。

## Abort / reshape triggers

- 重新生成的回退率明显高于 0.6%：先查输入构造。
- 一致率低于约 70%：评审结论不可直接采信，先分析不同意的条目、调整评审提示词，再决定是否重评（L1/L2）。

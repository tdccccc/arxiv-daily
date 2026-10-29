# P4 — baseline-and-calibration

goal_ref: ../goal.md
created: 2026-10-29T01:50:00+08:00
updated: 2026-10-29T01:50:00+08:00
revision: 1

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
- [ ] accepted on observed evidence

### Chunk 2 — 中文核对网页与一致率

- change: `summary-review.mjs --judgments <评审输出目录> [--extra <另一个评审输出目录>] --count N --seed --out <dir>` 生成一个自包含的中文网页 `review.html`（离线打开、无外部依赖）：逐条显示论文标题与编号、位置、维度、严重程度、摘要原句、原文依据（并附原文中该句前后的上下文）、评审说明，研究者点"同意 / 不同意 / 拿不准"并可写备注；进度保存在浏览器本地，可导出结果 JSON。`--import <结果 JSON>` 计算一致率（总体、按维度、按严重程度），写 `calibration.md`。优先从 `--judgments` 抽，不够 N 条用 `--extra` 补，来源在页面和报告中标明。
- kind: behavior
- first signal: Red — 抽题、页面内容、导入计算的测试失败。
- checks: 同一测试 Green；`npm run test:release-tools` Green。
- [ ] accepted on observed evidence

### Chunk 3 — 基线评审

- change: 用评审模型评审后台重新生成的 50 条（`--summaries`），报告写到 `~/quality-eval/summary-baseline-2026-10-29/judged/`；owner 读报告，抽查 3 条判定。
- kind: non-behavioral（真实运行）
- first signal: n/a
- checks: 评审完成数、无效数、错误数；预览规模不超过估算。
- [ ] accepted on observed evidence

### Chunk 4 — 研究者核对

- change: 生成核对网页（不少于 10 条），研究者完成核对并导出结果，导入得到一致率。
- kind: non-behavioral（研究者操作）
- first signal: n/a
- checks: `calibration.md` 存在，核对条数 ≥ 10。
- [ ] accepted on observed evidence

## Phase verification

- 基线报告与核对报告都在评估目录；goal 的成功标准逐条打勾或记录豁免。
- `npm run test:release-tools` Green。

## Abort / reshape triggers

- 重新生成的回退率明显高于 0.6%：先查输入构造。
- 一致率低于约 70%：评审结论不可直接采信，先分析不同意的条目、调整评审提示词，再决定是否重评（L1/L2）。

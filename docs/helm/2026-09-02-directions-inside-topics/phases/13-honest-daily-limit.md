# P13 — honest-daily-limit

goal_ref: ../goal.md
created: 2026-09-07T03:46:27+08:00
updated: 2026-09-07T04:07:50+08:00
revision: 2

## Outcome

日报明确说出哪些论文因每日总数上限没有展示，主题有命中但被限额截掉时不再显示“今日无相关论文更新”。

## Assumptions

- 保留默认20篇、全局相关性排序、无主题配额和ignored先排除的既有规则。
- 未展示计数只来自确定的筛选/截断结果，不由摘要模型推断。
- 短摘要缓存不依赖日报名额；更改上限仍可复用已生成的单篇摘要。

## Approach

pipeline在排除ignored并排序截断后，按未保留论文的category统计omittedByTopic。该可选计数随SummarizerDeps进入DailySummaryAssemblyInput。正常组装、rescue合同和emergency输出使用同一计数与本地化文本：有遗漏时报告概览说明总数；空主题说明因上限未展示的数量；有保留论文的主题可说明另有遗漏。没有遗漏时旧输出逐字不变。

## Chunks

### Chunk 1 — 组装与降级输出准确表达限额遗漏

- files: daily-summary-assembler.ts、daily-summary-rescue.ts、settings/summary-language.ts；对应assembler/rescue tests。
- interface: DailySummaryAssemblyInput.omittedByTopic?: Readonly<Record<string, number>>，仅合法topic tag的非负安全整数；未提供相当于全0。
- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: topic B明明有2篇因cap未保留，正常/emergency/rescue输出必须说明遗漏而不是无相关；当前仍无相关。中英文覆盖，0遗漏保留原输出，invalid计数拒绝。
- Green check: npm run test --workspace packages/core -- tests/daily-summary-assembler.test.ts tests/daily-summary-rescue.test.ts --maxWorkers=1
- regression checks: core typecheck；不能让rescue重写成无相关或篡改数目。
- [x] implementation and tests accepted — 接线4 Red、组装23 Red后联合152 Green；全工作区、typecheck、lint、boundaries、build通过。

### Chunk 2 — 真实筛选与摘要链路传递遗漏计数

- files: pipeline.ts、summarizer.ts；pipeline-daily-cap.test.ts、summarizer.test.ts。
- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 跨主题cap场景，真实pipeline调用的摘要输入必须携带正确omittedByTopic；ignored不计遗漏。真实summarizeDaily组装保留该计数；现输入缺失。
- Green check: npm run test --workspace packages/core -- tests/pipeline-daily-cap.test.ts tests/summarizer.test.ts --maxWorkers=1
- regression checks: npm run test:workspaces、npm run typecheck、npm run lint、npm run check:boundaries、plugin build。
- [x] implementation and tests accepted — 接线4 Red、组装23 Red后联合152 Green；全工作区、typecheck、lint、boundaries、build通过。

## Phase verification

- 正常与异常组装均保留遗漏解释，未改变保留论文集合、详情资格、摘要缓存或索引引用。
- 在P6生成的真实日报中查看来源；限额边界通过可控pipeline回归验证，不要求真实某天恰好每主题被挤满。

## Abort / reshape triggers

- 遗漏计数开始参与筛选/评分或引入主题配额：停止，恢复为纯结果说明。
- 摘要缓存被迫随名额变更失效：重新划清单篇摘要与整份日报装配的边界。

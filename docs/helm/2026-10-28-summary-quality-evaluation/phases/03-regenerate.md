# P3 — regenerate

goal_ref: ../goal.md
created: 2026-10-29T00:55:00+08:00
updated: 2026-10-29T01:15:00+08:00
revision: 2

## Outcome

一条开发者命令在缓存原文上，用当前提示词和生成模型重新生成抽样论文的日报条目和详细笔记；评审命令可以评这些重新生成的摘要，报告写明被评摘要的来源（历史或重新生成，以及生成模型和提示词哈希）。已对少量论文真实跑通"重新生成 → 评审"。

## Assumptions

- 重新生成走日报流水线的同一代码路径：日报条目用 core 的带校验重试的单篇摘要函数（与 pipeline 相同的重试与回退），详细笔记用 `summarizePaperDetail`；输入由 P1 的缓存原文按同样预算构造，与 `rebuildDailyInput` / `rebuildDetailInput` 一致。
- 生成模型来自 CLI 配置的 `[llm]`（gpt-6.1-sol），研究者的主题名称在插件设置里，CLI 配置不一定有；详细笔记提示词里的主题名缺失时用笔记的主题 tag。
- 抽样沿用 P2 的抽样函数和种子，所以同一种子下，重新生成和评审覆盖同一批论文。
- 重新生成的日报字段是模型原始 JSON 的值（未经报告渲染）；评审看原始值即可。

## Approach

新增 summary-regenerate 模块与命令（抽样 → 预览 → 确认后逐条生成 → 逐条缓存，可续跑），评审命令新增"评重新生成的摘要"的输入来源；历史模式下给多版本论文打标；最后真实小规模跑通。

## Chunks

### Chunk 1 — 重新生成命令

- change: `node scripts/quality-eval/summary-regenerate.mjs --config <cli toml> --vault ... --sources <dir> --out <dir> [--daily N --notes M --seed --ids --from --to --include-older-notes] [--yes] [--regenerate]`。不带 `--yes` 只预览（条目数、最多请求数、输入字符量、生成模型名），不调用不写入；带 `--yes` 逐条生成，每条写 `<out>/summaries/<kind>-<id>.json`（生成模型与推理设置、生成提示词哈希、原文内容哈希、输出：日报为五字段或回退原因，笔记为 Markdown、尝试次数、用量、耗时），原子写入；同一生成模型、提示词哈希、原文哈希的已有结果跳过，`--regenerate` 重做；连续 3 次错误停止；`run.json` 记录参数与抽样。输出不含接口地址和密钥。
- kind: behavior
- first signal: Red — 假模型测试（预览不调用、日报走校验重试与回退、笔记生成、续跑跳过、提示词变化重做、连续错误停止、无密钥泄漏）因模块不存在而失败。
- checks: 同一测试 Green；quality-eval 全部测试 Green。
- [x] accepted on observed evidence (240/240 quality-eval tests, Red 16/17 first; test:release-tools 1148/1148; transport-exhausted recorded as error so resume retries it)

### Chunk 2 — 评审重新生成的摘要

- change: 评审命令新增 `--summaries <regenerate out dir>`：被评对象改为重新生成的摘要（日报五字段 / 笔记章节），回退的条目单列不评；评审记录与报告写明被评摘要来源（历史 Vault 或重新生成，及其生成模型、提示词哈希、生成时间）；续跑的判断加上被评摘要的内容哈希。历史模式下，缓存原文版本高于 v1 的条目在报告中标"原文已有新版本，生成时可能看到的是旧版"。
- kind: behavior
- first signal: Red — 用假的重新生成结果构造评审输入、回退单列、来源写入报告、历史模式多版本标记的测试失败。
- checks: 同一测试 Green；`npm run test:release-tools` Green。
- [ ] accepted on observed evidence

## Phase verification

- 征得研究者同意后，对 2605.20779、2608.19669、2607.01175 三篇重新生成（日报条目 3 条 + 笔记 1 篇），再用评审模型评这 4 条；owner 读评审结果，与 P2 试跑（历史摘要）对照，确认"输入一致"后剩下的问题是否站得住。
- `npm run test:release-tools` Green。

## Abort / reshape triggers

- 如果 core 的单篇摘要函数无法脱离流水线调用（依赖流水线状态），先在脚本里用最薄的适配层调用；若必须改 core，只限导出，且先与研究者确认。
- 如果重新生成的回退率明显高于日报实际回退率（0.6%），先查是不是输入构造与流水线不一致，再继续。

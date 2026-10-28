# P1 — offline-checks

goal_ref: ../goal.md
created: 2026-10-28T22:50:00+08:00
updated: 2026-10-28T23:15:00+08:00
revision: 4

## Outcome

一条开发者命令扫描一个 Vault 里已生成的日报条目和详细笔记，不调用模型，写出一份可留存的报告（格式、回退、长度、"原文未说明"、空话、数字可溯源），并已对真实 Vault 跑过一次。

## Assumptions

- 评估对象：五字段日报（2026-06-12 起，约 470 条）和当前格式详细笔记（含"## 研究问题"，约 46 篇，均为 2026-07 以前；研究者手动关闭了详细笔记，近期没有新的）。旧格式日报和旧格式笔记只计数，不评。
- 日报里的字段是渲染后的文本（空白压成一行、`<` 与 `[[` 被转义），检查必须在这个口径上工作，不追溯模型原始 JSON。
- 生成时的原文没有保存；数字可溯源检查只能用评估时重新抓取的原文（arXiv 可能已有新版本），报告需说明这一点。
- 报告和笔记都不记录生成模型与提示词版本，报告只能写评估时的提示词文件哈希和数据日期范围。
- 研究者的 Vault 在 `/home/tiandc/Nextcloud/work/Notes`（日报目录 `arxiv-daily/daily`，笔记目录 `arxiv-daily/papers`），不是 CLI 配置里的 `vault_root`。

## Approach

在 `scripts/quality-eval/` 旁加 summary 系列模块，共用现有 `load.mjs` / `index.ts` 的编译方式和 `scripts/tests/` 的测试方式；日报解析复用 core 的 `parseDailyReportPapers`，公式检查复用 `canonicalizeScientificMarkdownMath`，原文抓取复用日报流水线的 arXiv 内容抓取（与生成同口径的预算）。先做纯离线部分（块 1–4），再做抓原文和数字核对（块 5–6）。

## Chunks

### Chunk 1 — 从 Vault 读出可评估的日报条目和详细笔记

- change: 给定 Vault 根目录（或 CLI 配置）、日报/笔记目录和可选日期范围，读出每条五字段日报条目（日期、主题、id、标题、五字段）、回退条目、当前格式详细笔记（id、主题、各章节正文），并对旧格式日报/笔记计数而不评估。
- kind: behavior
- first signal: Red — 新的 `scripts/tests/quality-eval-summary-*.test.mjs` 用夹具 Vault（平铺版式、折叠 callout 版式、早期标签、旧格式、回退条目、旧/新格式笔记）断言读出的记录，因模块不存在而失败。
- checks: 同一测试 Green；`node --test scripts/tests/quality-eval-*.test.mjs` 全部 Green。
- [x] accepted on observed evidence (71/71 quality-eval tests; real vault: 469 entries, 3 fallbacks, 46 current notes, 72 legacy notes)

### Chunk 2 — 日报条目的离线检查

- change: 每条日报条目产出：回退与否；公式问题（复用 core 公式校验，另加"相邻拆开的 `$..$ $..$` 片段"、`\(..\)`、`$$` 检测）；各字段长度（去掉公式与空白后的字符数），核心结果字段落在 80–180 之外标记；各字段"原文未说明"；提示词禁止的空话（"具有重要意义""提高了理解"等）；汇总为比例和分布。
- kind: behavior
- first signal: Red — 每项检查各有正反例测试，因检查不存在而失败。
- checks: 同一测试 Green；既有 quality-eval 测试仍 Green。
- [x] accepted on observed evidence (87/87 tests; Red recorded after the fact by moving the module aside; real vault: fallback 3/472, 0 math or generic-phrase findings, mainResult outside 80–180 in 298/464)

### Chunk 3 — 详细笔记的离线检查

- change: 每篇笔记产出：按章节标题集合分组（46 篇来自几版提示词：结果节有"核心结果/关键证据"，末节有"学术价值判断/阅读价值/一句话价值判断"，共 4 种顺序），报告各组篇数；对照当前提示词的八个章节给出缺失/多余/顺序，旧版本组单独汇总而不算作违规；各章节"原文未说明 / 原文未讨论"；公式问题（允许 `$$` 独立公式，禁止 `\(..\)` `\[..\]`、相邻拆开片段、公式外裸 TeX）；空话；正文（公式外）英文词占比。
- kind: behavior
- first signal: Red — 正反例测试因检查不存在而失败。
- checks: 同一测试 Green；既有测试仍 Green。
- [x] accepted on observed evidence (105/105 tests, Red 17/17 before implementation; real vault: 30 current + 16 older-prompt notes, 10 notes with math findings of which 4 are core-rule false positives, see journal)

### Chunk 4 — 报告与命令

- change: `node scripts/quality-eval/summary-check.mjs --vault <dir> [--daily-dir] [--papers-dir] [--from] [--to] --out <dir> [--force]` 写出 `report.md`（中文，含评估范围、提示词文件哈希、各项比例、问题条目清单）和 `findings.json`；已有报告默认不覆盖；不调用模型、不联网；README 增补用法。
- kind: behavior
- first signal: Red — 命令级测试（夹具 Vault → 报告文件内容、拒绝覆盖、参数校验）失败。
- checks: 同一测试 Green；`npm run test:release-tools` 中 quality-eval 部分 Green。
- [x] accepted on observed evidence (124/124 quality-eval tests, 1030/1030 test:release-tools; trial report on the real vault in /tmp/summary-check-trial; length demoted to reference data per the researcher)

### Chunk 5 — 抓取并缓存同口径原文

- change: 对待评估的论文，按日报流水线的同一抓取路径和预算（HTML → LaTeX 源码 → abs 页）取回原文，缓存到评估目录，可中断续跑，遵守现有 arXiv 限流/冷却；取不到的单列。供块 6 与 P2 评审共用。
- kind: behavior
- first signal: Red — 用假 HTTP 的测试断言缓存内容、续跑不重复请求、失败单列，因模块不存在而失败。
- checks: 同一测试 Green；测试不联网。
- [ ] accepted on observed evidence

### Chunk 6 — 数字可溯源检查

- change: 从日报字段和笔记正文里抽出数值（含公式内数值，规范化 ±、负号、千分位、科学计数、百分号），在缓存原文中查找；找不到的列入"疑似无出处数字"清单（带上下文），并给出每条/每篇的比例；无原文的条目不参与该项。
- kind: behavior
- first signal: Red — 正反例测试（原文有/无、格式不同但数值相同）失败。
- checks: 同一测试 Green；既有测试仍 Green。
- [ ] accepted on observed evidence

## Phase verification

- 对真实 Vault 全部历史跑离线检查，报告写到 `~/quality-eval/summary-<日期>/`，条目数与预期（约 470 条日报、约 46 篇笔记）一致。
- 抓取全部可评估论文的原文（只访问 arXiv，不调用模型）后跑数字核对；owner 人工抽查 10 条"疑似无出处数字"，记录误报率到 journal。
- `npm run test:release-tools` Green。

## Abort / reshape triggers

- 如果渲染后的日报文本让 core 公式校验大面积误报（转义导致），公式检查改为只用新增的正则检查，并在报告注明（L1）。
- 如果重新抓取的原文大比例取不到或与生成时明显不同（大量论文已更新版本），数字核对降级为参考项并在报告注明；若影响到 P2 评审的同口径前提，L3 重新讨论评审输入。
- 如果数字核对的抽查误报率高于约一半，先收紧规范化规则（L1），仍不行就把它从指标降为人工参考清单。

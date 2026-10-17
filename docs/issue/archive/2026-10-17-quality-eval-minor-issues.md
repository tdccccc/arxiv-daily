# 质量评估脚本的几处小问题：F1 显示、标注表编号和日期校验、Markdown 转义、日期参数校验

> 发现日期：2026-10-17 · 来源：main `e6eaed5`（#93–#95 合并后）代码审查，质量评估脚本部分由子代理审查、主会话核对
>
> 范围：开发者工具 `scripts/quality-eval/`，不影响产品 · 严重程度：低（各小节分别说明） · 状态：未修复 · 置信度：见各小节

下面四个问题互相独立，改动都很小，可以一起修。为了让每个问题自成一体，每节都按"问题、原因、复现、可能的解决方案、修复后应补的测试"展开。

## 1. 一篇都没选中、但确有相关论文时，F1 显示为 n/a，应为 0

严重程度：低 · 置信度：已确认（用 `metrics.ts` 的实际代码核算）

### 问题

筛选（记录的判定或某次重跑）一篇都没选中、而已标注行里确有 `y` 时，精确率无定义（显示 n/a），召回率为 0%，F1 也显示 n/a。按 F1 = 2TP / (2TP + FP + FN)，此时 TP = 0、FN > 0，F1 应为 0。反过来，没有任何 `y` 但选中了论文时，召回率无定义，F1 也应为 0。只有 TP、FP、FN 全为 0 时 F1 才无定义。显示 n/a 会掩盖"重跑完全失效"这类结果。

### 原因

`scripts/quality-eval/metrics.ts:80-84` 的 `f1Of`：精确率或召回率任一为 `null` 就返回 `null`，尽管下一行本来就按 2TP / (2TP + FP + FN) 计算。加权 F1 也经由 `summarize`（`metrics.ts:98-102`）调用同一函数。

### 复现

用 `metricSet` 实际计算：4 个跳过层行（2 个 `y`、2 个 `n`），没有选中行。结果精确率 n/a、召回率 0.0%、F1 n/a，加权 F1 也是 n/a。

### 可能的解决方案

1. **推荐：** `f1Of` 只看计数，分母 2TP + FP + FN 为 0 时返回 `null`，否则返回 2TP / 分母。
2. 保留 `null`，在 `report.ts` 渲染时对"有相关论文但一篇未选"显示"0（未选中任何论文）"。只改显示，`report.json` 里仍是 `null`，不推荐。

### 修复后应补的测试

`scripts/tests/quality-eval-evaluate.test.mjs`：一篇未选、有 `y` 时 F1 为 0；没有 `y`、有选中时 F1 为 0；TP、FP、FN 全为 0 时为 `null`。

## 2. 标注表不校验编号和日期格式，表格软件改写后会被静默接受（疑似）

严重程度：低 · 置信度：疑似（代码不做校验已确认；表格软件是否以及如何改写取决于软件和导入方式，子代理只做了模拟，没有用真实表格软件验证）

### 问题

README 让标注者用表格软件打开并填写 `labels.csv`（`scripts/quality-eval/README.md:51`）。常见表格软件可能把 `2610.10230` 这样的编号当成数字保存为 `2610.1023`，把 `2026-10-14` 改成别的日期格式。`parseSheet` 对此不报错：`id` 只检查非空（`sheet.ts:57-58`），`date` 完全不校验（`sheet.ts:75`）。

可能的影响（推断）：报告里显示的编号和日期是错的；分歧列表按日期字符串排序会乱；重跑请求里的编号随之改变（同一请求内前后一致，指标本身不受影响）。

### 原因

`scripts/quality-eval/sheet.ts:41-88` 的 `parseSheet` 只校验 `had_abstract_at_filter`、`model_score`、`label` 和方向编号，不校验 `id`、`date` 的格式，也不与 `labels.meta.json` 核对。

### 复现

按代码推断（未实际运行）：把某行的 `id` 改成 `2610.1023`、`date` 改成 `2026/10/14`，再运行 `evaluate --recorded`，不会报错，报告按改后的值显示。

### 可能的解决方案

1. **推荐：解析时校验格式。** `parseSheet` 校验 `date` 为 `YYYY-MM-DD`、`id` 为合法的 arXiv 新式编号，不符时按行报错（与现有的问题汇报方式一致）。注意 2015 年 1 月起序号是 5 位，`2610.1023` 不合法；但 core 的 `MODERN_ARXIV_ID_RE`（`packages/core/src/utils/arxiv.ts:1`）对任何年月都接受 4 或 5 位序号，直接复用它抓不到这种改写，需要按年月加一条位数规则。同时在 `labels.guide.md` 里提醒把 `id`、`date` 两列设为文本格式（`export.ts:398` 目前只提醒保存为 CSV UTF-8）。
2. 导出时在 `labels.meta.json` 里保存导出行的 `(date, id)` 列表或其摘要，评估时逐行核对，发现改写就报错。能发现任何改写，但元数据变大。
3. 导出时把编号写成不会被当成数字的形式（例如 `arXiv:2610.10230`），解析时去掉前缀。

### 修复后应补的测试

`scripts/tests/quality-eval-evaluate.test.mjs`：`id` 为 `2610.1023` 或 `date` 为 `2026/10/14` 时，`parseSheet` 抛出 `SheetError` 并指出行号。

## 3. 标注指南和报告里的主题标签、方向文本没有做 Markdown 转义

严重程度：低（只影响开发者工具的排版） · 置信度：缺少转义已确认；"标签含特殊字符时表格错列"按代码推断，未实际运行

### 问题

- `labels.guide.md` 的方向表（`scripts/quality-eval/export.ts:379-380` 的 `renderGuide`）对方向文本做了转义（反斜杠、竖线、空白），但主题标签 `entry.tag` 原样写进表格单元格。标签含 `|` 时，这一行会错列。
- 报告的方向说明行（`scripts/quality-eval/report.ts:149-150` 的 `directionLine`）把 `tag` 和 `text` 原样拼进列表项。文本含换行会打断列表，含 `*`、`_` 等字符会被当成格式。报告的分主题表已经用 `cell()` 转义标签（`report.ts:13-15`、`report.ts:104`），这两处没有用。

设置校验只要求主题标签非空且不重复（`packages/core/src/settings/validation.ts:139-145`），不限制字符。

### 原因

两处拼接字符串时没有复用 `report.ts` 里已有的 `cell()` 转义函数。

### 复现

按代码推断（未实际运行）：把某个主题的标签设为 `a|b`，跑出筛选记录后导出，`labels.guide.md` 方向表的这一行会多出一列。

### 可能的解决方案

1. **推荐：** 把 `cell()` 移到共享位置（例如新建 `scripts/quality-eval/markdown.ts`），`renderGuide` 对 `tag` 也调用它；`directionLine` 对 `tag` 和 `text` 先合并空白再转义。
2. 把报告的方向说明改成表格，统一用 `cell()` 渲染，与标注指南的方向表一致。

### 修复后应补的测试

标签含 `|`、方向文本含换行时，`labels.guide.md` 与 `report.md` 的表格列数正确，列表不被打断。

## 4. 日期参数：不存在的日期报出难懂的错误，起止日期颠倒也不检查

严重程度：很低 · 置信度：已确认（代码核对，并用 node 验证了报错）

### 问题

- `--from 2026-13-01` 这类格式正确但不存在的日期，报出的是 `Invalid time value`（入口 `export.mjs:29` 会打印为 `error: Invalid time value`），而不是脚本准备好的提示 "--from must be a date like 2026-10-14"。
- `--from` 晚于 `--to` 时参数解析不报错；随后找不到任何记录，报 "no filter decision records found under <vault> for <from> to <to>"（`export.ts:215-216`），没有指出是起止日期颠倒。

### 原因

- `scripts/quality-eval/export.ts:116-120` 的 `requireDate`：正则通过后直接调用 `new Date(...).toISOString()`，无效日期在这里就抛出 `RangeError`，走不到后面的友好提示。已用 node 确认 `new Date("2026-13-01T00:00:00Z").toISOString()` 抛出 `RangeError: Invalid time value`。
- `export.ts:84-114` 的 `parseExportArgs` 不比较 `from` 和 `to`。

### 复现

`node scripts/quality-eval/export.mjs --out /tmp/qe --from 2026-13-01`：按入口代码推断，会输出 `error: Invalid time value` 和用法说明。

### 可能的解决方案

1. **推荐：** `requireDate` 先构造 `Date` 并检查 `Number.isNaN(date.getTime())`，无效时抛出友好提示；`parseExportArgs` 结束前若 `from > to` 直接报错。
2. 只修 `requireDate`；起止日期颠倒时，在"找不到记录"的错误里补一句"请检查 --from 是否晚于 --to"。

### 修复后应补的测试

`scripts/tests/quality-eval-export.test.mjs` 的 "command-line arguments are validated" 用例（第 413 行起）补两条：`--from 2026-13-01` 得到友好提示；`--from 2026-10-15 --to 2026-10-14` 报起止日期颠倒。

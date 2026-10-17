# 标注表 id 校验只覆盖形如现代 arXiv 编号的值，形状不符的改写会被静默接受

> 发现日期：2026-10-17 · 来源：PR #96（修复 2026-10-17 审查发现）后的遗留项
>
> 范围：开发者工具 `scripts/quality-eval/`，不影响产品 · 严重程度：低 · 状态：未修复 · 置信度：已确认（代码核对；复现按代码推断）

## 问题

`parseSheet` 现在会校验 `date` 和 `id`，但 `id` 只在它"看起来像"现代 arXiv 编号时才校验：`ARXIV_ID_SHAPE_RE = /^\d{4}\./`（`scripts/quality-eval/sheet.ts:37`），只有匹配这个形状的值才会调用 `isModernArxivId`（`sheet.ts:79-81`）。因此：

- `2610.10230` 被表格软件存成 `2610.1023`：会被拒绝，这是最初复现的典型改写。
- 同一个单元格被存成科学计数法 `2.6101023E+8`、丢掉小数点变成 `261010230`，或变成任意其它字符串：**直接通过校验**。报告和重跑会使用改坏后的编号。
- 旧式编号（如 `hep-th/9901001`）也会直接通过；生产数据里是否会出现未核实。
- 编号错了以后，报告里的论文无法和 `labels.meta.json`、原始筛选记录对应；重跑请求也会把错误编号发给模型。同一请求内前后一致，指标本身可能不受影响，但报告不再能作为可追溯的记录。

日期没有这个问题：`isCalendarDate` 对每一行都执行，格式和真实日历日都会校验。

## 原因

- `sheet.ts:37-38` 定义了两个正则：形状正则 `ARXIV_ID_SHAPE_RE` 和现代编号正则 `MODERN_ARXIV_ID_RE`。
- `sheet.ts:79` 的 `else if (ARXIV_ID_SHAPE_RE.test(id) && !isModernArxivId(id))` 先要求形状匹配，形状不匹配就跳过校验。
- 这条形状守卫不是为了数据正确性，而是为了让 `scripts/tests/quality-eval-evaluate.test.mjs:44` 和 `scripts/tests/quality-eval-rerun.test.mjs:46` 里的 `s1`/`k1` 占位 id 继续可用。PR #96 期间这两个测试文件由另一个并行分支负责，无法同时改成真实编号，所以采用了折中。
- 导出表的 id 来自筛选记录里的 `paper.id`（`packages/core/src/pipeline/pipeline.ts:680`），由 arXiv 抓取器生成，正常应全部是现代编号。也就是说，占位 id 只存在于测试，生产数据可以严格校验。

## 复现

按代码推断（未实际运行）：

1. 取一份导出的 `labels.csv`，把某行的 `id` 改成 `261010230` 或 `2.6101023E+8`。
2. 运行 `node scripts/quality-eval/evaluate.mjs --recorded --sheet labels.csv --out /tmp/qe`。
3. 命令不报错；`report.md` 按改后的值显示编号，重跑请求里也是这个值。

`node --test scripts/tests/quality-eval-input-validation.test.mjs` 目前只覆盖了 `2610.1023` 这种"形状仍然匹配、序号位数不对"的改写，没有覆盖形状被整体改掉的情况。

## 可能的解决方案

1. **推荐：所有非空 id 都严格校验。** 去掉 `ARXIV_ID_SHAPE_RE`，对每个非空 id 直接调用 `isModernArxivId`（`sheet.ts:48-54`）。同步把两个主测试文件的 `s1..s7`、`k1..k4` 换成合法编号，例如选中层用 `2610.00001` 起、跳过层用 `2610.00011` 起；检查所有断言里出现的 id 字符串（`quality-eval-rerun.test.mjs:163` 等）。如果确认筛选记录里可能出现旧式编号，则扩展校验器接受旧式 `archive` 形式，或明确拒绝并在错误里说明。
2. **与导出元数据核对。** 在 `labels.meta.json` 里保存导出行的 `(date, id)` 列表，评估时逐行核对，能发现任何改写；代价是元数据变大，且要改导出格式。
3. **导出时把 id 写成非数字形式。** 例如 `arXiv:2610.10230`，解析时去掉前缀。能避免表格软件按数字处理，但仍需校验其它改写，且改变标注表的格式。

## 修复后应补的测试

- `scripts/tests/quality-eval-input-validation.test.mjs`：
  - `id` 为 `261010230`、`2.6101023E+8`、`abc` 时抛出 `SheetError` 并指出行号；
  - 合法现代编号通过：2015 年之前用 4 位序号（如 `1412.1234`），2015-01 起用 5 位序号（如 `2610.10230`）。
- 两个主测试文件的 fixture 换成合法编号后，原有断言（请求内容、分歧列表、指标数值）全部通过。
- 如果支持旧式编号，补一条 `hep-th/9901001` 的通过用例；如果不支持，补一条明确拒绝的用例。

## 相关

- 已修复部分见 `docs/issue/archive/2026-10-17-quality-eval-minor-issues.md` 第 2 节；日期校验和 `2610.1023` 这类序号位数改写在 PR #96 中已完成。
- 归档目录：`docs/issue/archive/`（2026-10-17 批次）。

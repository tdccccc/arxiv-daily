# 重跑时给原本只看标题判断的论文也带上了摘要，与记录的判定比较时混入了输入变化

> 发现日期：2026-10-17 · 来源：main `e6eaed5`（#93–#95 合并后）代码审查，质量评估脚本部分由子代理审查、主会话核对
>
> 范围：开发者工具 `scripts/quality-eval/`，不影响产品 · 严重程度：中低 · 状态：未修复 · 置信度：已确认（代码核对；现有测试固定了这一行为）

## 问题

日报筛选时拿不到摘要的论文，模型只看标题判断：筛选记录里 `hadAbstract=false`，标注表里 `had_abstract_at_filter=false`。导出时会为这些论文补取摘要供人标注，而重跑把标注表里的摘要原样发给模型（`scripts/quality-eval/rerun.ts:177`）。结果：

- **对比表混入输入变化。** "记录的判定与重跑对比"表（`report.ts:210-216`）里，这类论文的记录判定只看了标题，重跑看了摘要。两者的差别有一部分来自输入不同，而不全是提示词或模型的作用，报告没有提醒。
- **两边"排除无摘要行"的口径不同。** 重跑把 `hadAbstractAtFilter` 改成"标注表里有摘要"（`rerun.ts:288`），所以重跑小节的第二行（"排除表中没有摘要的行"，`report.ts:254`）和记录的判定的第二行（"排除筛选时无摘要的行"，`report.ts:173`）不是同一批论文，不能直接对照。两行的名称不同，说明这是有意的划分；问题在于报告没有提醒读者两者不可直接对照。
- **同一列名在两节含义不同。** 分歧列表的"筛选时有摘要"列（`report.ts:28-30`）在记录的判定小节里指日报筛选时有没有摘要，在重跑小节里指重跑请求里有没有摘要，也就是标注表里有没有摘要。对重跑本身而言这个值没错，但两节并排阅读时容易误解。

两次重跑之间的比较不受影响，例如 README 推荐的"改动前基线重跑 vs 改动后重跑"（`README.md:140-142`），因为两次都带摘要。

## 原因

- `rerun.ts:177`：`labelled.map((row) => ({ id: row.id, title: row.title, authors: "", abstract: row.abstract }))`，不看 `row.hadAbstractAtFilter`。
- `rerun.ts:288`：`hadAbstractAtFilter: row.abstract.trim() !== ""`，覆盖了标注表里的原值。
- 现有测试固定了这一行为：`scripts/tests/quality-eval-rerun.test.mjs:163` 断言每个已标注行都带着 `Abstract of <id>` 发出，而样例表里的 `s3`、`s5`、`k3`（第 49、51、56 行）是 `had_abstract_at_filter=false`。

## 复现

运行 `node --test scripts/tests/quality-eval-rerun.test.mjs`。用例 "the request is built by core from the sheet's direction snapshot and the labelled rows only" 通过，说明 `had_abstract_at_filter=false` 的 `s3`、`s5`、`k3` 在请求里都带着摘要；生成的 `report.md` 里没有相关提示。

## 可能的解决方案

1. **推荐：继续发送摘要，但让对比可以解释。** #95 之后，日报会先从 arxiv.org/abs 页补摘要，只看标题的情况会变少，带摘要更接近改动后的线上输入。具体做法：
   - 不再覆盖 `hadAbstractAtFilter`（删除 `rerun.ts:288` 的重新赋值），另加字段表示"本次请求是否带摘要"，分歧列表按原值显示"筛选时有摘要"；
   - 对比表增加一组"只看筛选时有摘要的行"的对比，让记录的判定与重跑在同一批、输入一致的论文上比较；
   - 已标注行里只要有 `had_abstract_at_filter=false`，就在报告开头提示：有多少篇当时只看了标题、重跑时带了摘要，与记录的判定比较时这部分差异来自输入变化。
2. 默认复现当时的输入：`hadAbstractAtFilter=false` 的行发送空摘要；提供开关，例如 `--with-abstracts`，改为发送标注表里的摘要。这样与记录的判定严格可比，但不再反映补摘要后的线上情况；需要同步修改 `quality-eval-rerun.test.mjs:163` 的断言。
3. 只在报告里加提示，不改统计。改动最小，但读者拿不到输入一致的对比。

## 修复后应补的测试

- 方案 1：存在 `had_abstract_at_filter=false` 的行时，重跑报告开头出现提示；"只看筛选时有摘要的行"的对比在记录的判定和重跑两侧行数相同；重跑小节分歧列表的"筛选时有摘要"列取原值。
- 方案 2：默认请求里这些行的摘要为空，加开关后带摘要。

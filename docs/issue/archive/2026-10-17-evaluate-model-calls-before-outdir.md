# 评估脚本先调用付费模型、后创建输出目录，目录不可用时结果全部丢失；已有报告会被直接覆盖

> 发现日期：2026-10-17 · 来源：main `e6eaed5`（#93–#95 合并后）代码审查，质量评估脚本部分由子代理审查、主会话核对
>
> 范围：开发者工具 `scripts/quality-eval/`，不影响产品 · 严重程度：低 · 状态：未修复 · 置信度：已确认（代码顺序明确；子代理用"输出目录位于普通文件之下"的场景复现过）

## 问题

- **钱可能白花。** `runEvaluate`（`scripts/quality-eval/evaluate.ts:109-133`）在重跑模式下先执行 `runRerun`（`evaluate.ts:122`），模型调用都在 `rerun.ts:207-210` 的循环里完成；之后才 `mkdir(outDir)`（`evaluate.ts:128`）并写 `report.md`、`report.json`（`evaluate.ts:129-131`）。输出目录建不出来或不可写时，异常到最后才抛出，已经完成的模型调用结果只在内存里，随异常一起丢失。`--runs` 最多 20 次（`evaluate.ts:16`），损失可能不小。
- **报告被静默覆盖。** 写报告时直接覆盖已有的 `report.md`、`report.json`。导出标注表时，`labels.csv` 已存在会要求 `--force`（`export.ts:196-198`），评估没有同样的保护。README 建议把改动前后的报告放在不同目录再比较（`README.md:140-142`），一旦误用同一个 `--out`，上一份报告就没了。

## 原因

- `evaluate.ts:122` 先跑模型，`evaluate.ts:128` 才建目录，中间没有任何对输出目录的检查。
- `evaluate.ts:130-131` 无条件 `writeFile`。
- `EvaluateOptions`（`evaluate.ts:18-33`）和 `parseEvaluateArgs`（`evaluate.ts:70-107`）都没有 `--force`。

## 复现

- 子代理的复现：先 `touch /tmp/qe-file`，再以 `--out /tmp/qe-file/report --runs 3 --yes` 运行重跑；3 次模型调用都完成后，`mkdir` 报 `ENOTDIR`，没有任何结果写出。
- 零成本复现：在测试里通过 `EvaluateDeps.llm` 传入假模型（`quality-eval-rerun.test.mjs` 已有 `fakeLlm`），把 `outDir` 设为普通文件下的路径，可以看到假模型被调用之后才报错。
- 覆盖：用同一个 `--out` 连续运行两次，第二次直接替换第一次的报告，没有任何提示。

## 可能的解决方案

1. **推荐：调用模型之前就准备好输出目录，并保护已有报告。** 在 `runEvaluate` 读完标注表之后、调用 `runRerun` 之前，执行 `mkdir(outDir, { recursive: true })`，再写入并删除一个临时文件确认可写；`report.md` 或 `report.json` 已存在且没有 `--force` 时直接报错（做法与 `export.ts:196-198` 一致），并在 `parseEvaluateArgs` 增加 `--force`。注意现有用例（`quality-eval-rerun.test.mjs:140-145`）断言不带 `--yes` 的试运行不写任何东西：如果希望试运行也提前发现目录问题，应只检查不创建（例如确认最近的已存在上级是可写目录），或相应调整该断言。
2. 每次运行结束立刻把该次的原始回答和判定写盘（例如 `run-1.json`），最终报告由这些文件生成。即使后面的步骤失败，已付费的结果也不会丢；多次运行时尤其有用。可与方案 1 同时采用。
3. 写报告失败时，把报告 JSON 写到系统临时目录并打印路径。能挽回结果，但属于事后补救。

## 修复后应补的测试

- `outDir` 位于普通文件之下时，断言报出清晰错误，且假模型被调用 0 次。
- `report.md` 已存在、未加 `--force` 时拒绝运行（同样断言假模型被调用 0 次）；加 `--force` 后正常覆盖。

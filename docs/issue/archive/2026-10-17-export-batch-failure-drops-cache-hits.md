# 导出摘要时一批请求失败，会连同这批里已命中缓存的摘要一起丢掉

> 发现日期：2026-10-17 · 来源：main `e6eaed5`（#93–#95 合并后）代码审查，质量评估脚本部分由子代理审查、主会话核对
>
> 范围：开发者工具 `scripts/quality-eval/`，不影响产品 · 严重程度：低 · 状态：未修复 · 置信度：已确认（代码核对；子代理复现：6 篇已缓存的论文全部被改从 abs 页重抓，表中摘要被替换）

## 问题

`fetchAbstracts`（`scripts/quality-eval/export.ts:289-345`）每批 100 篇（`export.ts:30` 的 `ABSTRACT_BATCH`）调用 core 的 `ArxivFetcher.fetchMetadataByIds`（`export.ts:324-337`）。只要这一批里有一篇需要联网且请求失败（例如遇到 429），整批调用就抛错，`catch`（`export.ts:333-336`）丢掉这一批的全部结果，包括其中已经从元数据缓存 `abstract-cache/` 命中的摘要。这些论文随后被当作缺摘要，改从 abs 页逐篇抓取（`export.ts:339-343`），每篇约 3 秒。

后果：

- 已缓存过的论文白白多发请求，导出变慢。
- abs 页提取的摘要文本（`extractAbsPageAbstract`，保留 TeX）可能与之前从接口取到的不同，同一篇论文在新旧标注表里的摘要会变。
- `export.ts:335` 的警告 "fetched ones stay cached, so a later run with --force picks up where this one stopped" 容易让人以为本次已缓存的部分会被用上，实际在本次运行里被丢掉了。README 第 59 行"取回的摘要有缓存，中断后重跑不会重复请求"在这种情况下也不成立。

与 `2026-10-17-export-docs-claim-429-retry.md` 叠加时影响更大：第一批遇到 429 后，后面每一批都会立即失败，所有批次里的缓存命中都被丢掉。

## 原因

- core 的 `fetchMetadataByIds`（`packages/core/src/pipeline/arxiv-fetcher.ts:357-419`）先逐个读缓存，命中的放进局部结果 `out`（第 368-375 行），未命中的交给批量请求；随后逐个等待未命中的请求（第 412-417 行），任何一个失败都会让整个函数抛错，`out` 里已命中的结果不会返回。这个接口是"全有或全无"。
- 导出脚本的预读只覆盖 abs 页缓存（`export.ts:316-322`），不覆盖元数据缓存，所以从接口缓存过的论文每次都要经过 `fetchMetadataByIds`。

相关情况（不在本 issue 范围，影响未核实）：文献库扫描的元数据解析也一次性调用同一接口（`packages/core/src/library/arxiv-library-metadata-resolver.ts:14`），遇到 429 时同样会丢掉其中已命中缓存的结果。

## 复现

- 子代理的复现：先用较小样本导出一次，摘要进入 `abstract-cache/`；再用包含这些论文的更大样本导出到同一个 `--out`（加 `--force`），并让接口返回 429。6 篇已缓存的论文全部改从 abs 页重抓，表中摘要被 abs 页的文本替换。
- 用 `quality-eval-export.test.mjs` 的 `fakeHttp` 构造：第一次导出正常；第二次导出时让第 1 次 API 调用返回 429，断言 `http.absCalls` 里出现了第一次已缓存的论文。

## 可能的解决方案

1. **推荐：导出脚本先单独读元数据缓存。** 仿照 `export.ts:316-322` 对 abs 页缓存的预读，用同一个 `AtomMetadataCache`（`export.ts:302-306`）逐个 `get`，命中的直接取摘要，只把未命中的交给 `fetchMetadataByIds`。改动局限在开发者工具里，不改变 core 接口的语义。同时把 `export.ts:335` 的警告改准确，例如"已取回的摘要已缓存，下次运行会复用"，并修正 README 第 59 行。
2. 让 core 提供"部分成功"的批量接口，例如 `fetchMetadataByIdsSettled(ids)`，返回 `{ found, failedIds, error }`，导出脚本和文献库元数据解析都能复用。好处是一并处理上面的"相关情况"；代价是改动 core 公共接口，需要补 core 测试并评估其他调用方。
3. 缩小批次（例如每批 10 篇），减少一次失败波及的缓存命中。只能缓解，不推荐。

## 修复后应补的测试

- `scripts/tests/quality-eval-export.test.mjs`：先导出一次使摘要进入缓存；第二次导出时让第 1 次 API 调用返回 429，断言已缓存的论文不出现在 `http.absCalls` 里，表中摘要与第一次相同。
- 采用方案 2 时：在 `packages/core/tests/arxiv-fetcher.test.ts` 断言批量请求失败时，缓存命中仍在返回结果里，失败的 id 单独列出。

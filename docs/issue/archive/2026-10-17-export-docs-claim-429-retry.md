# 导出脚本的说明和注释仍写"限流时等待重试"，与 #95 之后的实际行为不符

> 发现日期：2026-10-17 · 来源：main `e6eaed5`（#93–#95 合并后）代码审查，质量评估脚本部分由子代理审查、主会话核对
>
> 范围：开发者工具 `scripts/quality-eval/`，不影响产品 · 严重程度：低 · 状态：未修复 · 置信度：已确认（代码核对；子代理用 255 篇论文复现）

## 问题

下面三处仍按旧行为描述：查询接口限流时会等待重试，失败只影响一批。

- `scripts/quality-eval/README.md:58`："arXiv 的接口有时会限流；限流时脚本会等待并重试，仍失败的论文改从 `arxiv.org/abs/<编号>` 页面读取摘要……"
- `scripts/quality-eval/export.ts:224` 的进度提示："fetching abstracts for N papers from arXiv (waits and retries while arXiv rate-limits)"。
- `scripts/quality-eval/export.ts:281-288` 中 `fetchAbstracts` 的注释："it batches `id_list` queries, waits out HTTP 429 with backoff … A failed batch costs only that batch's abstracts"。

#95（"Stop retrying any query API 429"）之后的实际行为：

- 查询接口 `export.arxiv.org` 返回 429 后不再重试，见 `packages/core/src/pipeline/arxiv-fetcher.ts:841` 的 `shouldRetryArxivRequest`。
- 同一进程内，该主机进入"立即失败"冷却：有 `Retry-After` 时按它，没有时 10 分钟（`arxiv-fetcher.ts:53` 的 `QUERY_API_HEADERLESS_429_COOLDOWN_MS`，由 `arxiv-fetcher.ts:231` 起的 `recordCooldown` 设置）。冷却期内的后续请求不会发出，直接抛 `ArxivRetryDeferredError`（`arxiv-fetcher.ts:207-217` 的 `waitUntilEligible`）。

所以只要第一批遇到 429，后面所有批次都会立即失败，剩下的论文全部改从 abs 页逐篇读取，每篇约 3 秒。"失败只影响一批"不成立，耗时也远超说明给人的印象。`export.ts:333-336` 的 `catch` 不区分错误类型，回退本身能正常工作；问题在于说明和提示会误导，而且每个剩余批次都会各打一条 "abstract fetch failed" 警告。

## 原因

#95 改了 core 抓取器的重试和冷却策略。导出测试已经按新行为更新（`scripts/tests/quality-eval-export.test.mjs:319-330`，"a rate-limited query API request is not retried"），但 README、进度提示和注释没有同步。

## 复现

- 子代理的复现：样本 255 篇（每批 100 篇，见 `export.ts:30` 的 `ABSTRACT_BATCH`，共 3 批），查询接口对第一个请求返回不带 `Retry-After` 的 429。结果只发出 1 次 API 请求，第 2、3 批立即以 `ArxivRetryDeferredError` 失败，255 篇全部改走 abs 页。
- 用测试辅助函数构造：`quality-eval-export.test.mjs` 的 `fakeHttp({ respond })`（第 89-109 行）让第 1 次 API 调用返回 `{ status: 429, headers: {}, bodyText: "slow down" }`，配合 `instantCoordinator()`（第 112-118 行），样本超过 100 篇；断言 `http.calls.length === 1`，且 `http.absCalls` 覆盖全部论文。现有用例用的是 `retry-after: "0"`（冷却为 0），样本也只有一批，覆盖不到这种情况。

## 可能的解决方案

1. **推荐：同步说明、提示和注释。** README 第 58 行改为：查询接口一旦限流就不再重试，本次导出剩余的论文都改从 abs 页读取，每篇约 3 秒；已取回的摘要会缓存，稍后重新运行（新进程不受上次冷却影响）可以继续使用接口。`export.ts:224` 的进度提示去掉 "waits and retries"；`export.ts:281-288` 的注释改为"429 不重试；冷却期内其余批次立即失败；统一回退到 abs 页"。
2. 在方案 1 之外改进行为：`fetchAbstracts` 遇到 429 或 `ArxivRetryDeferredError` 后停止调用接口（跳出批次循环），只发一条汇总警告，例如"arXiv 接口限流，剩余 N 篇改从 abs 页读取，预计约 M 分钟"，避免每批一条重复警告。
3. 在导出脚本里自己等待 `Retry-After` 后重试接口。能减少 abs 页请求，但与 #95"不对限流的查询接口重试"的策略相反，不推荐。

## 修复后应补的测试

- 多批次、不带 `Retry-After` 的 429：只发出 1 次 API 请求，全部论文由 abs 页补齐；采用方案 2 时还应断言只有一条汇总警告。
- 采用方案 2 时：出现 `ArxivRetryDeferredError` 后不再调用 `fetchMetadataByIds`。

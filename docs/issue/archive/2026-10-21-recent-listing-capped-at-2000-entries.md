# arXiv 列表页每页最多 2000 条，大分类（cs.LG、cs.AI、cs.CV、cs.CL 等）较早的日期被误判为“永久失败”，被截断的那一天会静默少论文

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 PIPE-02
> 范围：日报流水线的论文发现（插件、CLI、工作台的补跑与重试） · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-21-batch2`，提交 e0fbb2c） · 置信度：已复现

## 问题

每次运行对每个分类只请求一页 `https://arxiv.org/list/<分类>/recent?skip=0&show=2000`，注释里写的是“一次拿到全部 5 天”。但真实页面（`packages/core/tests/fixtures/arxiv-recent-astroph.html`）明确写着“Showing up to 2000 entries per page”，页面顶部每个日期的链接也带着 `skip=59`、`skip=144` 这样的偏移量，说明 5 天的内容超过 2000 条时要翻页。cs.LG、cs.AI、cs.CV、cs.CL 这类分类 5 个公告日的新投稿加交叉列表远超 2000 条，只有最新的一到两天落在第一页。

代码对“页面被截断”没有任何感知：

1. **补跑较早的日期会被判为“永久失败”。** 调度器每次 tick 会检查最近 5 个日历日（`LOOKBACK_DAYS`），应用关闭了两三天之后重新打开，要补的日期落在第一页之外。`listForDate` 找不到那一天的分组，且日期比页面里最老的分组还早，于是返回 `failed_permanent`，原因写着“arXiv no longer lists it there”（arXiv 其实还列着，只是在后面的页上）。永久失败的日期不会被自动重试，手动重试在 5 天窗口内也一样失败。只要多个分类里有一个是大分类，整天就被拒绝（“partial discovery”规则）。
2. **落在第 2000 条附近的那一天只会拿到前半截。** 该日分组仍然存在，只是少了后面的条目；`parseRecent` 不看标题里的“(showing N of M entries)”，`listForDate` 把它当成完整的一天，过滤、摘要、日报照常生成，没有任何提示，缺的论文再也不会出现（日报已存在就不再重跑）。
3. 周一的公告日包含周末积压，条目最多；最大的分类在这一天的新投稿加交叉列表可能单日就接近 2000 条，同样会被静默截断（随 arXiv 投稿量增长越来越近）。

## 原因

- `packages/core/src/pipeline/arxiv-fetcher.ts:329-335`：只取 `skip=0&show=2000` 一页，没有按总条数翻页，也没有用页面上每个日期自带的 `skip` 链接。
- `packages/core/src/pipeline/arxiv-parser.ts:74-95`：忽略 `<h3>` 里的“showing N of M entries”和页首的“Total of N entries”，无法发现被截断的分组。
- `packages/core/src/sources/arxiv-source-adapter.ts:108-125`：缺少某天分组时，只按“比最老分组更早”就判定永久失败；`:131-133` 把被截断的分组当成完整数据。

## 复现

用测试里的 `ArxivSourceAdapter` 加假的 `fetchRecent`，页面照真实页面的结构构造（一个 `Total of 6200 entries` 的分页提示，最新一天 3 篇，第二天标成 `showing first 2 of 1300 entries` 只给 2 篇），结果（一次性测试，已删除）：

- 请求第三天（页面之外）：`{"kind":"error","failureKind":"failed_permanent","reason":"date 2026-05-07 is older than the oldest cs.LG /recent bucket 2026-05-08; arXiv no longer lists it there ..."}`；
- 请求被截断的那一天：`ok`，2 篇，`outcome: "available"`，没有任何告警。

页面构造依据真实页面的标记；“arXiv 只给前 2000 条”这一前提来自真实页面上的“Showing up to 2000 entries per page”文字，没有联网验证。

## 可能的解决方案

推荐：

1. 解析页首的“Total of N entries”和每个日期链接的 `skip` 偏移，对目标日期单独请求 `skip=<偏移>&show=<该日条数>`；或按 2000 一页顺序翻页直到取完。
2. 解析每个分组标题里的“(showing N of M entries)”，拿到的条目数小于 M 时视为不完整：补页，补不到就返回暂时失败而不是照常出报告。
3. “日期比最老分组更早”要在确认已翻完所有页之后才判为永久失败。

备选：在日志和运行结果里明确提示“该分类单日条数过多，可能被截断”，并在设置页提醒大分类用户。

## 修复后应补的测试

- 总条数超过 2000 的列表页：目标日期在第一页之外时能取到完整分组。
- 分组被截断（显示数小于总数）时不会静默继续。
- 只有在所有页都翻完仍找不到日期时才返回永久失败。

## 修复情况

日报现在能读到 arXiv 列表第一页（最多 2000 条）之后的日期。解析器读出页首的"Total of N entries"、每个日期链接的 `skip` 偏移和每个日期标题里的"of M entries"；目标日期不在第一页，或被页尾截断时，从该日期自己的偏移起翻页读取，每页最多 2000 条，沿用原有的请求间隔（至少 3 秒）和 429 冷却。读到的条数必须与该日条数完全对上才使用，否则按暂时失败稍后重试，原因里写出"900 of 1300"这样的数字，不会再带着少一半的论文出日报；"日期早于最老日期，永久失败"也只在日期链接证明它确实不再被列出时才成立（没有日期链接的长列表按暂时失败处理）。代价：大分类补跑较早的日期时多读 1 到 2 个页面，每页相隔 3 秒，页面也较大。此前的评审（`docs/reviews/2026-09-24-main-review.md`）实测 cs.LG 五个公告日共 1198 条，常规使用下尚未碰到 2000 条的上限；这次修复是让它碰到时不再出错。

测试：`arxiv-parser.test.ts`、`arxiv-fetcher.test.ts`、`sources/arxiv-source-adapter.test.ts`（含用真实 `ArxivFetcher` 加假 HTTP 层检查请求地址、3 秒间隔和 429 冷却的用例）、`services/recent-dates.test.ts`；测试用的分页列表由 `tests/recent-listing-world.ts` 按真实页面的结构生成。

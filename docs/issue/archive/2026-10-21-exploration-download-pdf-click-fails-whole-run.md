# Obsidian 探索里模型点一次“Download PDF”，当前任务和整次运行都会被判为失败（夹具没有 PDF 路由）

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 SCR-09
> 范围：`scripts/acceptance/fixtures.mjs`、`exploration-obsidian.mjs` 的受限控件策略、`run.mjs` 的夹具契约检查（Obsidian 探索宿主） · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-21-batch4`，提交 d16121f） · 置信度：已复现

## 问题

阅读面板每一行论文的操作列里有三个相邻的图标按钮：“Open arXiv”“Open PDF”“Download PDF”。探索的受限策略只拦住了第一个（`Open arXiv`，“link leaves the plugin for an external destination”）；`Open PDF` 的问题已在 EXP-04 里记录。第三个 `Download PDF` 既没有被限制，夹具服务器也没有对应的 PDF 路由，于是模型点一次就会同时造成两个错误的“失败”判定：

1. **当前任务被判失败。** 点击调用 `plugin.downloadPdf()` → `PdfService.downloadById()` → `fetcher.fetchPdf()`，请求 `https://arxiv.org/pdf/<id>`（`pdfUrl: https://arxiv.org/pdf/${canonicalId}`），经夹具代理到 `fixtures.mjs`。夹具只服务 `/list/<分类>/recent`、`/api/query`、`/html/<id>`、`/abs/<id>`，对 `/pdf/` 返回 502，并把这条请求记为 `kind: "unexpected"`。插件把这次失败写成 `logger.error("pdf: fetch failed …")`，也就是渲染进程里的 `console.error`；探索的观察器把新出现的、未被允许的控制台错误放进 `observation.errors`，循环一看到 `errors` 就把当前任务（包括角色任务）直接判为 `failed`，错误信息是插件日志，看起来像产品缺陷。
2. **整次运行被判失败。** `run.mjs` 在每个套件结束后检查夹具收到的请求，只要有 `unexpected` 或 `fixture-error` 就把套件状态改成 `failed`，追加一个 `exploration.fixture-contract` 场景，错误文字是“Unexpected or malformed external requests reached the fixture; inspect request evidence”。

这两个判定都是错的：在真实环境里这个按钮会从 arXiv 正常下载 PDF，产品没有问题，只是测试夹具没有实现这条路由。2026-10-21 当天的真实探索里，`first-time-user` 角色在同一排按钮里点了 “Open PDF”（步骤 14），再点一下旁边的 “Download PDF” 完全在这个角色的探索范围内；固定验收套件不受影响（它不点这个按钮）。

复现（夹具与总判定这一半，用真实的 `runAcceptance` 和真实夹具服务器；一个全部通过的假探索运行器，只是向夹具发了一次 `GET https://arxiv.org/pdf/2610.10001`）：夹具回 HTTP 502，报告里套件状态 `failed`、退出码 1，场景列表是 `library-search=passed, exploration.fixture-contract=failed`。

## 原因

- `scripts/acceptance/fixtures.mjs:117-152`：夹具对 `arxiv.org` 只处理 `recent`、`/api/query`、`/html/…`、`/abs/…`，其余一律 502 且记为 `unexpected`；没有 `/pdf/<id>`。
- `scripts/acceptance/exploration-obsidian.mjs:513-560`（`classifyControlRestriction`，外部链接规则在第 547 行）：外部链接那一条按固定名字匹配（`open arxiv`、`report a bug`、`documentation` …），没有覆盖 `Open PDF`、`Download PDF`，也没有其他“会发起夹具没实现的请求”的控件的概念。
- `scripts/acceptance/run.mjs:224-229`：对夹具收到的任何 `unexpected` 请求无条件把整个套件判为失败，不区分“产品违约”和“夹具没实现产品的合法请求”。
- `scripts/acceptance/exploration-obsidian.mjs:896-915`（`observe()`，错误来自 `classifyObsidianDiagnostics`）：把插件的 `console.error` 当作产品错误，立即让当前任务失败。

## 可能的解决方案

推荐：在夹具里补上 `https://arxiv.org/pdf/<id>` 的响应（返回一个由 `buildFixturePdf` 生成的小 PDF，`content-type: application/pdf`），让这个按钮在探索里有一个可验证的、确定的结果（PDF 被写进 Vault 的 `arxiv-daily/pdfs/`）；`/pdf/` 路由只对夹具内已有的论文编号放行，其余仍然 502。这样 `Download PDF` 不需要被限制，还能新增一个目标任务去验证它。

补强：加一个“控件对照”测试——枚举仪表盘行操作列、相似论文弹窗、文献库检索块里所有会触发网络或外部程序的按钮（`view.ts:1581-1591`、`similar-papers-modal.ts:206-207`、`library-search-block.ts:85`），逐个断言它要么在受限策略里，要么夹具有对应路由。以后新增一个会发请求的按钮时，测试先红。

备选：把 `Download PDF` 加进受限名单。一行改动，能避免误判，但这个按钮从此再也得不到探索覆盖。

## 修复后应补的测试

- `scripts/tests/acceptance-fixtures.test.mjs`：对夹具内已有论文编号请求 `/pdf/<id>` 得到 200 和 PDF 字节，对未知编号仍得到 502 并记为 `unexpected`。
- `scripts/tests/acceptance-exploration-obsidian.test.mjs`：`Download PDF` 在策略里的处理与夹具一致（要么受限，要么有路由）；上面的“控件对照”测试。
- `scripts/tests/acceptance-runner.test.mjs`：夹具契约失败时，报告里的错误文字能指出是哪个 URL（现在只说“inspect request evidence”），避免把夹具缺口当成产品缺陷。

## 修复情况

夹具为已登记论文（含版本号和动态加入的论文）提供可解析 PDF，未知编号继续返回 502 并记录 unexpected。合法 Download PDF 不再因夹具缺路由误判整次失败；契约错误文字直接列出请求 URL，方便定位。2 项失败回归修复后 fixture/runner 30 项通过。下载按钮不受限，外部打开另由 EXP-04 的记录桩处理。

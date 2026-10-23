# Obsidian 探索看不到插件之外发生的事：打开笔记后仍当作在仪表盘，“Open PDF”既无记录也没被限制

> 发现日期：2026-10-21 · 来源：模型探索（gpt-6.1-sol，high）+ 人工复核 · 编号 EXP-04
> 范围：`scripts/acceptance/exploration-obsidian.mjs`（观察与限制策略）、Obsidian 探索的 first-time-user 角色 · 严重程度：中 · 状态：未修复 · 置信度：已复现（第 3 点“真的拉起了外部程序”为代码推断）

## 问题

Obsidian 探索里，模型只能通过 `obsidianObservation` 看到插件自己的界面。产品做的两类事发生在这个界面之外，观察器要么描述错，要么完全看不到：

1. **打开日报/论文笔记后，观察仍然写“在仪表盘”。** 用户在仪表盘点日历里的日报或论文行的“Open detail summary”，Obsidian 在新标签页里打开 Markdown 笔记（产品行为正确，截图里笔记渲染正常）。但观察器仍报 `view: "dashboard"`、`rootMounted: true`，正文是后台标签页里那个已隐藏的仪表盘的文字，只有窗口标题变了（“2026-10-06 - vault”），控件为空。模型把这当成产品缺陷：“打开日报后没有可用的阅读或返回入口”、“打开论文后仍显示仪表盘文本”（2026-10-21 探索的 O1、O2）。
2. **点“Open PDF”（论文没有本地 PDF 时）什么都不记录。** 插件调用 `window.activeWindow.open(<arxiv.org PDF 地址>)`，由 Obsidian 主进程交给系统打开，应用内没有任何变化。模型报告“没有可见反馈，无法确认是否成功打开”（O5）。
3. **同一个按钮没有被限制。** 同一个函数打开 arXiv 页面的“Open arXiv”被探索策略限制（“link leaves the plugin for an external destination”），“Open PDF” 却是放行的。放行意味着一次未脚本化的点击可能让 Obsidian 主进程调用系统的“打开外部链接”，拉起真实的浏览器去访问真实的 arxiv.org。探索的网络保护只给 Obsidian 自己的 Chromium 进程加了不可达的代理（`scripts/desktop-acceptance/launch.mjs:38-43`），管不到被拉起的外部程序，这违背探索“外部请求全部走固定响应”的约束。这一点只从代码和策略表推断，没有在进程层面亲眼确认浏览器被拉起。

这对探索结果的影响：first-time-user 每次都会被要求“打开一篇论文读一读”，所以每次运行都会在这里踩到。2026-10-21 的 Obsidian 全量运行里，该角色 17 步中有 8 步（步骤 79-86）围绕“打开笔记、发现没控件、`home` 回去、再试”，并产生了 2 条虚假的 needs-review；另一次运行（personas2）又重复了一遍。2026-10-12 的发布探索里同类“打开普通笔记后走进死胡同”已被记录，当时只补了 `home` 动作，描述错误的部分没有处理。

## 证据

- `output/playwright/acceptance/2026-10-21-review-obsidian/exploration/`（本地，不入库）：
  - `exploration-events.jsonl` 步骤 79-86：点击日历/论文行后的观察 `title` 变成笔记名，`layout.view` 仍是 `dashboard`，`controls` 为 0，模型连发 3 次 `home`，步骤 82、85 记 finding。
  - `first-time-user-step-82.png`、`first-time-user-step-85.png`：实际界面是新标签页里渲染正常的日报和论文笔记。
- `output/playwright/acceptance/2026-10-21-review-obsidian-personas2/exploration/`：`exploration-events.jsonl` 步骤 8、17 同样 `home`；步骤 14-15 点“Open PDF”，观察里 notices 为空，标题和标签页都没变（`first-time-user-step-15.png`），模型记 finding。同一观察（步骤 24 的控件表）里 “Open arXiv” 标记为 RESTRICTED，“Open PDF” 没有。
- 限制策略确定性复现：对真实的 `classifyControlRestriction` 传入 `Open arXiv`、`Open PDF`、`Download PDF`，结果分别是“受限”“放行”“放行”（无网络，纯函数）。
- 产品侧对照：`open-today-report` 目标任务在同一次运行里通过，说明“打开日报”本身没问题。

## 原因

- `scripts/acceptance/exploration-obsidian.mjs:226-254`（`resolveSurface`）：最后一步 `document.querySelector(".arxiv-daily-dashboard")` 不检查它所在的标签页是否仍是活动页，后台标签页的 DOM 会一直保留，于是把隐藏的仪表盘当成当前界面。`:310-335` 在控件为 0 时追加的 `recoveryText` 只提醒“可能在普通笔记里”，但 `layout.view` 和正文依旧是仪表盘的。
- 观察器没有任何关于“当前活动标签页是什么”的字段，也读不到被打开笔记的文字。
- `plugin/src/dashboard/view.ts:2504-2510`：没有本地 PDF 时 `openPdf` 走 `openArxivResource(..., "pdf")`，经 `plugin/src/hosts/obsidian/resource-opener.ts:21-23` 调 `window.activeWindow.open(url, "_blank", "noopener")`。Obsidian 主进程（`resources/obsidian.asar` 的 `main.js`，只读查看）对 http(s) 先向渲染进程派发 `open-url`，没人处理就 `shell.openExternal`。
- `scripts/acceptance/exploration-obsidian.mjs` 的 `classifyControlRestriction` 里外部链接那条规则只匹配固定名字（`open arxiv`、`report a bug`、`documentation` 等），不含 “Open PDF”；仪表盘的相似论文弹窗（`similar-papers-modal.ts:207`）和文献库匹配行（`library-search-block.ts:85`，库外 PDF 会走 `file://`）里也有同名按钮。
- 夹具侧没有对 `window.open` 做任何桩：`grep` `scripts/desktop-acceptance`、`scripts/acceptance` 找不到对 `window.open`/`openExternal` 的处理。

## 可能的解决方案

推荐：把“产品在界面外做了什么”变成观察的一部分，同时挡住真实的外部打开。

1. 观察器改为只在仪表盘所在标签页处于活动状态时把它当作当前界面（例如检查 `.workspace-leaf.mod-active` 或元素有布局盒）；否则报 `view: "note"`（或 `outside`），给出活动标签页的类型、标题，并附一段该笔记预览区文字的只读摘录（截断），让模型能确认“日报真的打开了、内容是什么”。保留 `home`。
2. 在渲染进程里对 `window.open` 和 `activeWindow.open` 装一个记录桩（与现有 `installHttpProxyExpression` 同一思路）：不转发，只把 URL 追加到 `observation.layout.openedUrls`，并让观察显示“外部链接已被探索环境拦截：…”。这样 “Open PDF”“Open arXiv”都不会拉起真实程序，模型也能看到效果，之后可以统一放开这两个按钮的限制。

备选：只把 “Open PDF” 加进限制名单（一行改动，堵住隔离缺口），但模型仍然无法验证这个按钮，first-time-user 的两条虚假 finding 也不会消失。

## 修复后应补的测试

- `scripts/tests/acceptance-exploration-obsidian.test.mjs`：构造“仪表盘元素还在、但所在标签页已不活动，另有一个可见的 Markdown 视图”的 DOM，断言观察的 `layout.view` 不是 `dashboard` 且包含笔记文字；仪表盘活动时保持现有行为。
- 同文件：点击触发 `window.open(url)` 后，URL 出现在观察里，且没有调用到原始 `open`。
- 限制策略测试：`Open arXiv` 与 `Open PDF` 的处理一致（要么都限制，要么都放行并走桩）。

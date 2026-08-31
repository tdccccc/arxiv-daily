# P2 — 建索引的过程与结果落在 Library 行上

status: done
updated: 2026-08-31

## 问题

进度数据一直在流：`indexPersonalLibraryFullText` 全程往 `this.progress` 推，状态栏一秒能更新好几次。
但从设置页点 Build index 的人看不到其中任何一条 —— 设置页是盖在状态栏之上的模态框。
于是那个按钮按下去之后就没声了，一个大库要几小时。

取消同理：`operation.signal` 早就能中止，只是设置页没有入口。

跑完之后也没有痕迹：`setComplete` 的面板 4 秒后自动隐藏（`AUTO_HIDE_COMPLETE_MS`），
通知 10 秒后消失，状态栏只剩一句没有上下文的 `Complete`。
一个跑了一整夜的索引，第二天回来无从判断它到底跑完没有。

## 做了什么

三件事，都落在同一行上。

1. **进度**。说明变成当前阶段，主按钮变成不可点的 `Indexing… (N/M)`。
2. **取消**。索引期间第三个按钮是 Cancel；按下后读作 `Cancelling…` 并禁用。
3. **完成痕迹**（用户在本阶段确认要做）。空闲时说明末尾加一句
   `Last indexed 2026-08-31 05:15 · 128 papers searchable.`

## 关键决定

- **N/M 来自结构化回调，不解析字符串**。core 的 `onProgress` 原本只给一句话
  （`indexing <key> (5/120), ~2m remaining`）。给它加了可选的第二个参数
  `{ phase, completed, total }`，与原字符串同时发出 —— 状态栏继续用那句话，按钮用数字。
  从散文里正则抠数字会在文案改动时无声地坏掉。
- **完成痕迹取自 manifest，不取自 run summary**。summary 说的是"这次跑了什么"，
  manifest 说的是"现在能搜到什么"。重建、换文件夹、外部删除之后，只有后者不会撒谎。
  运行结束时用 `summary.manifestUpdatedAt` / `searchablePapers`（本次提交的那份 manifest，
  为此给 summary 加了这两个字段），启动和换文件夹时直接读 manifest —— 都不额外多读一次。
- **进度更新不重渲染整页**。每篇论文报一次，大库每秒好几次。
  开始和结束改变按钮集合，走 `refreshSettings()`（一次运行两次，滚动位置由既有机制保留）；
  中间只改文字，直接写进已渲染的按钮组件，250ms 限流。
  这样正在编辑的其它输入框不会被打断。
- **索引期间隐藏 Revoke，禁用 Change folder**。Revoke 让位是为了守住"最多三个按钮"
  （Change folder + Indexing… + Cancel 正好三个）；Change folder 禁用是因为运行中换文件夹
  会在下一次身份校验时把跑了几小时的任务打掉。
- **取消不算失败**。设置页的 catch 分支识别 `isCancellationError`，给一条
  "indexing cancelled. Nothing was saved, so the next build starts over." 的通知，
  而不是弹一个"索引失败"的错误 —— 那是用户自己按的按钮。文案说明白了：中途取消什么都不留
  （manifest 只在运行末尾提交一次）。

## 顺带修掉的布局缺陷

`Indexing… (12/40)` 是这一行出现过的最宽的按钮。在 448px 面板下按钮组的上限是 116px，
而这个标签要 134px —— 单个按钮宽于上限时，`flex: 0 0 auto` + `white-space: nowrap`
既不能收缩也不能折行，于是整个溢出到左边压在说明文字上。这正是那个上限本来要防的事，
只是从它没覆盖的口子里进来了。

改法：按钮改 `flex: 0 1 auto; min-width: 0; white-space: normal`。
hypothetical size 仍是 max-content，所以放得下的按钮照常按自然宽度排、
一行放不下两个按钮照常换行；flexbox 只在某个按钮独占一行且仍然过宽时才收缩它，
那时标签折成两行而不是被裁掉。在按钮本来就放得下的所有宽度上这两条都不生效 ——
既有几何断言的数字一个像素没变可以佐证。

这个缺陷是桌面验收当场量出来的，不是看出来的。

## 验证

- 单元（happy-dom 能定的）：`libraryRowPresentation` 的各状态、三按钮上限、
  计数缺失时不编造分数、store 的重放/去重/结束后忽略报告；
  core 的结构化计数与 manifest 字段；插件运行时的发布与取消。
- 桌面验收（只有真渲染器能定的）：新增 5 条 —— 行是否自己接住了运行、
  运行态几何、进度是否原地重写（按元素上的 mark 判定身份，重渲染会丢掉 mark）、
  Cancel 是否真的让某个 operation 进入 cancellationRequested、
  完成痕迹是否留在空闲行上。**先取红**：把 `libraryRowPresentation` 的运行分支
  和痕迹拼接临时去掉重跑，五条各自红在自己的原因上，措辞互不混淆。
- 验收里的索引运行是真的 operation（走插件自己的注册表，所以 Cancel 中止的是真信号），
  但不跑真的嵌入 —— 那需要模型和几分钟的 PDF 抽取，且不影响这一行长什么样。
  "真运行会报告到同一个 store"由插件单测断言。

## 没有做

不加 Search library 按钮。用户的判断是搜索时本来就看得到，完成提示够了。

# P1b — compact-library-settings

goal_ref: ../goal.md
updated: 2026-08-30

## Outcome

Personal library 设置默认只露出三行：选库/建索引、Embedding 下拉、Better PDF parser 开关。远程 URL/key 和 sidecar URL 按需展开。按钮右对齐，并在说明文字仍读得了的前提下排成一行——排不下时换行，每行仍右对齐（见 Verification 的最终结论）。

## Assumptions

- 本地默认不需要看到远程嵌入字段或 sidecar URL。
- declarative（1.13+）和 legacy display()（&lt;1.13）必须同一套分组与显隐。
- 索引进度与「去搜索」仍留给后面的 P2，本阶段不改通知或 Dashboard。

## Approach

把 Embedding 与 PDF parsing 并进 Personal library，放到 LLM 后面。`embedding.mode === "remote"` 才显示远程字段；sidecar 打开才显示 URL。切换 mode / sidecar 后刷新设置页。文献库按钮 `flex-wrap: nowrap`、右对齐。

## Test strategy

- change kind: behavior change（设置呈现）
- strategy: strict Red–Green–Refactor on definitions + declarative tab + CSS
- Red / baseline signal: 旧测试锁死独立 Embedding / PDF parsing heading，以及 local 模式下仍列出远程字段名
- Green / regression checks: settings-definitions、settings-declarative-tab、library-connection-lifecycle、settings-tab、plugin typecheck

## Tasks

- [x] 默认只渲染 Library / Embedding / Better PDF parser。
- [x] remote 与 sidecar 打开时展开对应字段。
- [x] legacy display() 与 declarative 同一顺序和显隐。
- [x] 文献库按钮右对齐；在不把说明文字压到可读下限以下的前提下排一行，排不下就换行（原文写的是「不换行」，见下方更正与最终结论）。

## Verification

- 默认定义不含 Embedding / PDF parsing heading，也不含远程 URL 字段名。
- remote + sidecar 打开后九行都在。
- CSS 锁死 `flex-wrap: nowrap` 与 `justify-content: flex-end`。
- Observed Green (2026-08-30): plugin 137 tests（settings-definitions / settings-declarative-tab / library-connection-lifecycle / settings-tab）与 typecheck 通过。
- **更正 (2026-08-30)**：上面那条「CSS 锁死 `flex-wrap: nowrap`」不是有效验收。它只在 `styles.css` 文本里找一段正则，证明的是「我们写了这条声明」，不是「按钮真的排成一行」——而这条声明当时是和 `min-width: 0` 一起加的，两条合起来让按钮整体溢出控件盒、压在描述文字上。也就是说：断言绿着，产品坏着，坏的正是这条断言声称守住的东西。真正的验收是桌面验收里的 `library-row-geometry` / `library-row-geometry-stacked`，在真实渲染进程里量按钮坐标；happy-dom 没有布局引擎，结构上给不出这个答案。同时 `nowrap` 本身也不该「锁死」：Obsidian 在 `@container (max-width: 340px)` 下会把每行拉成整列、每个按钮拉成整行宽，那里必须换行，否则按钮横铺出面板。修法见 journal 2026-08-30「Library 行按钮溢出，修布局」。

  “Abort / reshape triggers” 里那条「如果按钮在窄设置栏里被裁切到看不见主 CTA，改回允许换行」实际已经触发，并按它执行了——只是范围限定在 Obsidian 自己的堆叠断点之下，正常宽度仍是一行右对齐。

- **再次更正 (2026-08-30)**：上面这条更正本身也说窄了。本阶段的验收结论不是「按钮一行不换行」，正确的结论是 **「正常宽度一行右对齐，挤不下时换行，说明文字保持可读」**。

  上面那条只把中止条件的触发范围限定在堆叠断点之下，理由是「正常宽度仍是一行右对齐」——但那是拿两按钮状态说的。已授权的三按钮状态（Change folder + Build index + Revoke）在面板 448px 下，三个按钮加间距要 302px，而行内可用宽度只有 308px：按钮确实还在一行右对齐，代价是说明文字被压到 6px，107 个字符排成 13 行、一行一个字母。几何断言全绿而产品不可读——和这份文档开头那条「CSS 锁死 `flex-wrap: nowrap`」犯的是同一类错，只是这次断言量的是真实坐标，漏的是「读得了」这个维度。

  所以中止条件那条 **已触发，且触发范围不止堆叠断点**：现在的规则是按钮组拿一个上限（`max-width`，说明文字要留下的宽度写在里面），配 `flex-shrink: 0` 使它只有「自然一行宽」和「上限」两种宽度，换不换行由这两者是否相等决定，而不是由面板宽度阈值决定——阈值不知道行里有几个按钮，两按钮在 448px 放得下，三按钮放不下。主 CTA（Build index）在任何一档都可见，这一点由新增断言直接守着。

  代价如实记下：两按钮状态在面板 448px 下说明文字仍只有 81px，因为「两按钮一行」与「文字 150px」在 308px 的行里数学上互斥，而前者由 `library-row-geometry` 锁着，本次未放宽。

- **最终结论 (2026-08-30)**：上面那条代价不该被接受，它是规则写歪的症状，不是权衡的结果。可读下限当时只写进了三按钮那条断言，两按钮那条把「448px 必须一行」当成期望值锁着——于是出现了荒谬的对比：**三按钮换行、说明文字 176px；两按钮一行、说明文字 81px。按钮越多，这一行反而越好看。**

  本阶段的验收结论定为：**说明文字保持可读是统一不变式；按钮组能在不把说明压到可读下限以下的前提下排一行，就排一行，排不下就换行。这条规则对任何按钮数一样。**

  实现上只剩一个旋钮：`--arxiv-daily-library-description-floor: 176px`，即说明列最窄可以到多少还读得成一句话。按钮组的上限 `max-width: calc(100% - var(--…-floor) - 16px)` 与说明列的下限 `min-width: var(--…-floor)` 都由它推出，配 `flex-shrink: 0` 让按钮组只有「自然一行宽」和「上限」两种宽度——换不换行由这两者是否相等决定。**规则里不再出现按钮个数**：上一版按 `:has(> button:nth-of-type(3))` 分出的第二条上限已删除，两按钮、三按钮、以及将来任何按钮数走同一条路。堆叠档（`@container (max-width: 340px)`）把上限与下限一起解除，因为那里根本没有第二列。

  由此产生一处**有意的行为变更**：两按钮状态在面板 448px（窗口 700px）下现在会换行，排成 2 行、每行右对齐，说明文字从 81px / 每行 10.7 字符变成 176px / 每行 26.8 字符。这是本阶段任务里「按钮一行右对齐、不换行」那条要求的正式退役——它在窄面板下与可读性直接冲突，冲突时让位的是「一行」。宽面板（848px）下两按钮与三按钮仍都是一行右对齐，主 CTA（Build index）在任何一档都可见不被裁切。

  相应地，`library-row-geometry` 这条断言本身被改严：从「448px 必须一行」改成与三按钮同一套判据（`judgeLibraryWrappedGeometry` + `judgeDescriptionReadable`），即「能一行则一行、否则换行且每行右对齐」加上说明可读与主 CTA 可见。`judgeDescriptionReadable` 的 150px / 12 字符下限与 1.5px 容差一个字未改。详见 journal 2026-08-30「可读下限升成统一不变式」。

## Abort / reshape triggers

- 如果隐藏远程字段导致无法完成远程授权，停下来先保证授权入口仍从文献库主按钮进入。
- 如果按钮在窄设置栏里被裁切到看不见主 CTA，改回允许换行，但主按钮仍须可见。**已触发两次**：一次在 Obsidian 的堆叠断点之下（按钮横铺出面板），一次在三按钮状态的窄面板（按钮没被裁切，被挤没的是它旁边的说明文字）。两次都按它执行了——允许换行，主 CTA 始终可见。第二次同时说明这条触发器写窄了：需要保护的不只是主 CTA 可见，还有说明文字可读。

# P6 — scannable-proposal-list

goal_ref: ../goal.md
created: 2026-09-01T20:32:18+08:00
updated: 2026-09-01T20:32:18+08:00
revision: 1

## Outcome

Proposed 页能被扫读：一屏看到多个候选、主操作不用滚就看得见、内容不被横向裁切；想改某一条时就地展开成现有的编辑表单。

## Assumptions

- **裁切的根因是断点绑错了对象**。表单是两列网格、最小 12rem + 16rem（约 460px），而「塌成一列」写在 `@media (max-width: 700px)` 上——那是**窗口**宽度，模态框自己是 `min(60rem, 92vw)`。窗口略大于 700px 时断点不触发，模态框却已装不下两列，于是横向溢出。改成按模态框自身宽度判断（`container-type: inline-size`，与设置页同法）即可。若换成容器查询后仍溢出，说明还有别的固定最小宽度在起作用，需重新测量而不是继续调阈值。
- **现有能力一条不丢**。编辑、四个按钮、证据展开、簇成员展开全部保留，只是默认收起。
- **P2 的核心不动**。批量确认的事务语义（一次落盘、中途失败不留半截）已验收且新页面继续用它。
- 摘要行显示什么，取现成数据即可：线索条数、代表论文数、thin evidence 徽标。不新增字段、不新增计算。

## Approach

卡片拆成「一行摘要 + 折叠详情」。摘要行常驻：勾选框、名称、thin 徽标、`N cues · M papers`。详情默认收起，展开后是今天的表单与四个按钮。「Accept selected」连同选中计数移到列表**顶部**。样式上把塌成一列的判断从窗口媒体查询换成模态框容器查询。

## Chunks

### Chunk 1 — 候选行默认折叠，详情就地展开

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 新增测试——渲染两个候选时，两行摘要都在（名称、线索数、代表论文数可读），而编辑表单的输入框默认**不在**可见 DOM 里（或其容器为收起态）；展开一行后该行的表单与四个按钮出现。红在表单默认就渲染出来。
- Green check: `npm run test --workspace obsidian-arxiv-daily -- personal-library-interest-profile-modal`
- regression checks: 既有的编辑/保存/确认/丢弃/合并断言仍绿（它们要先展开才能取到控件）；插件全量；`npm run typecheck`。
- exception: 无
- [ ] implementation and tests accepted

### Chunk 2 — 主操作移到列表顶部并显示选中计数

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 新增测试——「Accept selected」在 DOM 中出现在第一张候选卡**之前**，且按钮文字带当前选中数量。红在它仍排在所有卡片之后。
- Green check: 同上测试文件。
- regression checks: 接受行为本身的既有断言（只提交勾选的、只调用一次）仍绿。
- exception: 无
- [ ] implementation and tests accepted

### Chunk 3 — 塌成一列改由模态框自身宽度决定

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 单测层面只能守规则本身——新增对 `styles.css` 的断言：两列网格的塌陷规则必须写在容器查询里、且不存在以窗口宽度为条件的同名规则。红在当前写的是 `@media (max-width: 700px)`。**这条断言证明的是「我们写对了规则」，不是「它不再裁切」**，后者只有 Chunk 4 能证明。
- Green check: 同上测试文件。
- regression checks: 窄屏塌叠行为不倒退——由 Chunk 4 的几何断言覆盖。
- exception: 无
- [ ] implementation and tests accepted

### Chunk 4 — 桌面验收指向这个模态框，几何被量出来

- change kind: behavior change（验收覆盖）
- strategy: 先取红（在修复前跑应为红），再取绿
- Red / baseline signal: 新增验收场景在真实渲染进程里打开复审页并测量：内容不横向溢出模态框、「Accept selected」在不滚动的情况下可见、首屏至少能看到 N 个候选行。修复前应红在溢出与主操作不可见上。
- Green check: `OBSIDIAN_TEST_VAULT=/home/tiandc/Desktop/plugin_test npm run test:desktop`
- regression checks: 既有 21 条设置页场景全绿、零 console / pageerror。
- exception: 无
- [ ] implementation and tests accepted

## Phase verification

- core 与 plugin 全量、`npm run typecheck`、`npm run lint` 0 error、`npm run check:boundaries`、`npm run test:desktop`。
- **交付证据必须包含用户实际看过**。这一阶段的起因正是「测试绿 + 我认为够用」而用户打开发现没法看；本阶段不重复该错误：截图与几何断言只是必要条件。

## Abort / reshape triggers

- 若换成容器查询后仍横向溢出，停下测量而不是调阈值——说明还有别的固定最小宽度，属另一类缺陷。
- 若折叠后用户仍觉得扫不动（例如一屏仍只有三四行），说明摘要行本身太高，下一步是削摘要行内容，而不是把表单加回去。

# P2 — readable-evidence-cards

goal_ref: ../goal.md
updated: 2026-08-29

## Outcome

Dashboard 与 Similar papers 的文献证据看起来像一张结果卡片：标题、文件、一段正文、页码和打开动作分得开；正文不再和按钮糊成一块。

## Assumptions

- 论文排序和每篇保留的 hit 集合不变；P3 才处理版权页/目录/参考文献作为默认主证据。
- Dashboard 与 Similar papers 已共用 `renderLibrarySearchBlock`；改这一处两边都会变。
- 当前难看主要来自：无层次的纯文本堆叠、按钮文案重复页码、正文 360 字截断仍显得像墙、无章节时空白、分数文案像调试输出。

## Approach

只改投影与样式。卡片结构固定为：标题；次要元数据（文件名，必要时才显示分数）；每条证据一块引用（章节若有、正文、页码+打开）。打开按钮改成短动作（如 `Open page 4`），页码仍单独可见。空白折叠成单空格，去掉连续点线，截断更短。

## Test strategy

- change kind: behavior change（呈现）
- strategy: strict Red–Green–Refactor on the shared renderer
- Red / baseline signal: 现有 `library-search-block` 测试在新结构下失败（按钮文案、元数据格式）
- Green / regression checks: 更新后的 renderer 测试；similar-papers / command modal 仍能点到同一 openEvidence；不改 Core 检索测试

## Tasks

- [x] 固定卡片结构：标题、文件名、每条证据的章节/正文/页码/打开。
- [x] 打开按钮改为短文案，页码仍单独可见；aria-label 保留页码。
- [x] 正文做空白折叠与点线压缩，截断到可读长度。
- [x] 样式让证据块像引用而不是又一段正文。

## Verification

- 同一 fixture：标题、文件名、章节、正文、`Page N`、打开动作都在，且按钮不再重复整句 `Open PDF at page N`。
- Similar papers 与 Dashboard 仍走同一 renderer。
- 不改变 match 顺序或 hit 条数。
- Observed Green (2026-08-29): plugin 19 tests（library-search-block / similar-papers / command modal）与 typecheck 通过。按钮可见文案为 `Open page N`，aria-label 仍为 `Open PDF at page N`；目录点线压缩为 `…`；分数调试文案不再出现。

## Abort / reshape triggers

- 如果必须改 Core 排序或 hit 选择才能让卡片好看，停在 P2，把选择问题留给 P3。
- 如果样式必须动日报阅读列表才能生效，停止并只改 library block 的 class。

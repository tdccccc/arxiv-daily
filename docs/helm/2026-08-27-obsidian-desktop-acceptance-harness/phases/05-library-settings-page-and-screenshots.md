# P5 — library-settings-page-and-screenshots

goal_ref: ../goal.md
updated: 2026-08-30

## Outcome

`2026-08-30-library-setup-path` 的 P1c / P1d 改动（Personal library 分组下移、Library 行最多三按钮无 Manage、远程授权改为就地确认）在真实 Obsidian 渲染进程中产出断言级证据，取代「用户手动开 Obsidian 肉眼确认」这一步；并在关键状态落盘截图，供人判断断言无法覆盖的部分（措辞是否得体、时机是否突兀、观感是否拥挤）。

## Assumptions

- 本机 Obsidian 1.11.5 走的是 legacy `display()` 渲染路径，不是 1.13+ declarative 路径。分组顺序在两条路径上各写一份，因此本场景验的是 legacy 一条；declarative 那条仍只有单元测试覆盖。这一点必须实测确认，不得假定。
- 设置页可用 `app.setting.open()` + `app.setting.openTabById("arxiv-daily")` 打开，行结构为 `.setting-item` / `.setting-item-heading` / `.setting-item-name` / `.setting-item-control`。
- 文献库连接状态可由 fixture 的 `data.json` 顶层 `libraryConnection` 预置；`rootIdentity` 必须是文件夹真实的 `dev:ino`，因为 `ScopedLibrarySource` 会重新 stat 校验。
- 面板宽度可由 CDP `Emulation.setDeviceMetricsOverride` 驱动窗口宽度间接得到，从而落在 Obsidian 自己的响应式规则内，而不是注入内联样式伪造一个用户到不了的布局。
- 「取消后未开始索引」可由 `plugin.operations.snapshot()` 与通知文本共同观察，无需真的跑一次索引。

## Approach

新增一个独立 session：既有四项场景需要「早于 sidecar 的旧 settings」，本场景需要「已连接未授权的文献库」，两种持久化状态无法共存于一次启动，因此按 session 分开，各自由既有 state guard 括起来。

场景一次走完 local → 远程切换被取消 → 遗留 remote 未授权 → Build index 被取消四个状态，每到一个状态断言一次、截图一次。判定逻辑（顺序、按钮集合、几何、状态未变）抽成纯函数，可在无渲染进程时单测；取数表达式集中在一处，失败信息一律回读页面实际值，使断言不可能恒真。

截图走 CDP `Page.captureScreenshot` 的 `clip`，只截目标元素区域；输出目录 `.acceptance-out/`（已被 `.gitignore` 的 `.*` 覆盖），每轮覆盖，不入库、不比对。

## Test strategy

- change kind: behavior change（新增验收能力）
- strategy: 判定逻辑与截图写入走 strict Red–Green–Refactor（注入假 client / 假 fs / 构造的几何数据）；真实宿主行为走 Green characterization，并以「篡改期望值后必须变红」作为反向证据
- Red / baseline signal: `node --test scripts/tests/desktop-acceptance-library-settings.test.mjs scripts/tests/desktop-acceptance-screenshots.test.mjs` 在实现前因模块缺失而失败
- Green / regression checks: `npm run test:release-tools` 全绿；一次真实运行产出全部断言与五张截图；`npm test`、`check:boundaries` 不受影响
- exception: 真实布局几何无法在单元层制造有意义的 Red（happy-dom 无布局引擎），改为在真实运行中逐条篡改期望值取得反向证据

## Tasks

- [x] `screenshots.mjs`：按元素矩形裁剪的 PNG 写入，slug 文件名，git 忽略的输出目录，零新增依赖。
- [x] `cdp.mjs` 视口控制：`Emulation.setDeviceMetricsOverride` / `clearDeviceMetricsOverride`，用于取得两种真实可达的面板宽度。
- [x] `settings-fixture.mjs`：已连接未授权的文献库 fixture，`rootIdentity` 取自真实 `dev:ino`；库根从测试 vault 内部挑选，不引用 vault 之外任何路径。
- [x] `library-settings.mjs`：分组顺序、按钮集合（local 与 remote 两态）、布局几何（两种宽度）、切远程弹披露、取消回滚、Build index 前置披露与取消、场景内零渲染错误。
- [x] `runScenarios` 支持一个场景返回多条结果，使一次页面走查能逐项报告而不是给一个笼统结论。
- [x] 反向证据：逐条篡改期望值后重跑，确认每条断言都会变红且失败信息回读页面实际值。
- [x] 堆叠档断言：`library-row-geometry-stacked`。Obsidian 在 `@container (max-width: 340px)` 下把设置行变成一列、把按钮拉成整行宽，「同一行」「右对齐」在那里不再有意义，能守的只剩「不溢出控件盒、不溢出行」，故另立判据而非复用。
- [x] 已授权的三按钮状态：`library-row-three-buttons-geometry` 与 `library-row-three-buttons-geometry-stacked`。场景走真实披露框拿到 Revoke——从 Embedding 下拉切 remote 再点 Authorize，不走 Build index（后者会顺带真的开始索引，越过本阶段「不真跑索引」的边界）。
- [x] 可读性判据：`judgeDescriptionReadable`。三按钮档不能只断言「没重叠」——坏掉的那一档正是没重叠但不可读（说明文字 6px 宽，107 个字符 13 行，一行一个字母）。判据取两条下限：说明文字可用宽度 ≥ 150px，且平均每行 ≥ 12 个字符；单看宽度会放过「宽盒子里一行一个词」，单看每行字数会放过「窄缝里两行短句」。行数由 `Range.getClientRects()` 数真实行盒得出，不靠高度除以假定行高。
- [x] 布局几何转绿。红的产品缺陷由 `2026-08-30-library-setup-path` 修，本 initiative 只出断言与证据；三轮（按钮溢出、三按钮说明压成一列、两按钮说明压成 81px）均先取红再转绿。
- [x] 可读性升为统一不变式：`library-row-geometry` 从 `judgeLibraryGeometry` 改为 `judgeLibraryWrappedGeometry`。这是本 initiative 迄今唯一一次改动既有断言，方向是**加严**——见下方「断言变更」。
- [x] 披露框改为按稳定 class 定位，标题另立断言：`DISCLOSURE_MODAL_CLASS` + `judgeDisclosureTitle`，新增 `remote-disclosure-title`（15 项 → 16 项）——见下方「定位方式变更」。

## 断言变更（2026-08-30）

`library-row-geometry` 原来对两按钮档跑 `judgeLibraryGeometry`，等于把「面板 448px 下必须一行右对齐」当成期望值。那正是要修的坏状态：一行被锁死的代价是说明文字只剩 81px。于是这条断言本身改成与三按钮同一套判据。

- 改前：`const verdict = judgeLibraryGeometry(geometry);` —— 要求同一行（tops 展开 ≤ 1.5px）、贴右缘、不溢出控件盒 / 行、不压住描述。对说明文字是否读得了**没有任何要求**。
- 改后：`const verdict = judgeLibraryWrappedGeometry(geometry);` —— 保留上面除「同一行」以外的全部内容，把「同一行」换成「每一行都贴住控件右缘」，并追加两条原本没有的：主 CTA（Build index）必须被排出可见尺寸且不被行裁切；`judgeDescriptionReadable`（说明列 ≥ 150px 且平均每行 ≥ 12 个字符）。

这是**加严不是放宽**：判据从五条变七条，放弃的只有「行数」这一条——而行数本来就不是承诺，可读才是。`MIN_DESCRIPTION_WIDTH_PX = 150`、`MIN_DESCRIPTION_CHARACTERS_PER_LINE = 12`、`ALIGNMENT_TOLERANCE_PX = 1.5` 三个常量一个字未改。其余断言（`library-row-geometry-stacked`、三按钮两条、按钮集合、分组顺序、弹框时机、零渲染错误）一条未动。`judgeLibraryGeometry` 保留但已无场景调用，其单元测试与 doc comment 一并保留，comment 里写明了为何不再被应用。

## 定位方式变更（2026-08-30）

披露框原来是**按标题文本**找到的：`Array.from(document.querySelectorAll(".modal-container .modal")).find((m) => m.querySelector(".modal-title").textContent.trim() === "Authorize personal library")`。取数、点 Cancel、点 Authorize 三处都走这一条，也就是说整个同意流程的可达性挂在一句产品文案上。

这是个错报的来源，不是理论隐患：文案一改，场景就红在「切下拉没有弹出披露框」上，读起来像同意机制坏了，而实际只是标题换了词。失败信息指向的地方是错的。

- 改前：`modalExpression(title)` / `clickModalButtonExpression(title, label)` 用标题文本选中弹框；`DISCLOSURE_TITLE` 常量同时充当定位键和（隐含的）文案期望。
- 改后：`DISCLOSURE_MODAL_CLASS = "arxiv-daily-library-authorization-modal"`，插件在披露框根元素上打这个 class（`plugin/src/library/modal.ts`，沿用 `arxiv-daily-hub-modal` 的既有做法）。`modalExpression()` 按 `.modal-container .modal.<class>` 定位，并把读到的标题当**数据**返回；`clickModalButtonExpression(label)` 同样按 class 定位。
- **断言与定位分开**：新增判据 `judgeDisclosureTitle(modal, depth)` 与 `DISCLOSURE_TITLES`（`full-text` → `Send full text off this device?`，`metadata-and-abstracts` → `Send titles and abstracts off this device?`），新增场景断言 `remote-disclosure-title`。文案改了红在「标题不符」，框没弹才红在「找不到框」。

这是**加严不是放宽**：原来标题只是被隐式当作定位键，从来没有一条断言说「标题应当是什么」；现在它有了独立断言，且期望值随处理深度取。既有断言（几何四条、按钮集合两条、分组顺序、弹框时机两条、取消回滚、零渲染错误）与 `MIN_DESCRIPTION_WIDTH_PX` / `MIN_DESCRIPTION_CHARACTERS_PER_LINE` / `ALIGNMENT_TOLERANCE_PX` 三个常量一个字未改。

class 两头各有单测守着，改哪一头都红：插件侧 `plugin/tests/library-modal.test.ts` 断言弹框根元素带这个 class，验收侧 `scripts/tests/desktop-acceptance-library-settings.test.mjs` 断言取数与点击表达式里出现这个 class、且不再对 `.modal-title` 做等值匹配。两处都写死字面量而不是从被测模块 import，否则断言会与实现自说自话。

**`metadata-and-abstracts` 那条标题本场景够不到**：设置页的 `requestRemoteFullTextConsent` 在没有嵌入接口可披露时返回 `undisclosable`、根本不开框，所以这个页面能弹出来的披露只可能是 full-text 深度。metadata 深度的标题在 `plugin/tests/library-modal.test.ts` 覆盖，那里可以直接调 `confirmLibraryAuthorization`。这一点写在 `DISCLOSURE_TITLES` 的注释里。

## Verification

- 定向：`node --test scripts/tests/desktop-acceptance-*.test.mjs` 全绿（`library-settings` 29 例，含三按钮换行、右对齐每行、主 CTA 被压成零宽、可读性两条下限各自的红，以及定位 class 与两条深度标题各自的红）。
- 端到端：一次真实运行中 16 条断言全绿，十张截图与其命名状态一致。
- 三按钮断言的红：加断言、未改 CSS 时跑出 `FAIL  library-row-three-buttons-geometry / narrow panel 448px (window 700): the description column is 6px wide, under the 150px a sentence needs to read as one; its 97 characters are spread over 13 lines — 7.5 characters a line, under 12`；修复后同一条读出 176px 宽、每行 24.3 个字符。
- 两按钮断言改严后的红：只换判据、未改 CSS 时跑出 `FAIL  library-row-geometry / narrow panel 448px (window 700): the description column is 81px wide, under the 150px a sentence needs to read as one; its 107 characters are spread over 10 lines — 10.7 characters a line, under 12`。数字全部来自真实渲染进程的测量，不是构造值。改完 CSS 后同一条读作 `narrow panel 448px (window 700): 2 lines, each right-aligned, Build index visible at 496.1..588, description 176px wide, 26.8 characters over 4 lines; wide panel 848px (window 1400): 1 line, each right-aligned, Build index visible at 1081.1..1173, description 481px wide, 53.5 characters over 2 lines` —— 两按钮档在窄面板换行是**有意的行为变更**，宽面板仍是一行。三按钮那条与两条堆叠断言的数字逐字未变。
- 反向证据：把期望分组顺序倒置、把期望按钮改成 `Manage`、把披露文案匹配串改成不存在的短语、把取消后期望的下拉值改成 `remote`、把「未开始索引」反转、在走查开始处注入一条 `console.error` —— 六项各自使对应断言变红，且失败信息里带着页面的实际值。
- 定位与文案解耦后的红：把 `DISCLOSURE_TITLES["full-text"]` 临时改回退役文案重跑，红的原文是 `FAIL  remote-disclosure-title / the disclosure is titled "Send full text off this device?", expected "Authorize personal library" at full-text depth` —— 红在标题不符，而同一次运行里 `remote-switch-asks-in-place`、`declined-remote-switch-changes-nothing`、`build-index-asks-before-remote-indexing` 与两条三按钮断言仍全绿，后三者都必须先找到弹框、点它的按钮才能成立，证明定位已经不吃文案。
- 安全边界：运行前后用户真实 Obsidian 会话存活；`data.json` 与 `workspace.json` 逐字节还原；截图目录被 git 忽略，`git status` 干净。
- 门禁：`npm test`、`check:boundaries`、`test:release-tools` 通过；桌面验收不进入默认 `npm test`。

## Abort / reshape triggers

- 如果断言跑不过是因为产品有缺陷，停止并汇报，不得在本 initiative 内修改 `plugin/` 或 `packages/` 下的产品代码，也不得调低断言以求转绿。**已触发**：`library-row-geometry` 在两种宽度下均红，原因是 `09d43f1` 引入的 `.arxiv-daily-settings__library-controls` 同时设了 `min-width: 0` 与 `flex-wrap: nowrap`，前者让控件盒收缩到内容以下、按钮向左溢出并压住描述文字（1400px 窗口下 13.6px，700px 窗口下 118.2px），后者在 Obsidian 窄屏堆叠布局（窗口 < 700px）下让按钮横向铺出面板之外。已落盘截图为证，未改产品代码。**第二次触发**：`library-row-three-buttons-geometry` 在面板 448px 红——三个按钮一行占满 302px，只剩 6px 给说明文字，一行一个字母。同样只出断言与截图，产品侧的修法记在 `2026-08-30-library-setup-path` 的 journal 与 P1b 更正里。**第三次触发**：修完三按钮档后出现了荒谬的对比——三按钮换行、说明 176px，两按钮被 `library-row-geometry` 锁在一行、说明 81px，按钮越多这一行反而越好看。这一次红的原因不只在产品，也在断言：那条断言把「448px 必须一行」当成了期望值。于是按上面「断言变更」一节改严判据先取红（81px / 每行 10.7 字符），产品侧再收敛成一条由可读下限推出的样式规则转绿。
- 如果截图需要注入内联样式才能造出目标宽度，停止并在文档中说明这是伪造的布局，不得把它描述为「用户可达的宽度」。未触发：两种宽度均由窗口尺寸驱动。
- 如果场景需要真的跑一次索引或发出真实网络请求才能断言，停止并把断言收敛到设置层与对话框层。未触发。

## 按钮定位方式变更（2026-08-31）

上一轮只把**弹框**的定位从标题文本换成了稳定 class，**按钮仍然按文案点**：`clickModalButtonExpression` 收的是按钮文字，在弹框里找 `textContent` 相等的 button。于是改确认按钮文案会直接弄坏验收，且红的原文是「the modal has no Authorize button」——读起来像按钮不见了，实际只是换了词。这是标题那次同一个失败模式，只是从弹框挪到了按钮。

两个按钮各加一个稳定 class（沿用 `arxiv-daily-` 前缀）：`arxiv-daily-library-authorization-confirm`、`arxiv-daily-library-authorization-cancel`。`clickModalButtonExpression(buttonClass)` 收标识不收文案，`modalExpression()` 把两个按钮的文字当**数据**读出来。新增判据 `judgeDisclosureButtons(modal, depth)` 单独断言文案。定位与断言就此分开：文案改了红在「按钮文案不符」，按钮真没了才红在「找不到那个 class 的按钮」，两种失败各自指得准，各有单测守着且断言互相排斥。

新增场景断言 `remote-disclosure-buttons`，场景总数 16 → 17。

- 红的证据：把 `DISCLOSURE_CONFIRM_LABELS["full-text"]` 临时改回 `Authorize` 重跑，红的原文是 `FAIL  remote-disclosure-buttons / the confirm button reads "Send full text", expected "Authorize" at full-text depth`——红在**文案不符**而非找不到按钮。同一次运行里 `remote-switch-asks-in-place`、`declined-remote-switch-changes-nothing`、`build-index-asks-before-remote-indexing` 与两条三按钮断言全绿，而这几条都必须先找到弹框、点它的取消或确认按钮才能成立（三按钮状态正是靠点确认按钮授权出来的），证明按钮定位已不吃文案。
- `metadata-and-abstracts` 深度的按钮文案本场景够不到，原因与标题那条相同（设置页在无嵌入接口时返回 `undisclosable`、不开框），在 `plugin/tests/library-modal.test.ts` 覆盖。

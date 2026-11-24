# 0.6.7 后探索的两项待复核问题：Run today 重复反馈与 More 菜单键盘焦点

> 发现日期：2026-11-22（0.6.7 发布后探索）· 复核与处理：2026-11-24 · 来源：模型探索（Obsidian，DeepSeek 官方，thinking 开，high）+ 人工复核
> 范围：`plugin/src/dashboard/view.ts`、`plugin/src/commands.ts`、`plugin/src/settings/tab.ts`、`scripts/acceptance/exploration-obsidian.mjs` · 严重程度：低（体验） · 状态：已修复（分支 `fix/0.6.8-run-today-busy`） · 置信度：证据复核
> 探索产物（本地，不入库）：`output/playwright/acceptance/2026-11-22-067-obsidian/exploration/`

## 问题 1：Run today 出现“running…”与“skipped: already done”并存

探索在 `impatient-clicker` 第 118 步记录：快速连点 Run today 后，同时出现“arXiv Daily: running for 2026-11-22…”和“arXiv Daily 2026-11-22: skipped: already done”，且按钮看起来仍可用。

复核动作日志后确认：第 117 步其实只有**一次**点击（`impatient-clicker-step-118.png` 中两条通知都来自这一次点击）。当天日报已存在，`runToday` 先弹出“running…”通知，再调用调度器，调度器发现该日期已完成并返回 `skipped: already done`。也就是说，问题不是“重复提交”，而是**一次无效运行也先宣告自己在运行**，两条通知互相矛盾。原始 finding 对“重复点击”的描述与证据不符，但反馈缺陷本身成立。

### 处理

- 所有手动单日运行入口（Dashboard 工具栏/空态/日历、More 菜单的 Run for date…/Force run for date…、命令面板、设置页“Generate first report”）保留“running…”通知的句柄；结果为 `skipped`（已完成、锁被占用、不可运行）时隐藏该通知，只留下最终结果。
- 新增插件级单飞闸门 `plugin/src/services/manual-run-gate.ts`，按日期记录手动运行：同一天正在运行时，第二次请求立即提示“DATE is already running”，不再排到日期锁后面等第一次结束、再报告过期的“already done”。不同日期互不影响；跨窗口仍由调度器自己的跨进程锁协调。
- 新增测试：`plugin/tests/manual-run-gate.test.ts`、`plugin/tests/commands.test.ts`（跳过时隐藏通知、运行中拒绝第二次、结束后恢复）、`plugin/tests/dashboard/calendar-state.test.ts`（同样的窗口侧行为）。

## 问题 2：打开 More 菜单后键盘焦点逸出、Tab/Escape 被探索工具拒绝

`keyboard-only` 第 130 步记录：按 Enter 打开“More arXiv Daily actions”后，观察显示 `focus.outside=true`，后续 Tab 与 Escape 均返回“The focused element is outside the plugin's scope; the key press was not sent”，菜单无法用键盘关闭。

### 复核结论：探索工具限制，不是产品缺陷

- 该菜单是 Obsidian 原生 `Menu`（`plugin/src/dashboard/view.ts` 的 `showMoreMenu`），菜单项是普通不可聚焦的 `div`，插件没有也不应该接管它的焦点。
- 只读检查本机安装的 Obsidian（`/opt/Obsidian/resources/obsidian.asar` → `app.js`）确认：`Menu` 构造函数自建 `Scope`，并注册 `ArrowUp/ArrowDown/ArrowLeft/ArrowRight/Enter/Escape`，其中 `Escape` 调用 `menu.hide()`。真实会话里 Escape 会由 Obsidian 自己的键盘作用域关闭菜单，与 DOM 焦点无关。
- 探索工具在焦点不在插件界面内时拒绝发送 `press` 按键（`scripts/acceptance/exploration-obsidian.mjs`，有意的安全边界；keyboard-only 角色又不允许使用会直接发 Escape 的 `back` 动作），因此无法驱动这个原生菜单。维持工具边界，不为此放宽按键限制，也不改产品。

### 处理

记录于发布说明 `docs/releases/0.6.8.md` 的 Exploration notes，不产生产品改动。

## 附带：探索工具对“早于最近窗口的日期”的分类

`first-time-user` 角色输入了 `2025-06-10`，产品正确地把它记为永久失败并记录日志“date 2025-06-10 is older than the oldest cs.AI /recent bucket …; arXiv no longer lists it there”，但探索循环把任何观察到的控制台错误都判为场景失败。现对该产品日志做窄豁免（`isExpectedOutOfRangeDateLog`）：只接受控制台 error、且消息前缀完全匹配；页面异常（`pageerror`）和其它控制台错误仍然照常失败。测试见 `scripts/tests/acceptance-exploration-host.test.mjs`。

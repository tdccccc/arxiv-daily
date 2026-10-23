# 探索的“只用键盘”角色看不到焦点在哪，在 Obsidian 里还会被带出插件后无路可回

> 发现日期：2026-10-21 · 来源：模型探索（gpt-6.1-sol，high）+ 人工复核 · 编号 EXP-05
> 范围：`scripts/acceptance/exploration-browser.mjs`、`exploration-obsidian.mjs`（观察）、`exploration-tasks/personas.mjs`（keyboard-only 的角色设定）、Obsidian 探索的任务间重置 · 严重程度：中 · 状态：未修复 · 置信度：已复现

## 问题

`keyboard-only` 角色的全部意义是检查“只用键盘能不能用”：焦点能不能移动到可见的控件上、焦点是不是丢了、按键后有没有反应（角色说明里原话就是这样要求模型记 finding 的）。但两个宿主的观察里都没有“当前焦点在哪个元素上”这一项，模型只能盲按：

- 2026-10-21 的四次 keyboard-only 运行（工作台两次、Obsidian 两次）全是连续 10 到 12 次 Tab（工作台第一次中间夹了一次 Enter），每次的观察除了控件表外没有任何变化。模型无从知道 Enter 会激活什么，也无从判断焦点环是否可见。工作台的两次一条 finding 都没有；Obsidian 第一次的两条 finding 反过来是在抱怨观察器的缺口：“键盘导航时无法从界面状态确认焦点位置”（O3）、“键盘焦点逸出插件后，反向 Tab 被拒绝，无法返回可见控件”（O4）。
- 真实界面是有焦点反馈的：Obsidian 的截图里能看到按钮上的焦点环（`keyboard-only-step-114.png` 里“Settings”按钮），但模型拿不到截图，只拿到文字观察。

O4 在 Obsidian 里还有一条具体的路径把角色逼进死胡同：

1. 仪表盘所在的标签页不是 Obsidian 所说的“导航视图”（`ItemView` 默认 `navigation = false`）。在这种视图里按 Escape，Obsidian 自己的全局 Escape 处理（`obsidian.asar` 的 `app.js` 里 `app.scope.register([], "Escape", …)`，只读查看）会把最近用过的“导航”标签页激活，也就是之前任务留下的 Markdown 笔记标签页。
2. 探索环境不会在任务之间关掉这些笔记标签页（`reset()` 只关设置窗口和插件弹窗，见 `exploration-obsidian.mjs:1023-1058`），截图里能数到 6 个标签页。所以 keyboard-only 在第 8 次 Tab 之后按 Escape，焦点就跳进了论文笔记的属性编辑区（`keyboard-only-step-117.png`）。
3. 随后 `press` 被“焦点在插件范围之外”的安全规则拒绝（`exploration-obsidian.mjs:709-710`，这条规则本身是 2026-10-13 有意加的）。而角色说明只允许 press 和 wait，`home` 被模型理解为不允许，于是没有任何键盘方式回到插件，角色被卡住，只能提 finding 然后结束。

插件本身没有处理 Escape 或键盘事件的代码（`plugin/src/dashboard` 下没有 `keydown`/`Escape`），这个跳转是 Obsidian 对所有非导航视图的统一行为，不是插件缺陷。

## 证据

- `output/playwright/acceptance/2026-10-21-review-obsidian/exploration/`（本地，不入库）：`exploration-events.jsonl` 步骤 108-119（7 次 Tab 后 finding、再 Tab、Escape，步骤 118 的 Shift+Tab 返回 `ok:false`、错误为 “The focused element is outside the plugin's scope; the key press was not sent”，此时观察标题已变成 “2610.10001 - vault”，控件为 0）；`keyboard-only-step-116.png`（仪表盘有焦点环）、`keyboard-only-step-117.png`（Escape 之后进入笔记标签页）。
- 其余三次运行：`2026-10-21-review-workbench` 步骤 92-103、`2026-10-21-review-workbench-personas2` 步骤 37-48、`2026-10-21-review-obsidian-personas2` 步骤 37-48，全部是同样的盲按 Tab，没有 finding。
- 观察结构：`exploration-browser.mjs` 的 `browserObservation`（约 90 行起）和 `exploration-obsidian.mjs:294-335` 的 `obsidianObservation` 都没有焦点字段；焦点只在执行 `press` 时才被读取，用于安全检查（`exploration-browser.mjs:42-48`、`exploration-obsidian.mjs:~430-445`）。系统提示词（`exploration-model.ts:20-22`）告诉模型 press “作用于当前焦点”，却没给它看焦点。

## 原因

- 观察不包含 `document.activeElement`（两个宿主都是）。安全检查已经有现成的“描述当前焦点元素”的函数，只是没有放进观察里。
- `exploration-tasks/personas.mjs` 里 keyboard-only 的说明把动作限制为 press 和 wait，没有给“被带出界面之后怎么回来”留出口。
- `exploration-obsidian.mjs:1023-1058`：`reset()` 不关闭非插件的标签页，任务间会累积笔记标签页，Escape 才有地方可跳。
- Obsidian 的 Escape 行为见上。

## 可能的解决方案

推荐：

1. 两个观察都加一个 `focus` 字段：焦点在某个已编号的控件上时给出它的 `ref`、名称和标签；焦点在界面范围之外时给出 `{ outside: true }`；没有焦点时为空。直接复用现有的描述函数（工作台的 `liveControlDescriptor`、Obsidian 的 `obsidianFocusedControl`），并把 `data-acceptance-ref` 打到焦点元素上。系统提示词里补一句“`focus` 字段告诉你下一次 Enter/Space 会作用在哪里”。
2. keyboard-only 的角色说明里允许 `home` 作为唯一的“回到起点”恢复动作（它不属于鼠标类输入），并在角色说明里要求“用到 home 就提一条 finding 说明键盘无法回来”。
3. Obsidian 的 `reset()` 在每个任务开始前关闭仪表盘之外的标签页（例如对 `markdown` 类型调用 `detachLeavesOfType`），让每个任务从只有仪表盘的工作区开始。

备选：只做第 1 点。这已经让角色“看得见”，但在 Obsidian 里仍可能因为 Escape 被带出插件而浪费剩余步数。

## 修复后应补的测试

- `scripts/tests/acceptance-exploration-obsidian.test.mjs`：焦点在某个控件上时观察的 `focus` 带 `ref` 和名称；焦点在 `.workspace-leaf` 之外或另一个标签页里时是 `{ outside: true }`。
- `scripts/tests/acceptance-exploration-host.test.mjs`（工作台一侧同理）：Tab 之后观察的 `focus.ref` 变为下一个控件。
- `scripts/tests/acceptance-exploration-host.test.mjs`：`reset()` 之后工作区里只剩仪表盘标签页。
- `scripts/tests/acceptance-exploration-labels.test.mjs` 或角色测试：keyboard-only 的角色说明包含 `home` 的使用约定。

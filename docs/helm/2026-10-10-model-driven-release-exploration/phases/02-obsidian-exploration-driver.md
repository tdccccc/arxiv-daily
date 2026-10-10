# P2 — obsidian-exploration-driver

goal_ref: ../goal.md
created: 2026-10-11T01:40:00+08:00
updated: 2026-10-11T10:30:00+08:00
revision: 3

## Outcome

`--suite exploration --host obsidian` 在隔离的真实 Obsidian 中启动当前构建，模型只能看到并操作本插件界面（阅读面板、本插件设置页、本插件弹窗、提示），危险操作在观察和执行两层都被拦截，任务间复位到确定起点；受控模型和真实模型各跑通 1–2 个简单目标任务。

## Assumptions

- `scripts/desktop-acceptance/*`（启动、XDG 隔离、进程组、Vault 状态、诊断、截图）与 `scripts/acceptance/obsidian*.mjs`（设置夹具、HTTP 代理到固定响应）可直接复用；探索循环与动作校验与宿主无关（P1 已验证）。
- 本插件界面有稳定的 DOM 根可用来限定观察范围（如 `.arxiv-daily-dashboard`、本插件设置页容器、本插件弹窗的类名）；若某些弹窗无可识别标记，需要在观察层用插件类名/打开来源识别，或在 P2 内给产品加稳定标记（需单独验收）。
- CDP `Input.dispatchMouseEvent` / `Input.insertText` / `Input.dispatchKeyEvent` 足以产生真实点击、输入、按键与滚轮；现有 `waitForStableTarget` 可防止点到移动中的元素。
- 只能通过命令面板触发的功能（如按日期运行）留到 P3 决定：作为白名单“插件命令”控件暴露，或只经由界面入口。

## Approach

镜像 `exploration-browser.mjs` 新建 `exploration-obsidian.mjs`：先做可在 happy-dom 下单测的范围化观察与限制分类（纯函数，页面内执行），再做 CDP 动作执行与执行时重新校验，最后接入 `runExploration` 的宿主选择、会话启动、复位与截图，并用 1–2 个简单 Obsidian 任务做真实宿主验收。保持动作 schema 不变，除非键盘角色确实需要按键动作（若新增，宿主无关地加入并测试）。

## Chunks

### Chunk 1 — plugin-scoped observation and restriction policy

- change kind: behavior change (test infrastructure)
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 新测试 scripts/tests/acceptance-exploration-obsidian.test.mjs 在 happy-dom 构造的 Obsidian 样式 DOM 上断言：只收集本插件根内的可交互控件（不含 ribbon、文件栏、状态栏、其他插件、Obsidian 自身设置页）；密钥字段 sensitive；发测试邮件、清除运行状态、调度开关、选择资料目录（系统对话框）、外部链接被标 restricted；observation 形状与 workbench 一致（refs、rect、inViewport、layout）。当前模块不存在而失败。
- Green check: node --test scripts/tests/acceptance-exploration-obsidian.test.mjs
- regression checks: npm run test:acceptance:tools
- [x] implementation and tests accepted (f3b3563) — 设置页按 `app.setting.activeTab.id === "arxiv-daily"` 限定（真实宿主待 Chunk 4 验证）；10 个插件弹窗无标记，临时以“唯一打开的非设置弹窗”兜底；Obsidian `Menu`（阅读面板“更多”）对模型不可见，留给 P3 决定。

### Chunk 2 — stable marker on every plugin modal

- change kind: behavior-preserving product change (DOM marker only) + test infrastructure
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 插件测试断言所有插件弹窗（DatePickerModal、commands.ts 中 9 个弹窗、已有类名的 4 个弹窗）打开后 modalEl 带统一标记类（如 `arxiv-daily-modal`）；观察测试断言无标记的非插件弹窗不再可见、兜底规则移除。当前无标记而失败。
- Green check: plugin 相关 vitest；node --test scripts/tests/acceptance-exploration-obsidian.test.mjs
- regression checks: plugin 全量测试、typecheck、build；npm run test:acceptance:tools
- [x] implementation and tests accepted (7457134) — 共 13 个子类 + 7 个直接创建的弹窗加标记；源码守卫测试防止新弹窗遗漏。

### Chunk 3 — CDP action execution with live re-validation

- change kind: behavior change (test infrastructure)
- strategy: strict Red-Green-Refactor（CDP 客户端用可记录调用的假客户端单测）
- Red / baseline signal: 新测试断言 click 走稳定命中后的真实鼠标事件、fill 先聚焦再插入文本并触发 input/change、select、scroll（鼠标滚轮）、wait；执行前按当前 DOM 重新校验 ref 未失效且未变为 restricted/sensitive/disabled，否则拒绝且不发送任何输入事件；back/reload 在 Obsidian 下的映射或拒绝有明确行为。当前不存在而失败。
- Green check: node --test scripts/tests/acceptance-exploration-obsidian.test.mjs
- regression checks: npm run test:acceptance:tools
- [ ] implementation and tests accepted

### Chunk 4 — host wiring, reset, and first Obsidian tasks

- change kind: behavior change (test infrastructure)
- strategy: Red-Green for selection/wiring contracts (fake session); real Obsidian run as compensating verification for launch/reset
- Red / baseline signal: 新测试：`--host obsidian` 解析与传递；宿主选择决定驱动和任务库；Obsidian 默认任务需程序 verify（读 data.json / Vault 文件）；缺少 Obsidian/Xvfb 时整套 blocked 而非 passed。当前无宿主选项而失败。
- Green check: node --test 相关测试；受控脚本模型在真实 Obsidian 跑 1–2 个任务（如修改每日论文上限并验证 data.json、打开阅读面板收藏一篇论文并验证标记）均 passed，复位后第二个任务从确定起点开始。
- regression checks: npm run test:acceptance:tools；npm run test:acceptance -- --suite obsidian 仍全部通过
- [ ] implementation and tests accepted

## Phase verification

- 受控脚本模型：真实 Obsidian 中 1–2 个任务通过，一个尝试操作受限控件的动作被拒绝且未产生输入事件，报告含截图与逐任务用量。
- 真实模型：同样任务在小预算下运行一次（用户已接受 token 消耗），记录调用数与 token；结果如实记录（通过、失败或待复核）。
- npm run test:acceptance:tools 全部通过；固定 Obsidian 与 workbench 验收不回归。

## Abort / reshape triggers

- 若本插件界面无法可靠限定范围（弹窗或设置页无法区分来源）且需要较大产品改动，停下与用户确认后再加产品标记。
- 若 CDP 真实输入在 Xvfb 下不稳定到无法让受控模型可重复通过，先缩小动作集合；仍不行则 L2 重新划分。
- 若真实 Obsidian 每任务复位需要重启宿主且耗时过高（单任务复位 > 30 秒），记录数据后与用户确认取舍。

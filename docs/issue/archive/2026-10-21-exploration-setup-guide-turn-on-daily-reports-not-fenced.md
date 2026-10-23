# 探索的“禁止调度开关”拦不住首次使用清单里的“Turn on daily reports”按钮

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查（含 2026-10-21 当天真实模型探索证据） · 编号 SCR-04
> 范围：`scripts/acceptance/exploration-browser.mjs`、`exploration-obsidian.mjs` 的受限控件策略（workbench 与 Obsidian 两端） · 严重程度：低 · 状态：已修复（分支 `fix/2026-10-21-batch4`，提交 433bc0f） · 置信度：已复现

## 问题

`scripts/acceptance/README.md` 的发版探索说明承诺：“调度开关等危险或越权操作在驱动层直接拦截，不会真的执行”，模型的系统提示里也写着“不要修改……邮件或调度设置”。但两端的首次使用清单（Getting started，5 步）里最后一步的按钮——workbench 里的“开启自动日报”、Obsidian 里的“Turn on daily reports”——不在受限名单里，模型可以直接点：

- 用真实的 `classifyControlRestriction` 检查 2026-10-21 当天真实 workbench 探索记录里原样观察到的这个控件（`tag: button`、`name: 开启自动日报`、`field: ""`、`dataAction: ""`），返回 `undefined`（放行）；`parseExplorationAction` 也接受对它的 `click`。对照：同一功能的复选框 `schedule.enabled` 会被判为“external service and scheduling settings are fixed by the fixture”。
- Obsidian 一端同理：`Enable scheduler`（仪表盘“更多”菜单）被限制，`Turn on daily reports`（设置页清单）返回 `undefined`。当天 Obsidian 探索记录里有 23 次观察把它作为可点控件提供给模型，限制字段为空。
- 当天 4 次本地探索（两端各两次）里模型都没有点它，所以没有发生实际越界；但 `first-time-user` 角色的任务正是“摸索着完成必要设置、生成日报、打开论文”，面对“已完成 3/5、下一步：Turn on daily reports”的清单，点它是很自然的操作。

点了之后会怎样（代码推断）：workbench 里 `data-setup-action="enable"` 会勾上 `schedule.enabled` 并保存整份设置草稿，调度计时器随即启动（默认每 20 分钟检查一次，窗口内会发起 `run --scheduled`）；`reset()` 只恢复文献库状态和配置修订号，不会把调度关回去，所以这个状态会带进后面的任务和角色，后台定时运行可能占用唯一的运行槽，让后面“生成日报”类任务启动失败，得到与产品无关的失败或阻塞。Obsidian 里会弹出“Enable arXiv Daily / Cancel / Skip today / Run today”对话框（每个任务的 `reset()` 会重写 `data.json` 并重载插件，状态不会跨任务保留，但对话框本身是一个没有被预期的流程）。全部发生在一次性夹具里，不会触及用户真实数据或服务，所以严重程度只算低；问题在于文档承诺的边界和实际实现不一致，且受影响的是发版证据的可信度。

## 原因

- `scripts/acceptance/exploration-browser.mjs:17-18`：调度相关限制只匹配控件的 `name` 属性（`^schedule\.` 等）和 `data-action`/`data-settings` 里的 `email|library-connect|…`；清单按钮用的是 `data-setup-action`（`apps/cli/src/workbench/web/settings-setup.ts:23`），驱动读取 `dataAction` 时根本不看这个属性（`exploration-browser.mjs:138`），名字也不匹配。
- `scripts/acceptance/exploration-obsidian.mjs:527`：调度限制只列了设置页的 `Enable ·` 行、`check every (minutes)`、运行窗口字段和菜单项 `Enable/Disable scheduler`；设置页清单的 `Turn on daily reports`（`plugin/src/settings/tab.ts` 的 `renderSetupItem` → `enableDailyReports()` → `plugin/main.ts:2414` 的 `setScheduleEnabled(true)`）不在其中。
- 两个策略都是按“名字白名单”写的，每当产品新增一个会改调度的入口（这里是 5 步清单），策略就悄悄漏掉它，没有任何测试把“所有能打开调度的控件”和策略对应起来。

## 可能的解决方案

推荐：把清单按钮加入两端的受限策略（workbench：`dataAction` 读取增加 `data-setup-action`，并限制 `enable`；Obsidian：把 `^turn on daily reports$` 加进调度那条规则）。同时让 first-time-user 的 brief 说明“清单里的‘开启自动日报’不属于探索范围”，避免模型因为无法点击而反复尝试。

补强：在两端各加一个“真实产品文案”测试：从产品源码里读出所有会写 `schedule.enabled` 的入口文案（workbench 的 `settings-setup.ts`、Obsidian 的 `tab.ts`/`view.ts`），逐个断言受限策略会拦截它们。

备选：不限制按钮，改为在 `reset()` 里强制恢复调度设置（workbench 写回 `workbench_schedule.enabled = false` 并重新同步修订号）。这能消除跨任务污染，但与文档承诺的“驱动层拦截”不一致，不推荐。

## 修复后应补的测试

- 把 2026-10-21 当天记录里的两个控件（workbench `开启自动日报`、Obsidian `Turn on daily reports`）作为夹具，断言 `classifyControlRestriction` 返回“调度设置由夹具固定”。
- `press Enter/Space` 落在这两个按钮上时同样被拦截（现有的 press 策略复用同一分类函数，补一个用例确认）。
- 上述“所有调度入口文案”的源码对照测试。

## 修复情况

首次使用清单的调度按钮现在与设置页开关采用相同限制，鼠标点击及 Enter/Space 激活都不能开启调度。工作台读取 data-setup-action，两端中文/英文清单文案均受限；新用户角色明确排除这一步。4 项新增回归先失败，6 项调度/键盘测试及完整探索相关回归通过。

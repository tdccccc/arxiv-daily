# P3 — task-library

goal_ref: ../goal.md
created: 2026-10-11T16:00:00+08:00
updated: 2026-10-12T03:00:00+08:00
revision: 5

## Outcome

`--suite exploration --host obsidian` 默认运行 10–12 个 Obsidian 目标任务，`--host workbench` 在现有 3 个之外新增 2–3 个；两端各有 2–3 个用户角色可用 `--persona` 选择。每个目标任务有独立程序判定，并经受控模型证明“完成动作能通过、不做动作不能通过”。

## Assumptions

- P2 的 Obsidian 宿主、复位（整插件重载，约 0.2 s）、动作执行和两个任务可直接扩展；任务准备可用插件自身命令确定性地生成前置数据（上限 60 s）。
- 固定夹具已能模拟认证失败、取消与重跑（固定 Obsidian 场景已覆盖），可复用其开关。
- 阅读面板“更多”菜单（Obsidian `Menu`，无公开 DOM 句柄）可在“由本插件按钮打开”的前提下被识别；若做不到，相关任务改走插件界面内的其他入口或白名单命令控件。
- 程序判定依据 data.json、papers.json、Vault 文件、运行状态文件和重新打开后的界面，不依赖模型自述。

## Approach

先补齐入口：让模型能使用插件“更多”菜单中的安全项（调度开关仍禁止），并把任务库按宿主拆成独立模块。然后按功能分三批写 Obsidian 目标任务，每批都附受控模型的正向与反向（不做动作不通过）验证。最后补 workbench 任务与两端用户角色。真实模型的完整运行留给 P4。

## Chunks

### Chunk 1 — plugin menu access and task-library structure

- change kind: behavior change (test infrastructure; product marker only if needed)
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 观察测试断言：由本插件“更多”按钮打开的菜单项可见，调度开关项 restricted，非本插件菜单不可见；任务库按宿主分模块、id 唯一，`--list` 列出两端任务与角色。当前菜单不可见而失败。
- Green check: node --test scripts/tests/acceptance-exploration-obsidian.test.mjs scripts/tests/acceptance-exploration-host.test.mjs
- regression checks: npm run test:acceptance:tools；真实 Obsidian 中受控模型打开“更多”菜单并点击“按日期运行…”出现日期框
- [x] implementation and tests accepted (8767e5c, d8c0271) — 产品给“更多”菜单加 `arxiv-daily-menu` 标记；调度开关、清除运行状态、外部帮助链接受限；任务库移至 scripts/acceptance/exploration-tasks/。

### Chunk 2 — Obsidian settings and topic tasks

- change kind: behavior change (test infrastructure)
- strategy: Red-Green per task verifier（反向：不做动作时 verify 不通过），真实宿主受控模型正向通过
- Red / baseline signal: 每个任务的 verify 对“未操作”的初始状态返回未通过；任务尚不存在时选择报错。
- Green check: 受控模型在真实 Obsidian 中完成：设置修改后重载插件仍生效（扩展现有 daily-paper-limit）、新增研究主题方向、删除方向、调整方向顺序；data.json 与重开后的界面一致。
- regression checks: npm run test:acceptance:tools；固定 Obsidian 验收
- [x] implementation and tests accepted — 方向排序无界面控件，改为添加 arXiv 分类任务；本机 Obsidian 1.11.5 走旧版设置渲染，判定兼容两种渲染。真实宿主受控运行两次 5/5 通过。

### Chunk 3 — Obsidian run tasks

- change kind: behavior change (test infrastructure)
- strategy: 同 Chunk 2
- Red / baseline signal: 同 Chunk 2
- Green check: 受控模型完成：按日期生成日报（经菜单或日期框）、运行中取消后重跑、认证失败后在设置中修正再重试（夹具切换 401）、对已完成日期再次运行不重复生成；以运行状态文件、日报文件与索引判定。
- regression checks: npm run test:acceptance:tools；固定 Obsidian 验收
- [x] implementation and tests accepted — 认证失败任务：模型无法修改密钥，须先在运行历史中找到失败记录，程序才恢复夹具（模拟用户已修正）；日期字段经 DOM 填写。真实宿主受控运行两次 9/9 通过。

### Chunk 4 — Obsidian reading tasks

- change kind: behavior change (test infrastructure)
- strategy: 同 Chunk 2
- Red / baseline signal: 同 Chunk 2
- Green check: 受控模型完成：阅读面板切换标签并翻页、标记待读、生成单篇详细总结（Paper note）、按 arXiv 编号总结、打开今日日报、打开 PDF 到指定页（复用固定场景的判定）；以 papers.json、Vault 文件与界面判定。
- regression checks: npm run test:acceptance:tools；固定 Obsidian 验收
- [x] implementation and tests accepted — Obsidian 共 14 个目标任务；放弃 PDF 定页（界面无入口）；经用户同意在“更多”菜单新增“设置论文标记…”（产品改动单独提交）。真实宿主受控运行两次 14/14 通过。

### Chunk 5 — workbench tasks and personas for both hosts

- change kind: behavior change (test infrastructure)
- strategy: 同 Chunk 2（角色无 verify：受控模型证明以 explored 结束、疑点可见）
- Red / baseline signal: 选择尚不存在的任务/角色 id 报错；新 workbench 任务 verify 对初始状态不通过。
- Green check: workbench 新增 2–3 个任务（如收藏后切换筛选并恢复阅读、单篇详细总结、设置冲突后放弃重开）受控通过；两端角色（首次使用的新用户、没耐心快速乱点的用户、只用键盘的用户）受控运行以 explored 结束；`--list` 列出全部。
- regression checks: npm run test:acceptance:tools；固定 workbench 与 Obsidian 验收
- [ ] implementation and tests accepted

## Phase verification

- 受控模型分别在两端跑完整默认任务库：全部目标任务 passed，角色 explored；`--list` 显示 Obsidian 10–12 个、workbench 5–6 个任务与各 2–3 个角色。
- 每个目标任务都有“未操作不通过”的反向证据。
- npm run test:acceptance:tools、固定 workbench 与 Obsidian 验收均通过。

## Abort / reshape triggers

- 若某功能只能经由禁止的入口（系统对话框、外部链接、真实邮件）完成，删去该任务并在报告中记录，不放宽限制。
- 若单个任务的确定性准备超过 60 s 或复位无法隔离任务之间的 Vault 影响，先记录数据再与用户确认取舍。
- 若受控模型在真实宿主上无法稳定通过某任务（三次中两次以上失败），先查观察与动作是否足够，再决定删减或改写任务。

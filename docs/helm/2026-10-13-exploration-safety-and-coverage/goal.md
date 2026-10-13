# Exploration safety and coverage

status: active
created: 2026-10-13T10:05:00+08:00
updated: 2026-10-14T00:30:00+08:00
revision: 8
owner: main-session-3a9f449b

## Intent

补上模型驱动发版探索的安全漏洞，并把探索覆盖扩展到尚未被模型操作过的功能，尤其是个人文献库，使发版前的探索报告既守住"只能安全操作本插件界面"的边界，又能在更多真实交互上发现问题。

## Success criteria

- [x] 两端的键盘按键动作和点击一样受危险操作拦截：焦点落在外部链接、调度开关、清除运行状态、密钥显示等受限或敏感控件上，或焦点不在本产品界面内时，按键不发送；固定测试覆盖这些情形。
- [x] Obsidian 端截图与 workbench 一样遮住密钥字段；固定测试覆盖。
- [x] 每个目标任务依赖的界面文字由一项普通固定测试（不调用模型）核对仍存在于产品中；产品改名后该测试失败，并指出需要更新的任务和缺失的文字。
- [ ] 个人文献库在 Plugin product 和 Reading workbench 上各有一组带程序判定的目标任务（覆盖范围在对应阶段确定），起点由夹具确定性准备，不使用系统文件选择框，不读写真实文献库。
- [ ] 至少一次真实模型完整运行取得证据（运行前征得用户同意费用）；疑点逐条复核处置；发版执行说明更新任务数量与成本。

## Non-goals

- 为 Dashboard "More" 菜单的批量操作 "Retry failed dates"、"Run pending" 增加目标任务：单日重跑已由日历点击与 Force run 任务覆盖，批量选日期属于程序逻辑，更适合普通测试。
- 调整用户角色的步数预算，或新增程序侧的疑点自动发现机制。
- 让模型能使用 Obsidian 命令面板；仅能从命令面板触达的命令不在本次范围。
- 为 `scripts/` 新增常驻 lint / typecheck。
- 作为自动发版门槛；普通固定验收和 CI 仍不调用模型 API。
- 修改其他 Helm 计划的状态。

## Constraints

- 分支 `fix/exploration-safety-coverage`，worktree `/home/tiandc/Documents/code/arxiv-daily-exploration-coverage`。
- 复用 `scripts/acceptance/exploration*.mjs`、`exploration-tasks/*`、`scripts/desktop-acceptance/*` 现有的循环、动作校验、受限控件分类、夹具与重置；不复制平行实现。
- 普通固定验收和 CI 不调用模型 API、不读取真实密钥；文献库夹具只在临时目录中构造。
- Reading workbench 文献库任务允许测试工具在模型开始操作之前，把 PDF 运行时预装进测试专用缓存（首次需联网）；模型操作期间的请求仍全部走固定响应。
- 真实模型运行会产生费用，每次运行前先征得用户同意。
- 实现块尽量交给 subagent；owner 验收后按独立意图提交；不推送，推送与 PR 需用户逐次同意。
- 起因与证据：2026-10-13 审查报告（harness / task library / offline run），要点记录在 journal。

## Phases

<!-- Single source of truth for phase status. PN ↔ filename NN. Integer IDs are permanent and never reused; replacement phases take the next unused number. Outcomes only — no steps. The active line is the current focus. -->
1. P1 — 两端按键动作受与点击相同的受限/敏感/范围拦截，Obsidian 截图遮住密钥字段 — status: done
2. P2 — 一项普通固定测试核对所有目标任务依赖的界面文字仍存在于产品中，改名时指出需更新的任务 — status: done
3. P3 — Plugin product 上的个人文献库目标任务就绪，起点由夹具确定性准备 — status: done
4. P4 — Reading workbench 上的个人文献库目标任务就绪 — status: active
5. P5 — 真实模型完整运行、疑点复核与处置、发版说明更新 — status: pending
6. P6 — 界面文字检查按出处文件、区分大小写的完整字符串核对，同一文字在别处残留时改名也能发现（在 P3 之后、P4 之前执行） — status: done

## Open questions

- Obsidian 打开设置后键盘焦点不进入设置页，只用键盘的角色需先点击一次才能按键（P1 验证中发现）；P5 真实运行后再决定是否处理。
- 审查中"只要有一个非等待动作即可通过"的门槛是否可能借前一任务残留状态误判通过（未找到具体实例）；P2 或 P3 补任务时顺带确认，确认无实例则在 journal 记录后关闭。

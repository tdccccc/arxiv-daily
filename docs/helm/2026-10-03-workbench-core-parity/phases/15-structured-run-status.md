# P15 — Structured run status

goal_ref: ../goal.md
created: 2026-10-04T21:58:40+08:00
updated: 2026-10-04T21:58:40+08:00
revision: 1

## Outcome

Core明确区分尚未发布、确定无更新、筛选无匹配和真实失败；CLI/Obsidian/DSH消费结构化状态，不各自猜错误字符串。

## Assumptions

- 先读已有source/pipeline/scheduler/state/history合同，再确定向后兼容的具体状态字段。
- 尚未发布可稍后重试，但不能等同网络失败计数耗尽；无匹配论文仍与源无更新不同。
- 保留真实HTTP/解析错误、部分分类抓取失败和日期超出recent窗口的诊断。
- 不把日历周末规则当作所有来源无更新的证明；明确公告日期与用户时区边界。

## Chunks

1. Source结果合同：strict Red→Green。复现用户2026-10-03/newest2026-10-02案例；覆盖所有分类待发布、部分失败、空bucket、旧日期。
2. Pipeline/scheduler持久化：strict Red→Green。状态跨层传递、重试/计数/恢复语义和已有state兼容；不能误记完成或清除旧历史。
3. 各端适配：strict Red→Green。CLI输出、Obsidian状态、DSH日历与任务条显示同一含义；移除对新结果的错误文本猜测，旧记录只读兼容。

## Verification

- core源适配器/pipeline/scheduler/state/history定向测试，CLI与Obsidian相关回归，跨宿主合同，typecheck/boundaries/build，DSH隔离Host。
- 临时数据和模拟HTTP；不调用真实付费模型、发邮件或修改用户研究记录，不用Computer Use。

## Abort / reshape triggers

- 若需要破坏旧state枚举或迁移历史，先采用兼容字段/读取适配，不静默重写用户数据。
- 如果只是在前端把failed文案改成skipped，退回core状态语义设计。

# P15 — Structured run status

goal_ref: ../goal.md
created: 2026-10-04T21:58:40+08:00
updated: 2026-10-04T23:03:54+08:00
revision: 2

## Outcome

Core明确区分尚未发布、确定无更新、筛选无匹配和真实失败；CLI/Obsidian/DSH消费结构化状态，不各自猜错误字符串。

## Assumptions

- 先读已有source/pipeline/scheduler/state/history合同，再确定向后兼容的具体状态字段。
- 尚未发布可稍后重试，但不能等同网络失败计数耗尽；无匹配论文仍与源无更新不同。
- 保留真实HTTP/解析错误、部分分类抓取失败和日期超出recent窗口的诊断。
- 不把日历周末规则当作所有来源无更新的证明；明确公告日期与用户时区边界。

## Chunks

1. [x] Source结果合同：strict Red→Green。复现用户2026-10-03/newest2026-10-02案例；覆盖所有分类待发布、部分失败、空bucket、旧日期。
2. [x] Pipeline/scheduler持久化：strict Red→Green。状态跨层传递、重试/计数/恢复语义和已有state兼容；不能误记完成或清除旧历史。
3. [x] 各端适配：strict Red→Green。CLI输出、Obsidian状态、DSH日历与任务条显示同一含义；移除对新结果的错误文本猜测，旧记录只读兼容。

## Verification

- core源适配器/pipeline/scheduler/state/history定向测试，CLI与Obsidian相关回归，跨宿主合同，typecheck/boundaries/build，DSH隔离Host。
- 临时数据和模拟HTTP；不调用真实付费模型、发邮件或修改用户研究记录，不用Computer Use。

## Abort / reshape triggers

- 若需要破坏旧state枚举或迁移历史，先采用兼容字段/读取适配，不静默重写用户数据。
- 如果只是在前端把failed文案改成skipped，退回core状态语义设计。

## Accepted implementation and evidence

- Core uses additive `RunOutcome` and `failureAttempts` fields, retaining existing status enums/schema. Newer-than-latest announcements remain pending; explicit empty buckets and the arXiv weekend calendar finish as no_updates; filtered/ignored zero results are no_matches. Unexplained missing buckets, old dates and real category errors retain failure diagnostics.
- Waiting attempts preserve total attempts but do not consume the actual failure budget. Pending/completed terminal-write recovery does not rerun the source; history and reload preserve outcomes. Legacy entries use their existing attempts as the fallback budget and are not rewritten wholesale.
- L1 integration detail: workbench launches CLI as a child process. An in-process callback alone failed the actual packed DSH Host test (outcome undefined in both first-run/configured paths). Typed IPC envelopes now deliver results without parsing logs; both paths passed with papers_written, awaiting_announcement and no_updates assertions and persisted calendar checks.
- Workbench/Obsidian distinguish waiting/no updates/no matches. Core handles manual weekends; the old narrow weekend-string adapter remains read-only for legacy records. Saved reports and authoritative durable failures take precedence. Shared first-discovery readiness excludes no-update days.
- Observed Red→Green: source 3 failures, pipeline 3 plus weekend 1, scheduler/state 5 plus history rendering, CLI/HTTP 6, UI/calendar 5 plus stale-error cases, Obsidian calendar 4 and formatter 1, onboarding 2, packed Host 2. Added characterization checks cover partial-ready categories, malformed listings and missing in-window dates.
- Green: core focused 279 tests (11 files); CLI full 312 tests (36 files), followed by 75 settings/CLI/HTTP and 27 calendar/UI regression checks after final changes; Obsidian full 774 tests (46 files); DSH 20 tests, zero skipped, including actual isolated installed Host. Core/CLI/plugin typechecks, boundaries, product inventory and both builds pass.
- DSH0.1.13 is packaged for local linux/x64. Tests use temporary DSH_HOME/config/vault and HTTP/model fixtures. No real model calls, email delivery, user-vault writes or Computer Use. Electron visual testing, the entire core suite, and cross-platform native builds were not run. No production plugin installation was changed; Obsidian was built only.

Commits: core `c00cff9`; host adaptation `1c1f33b`. P2/P3/P4 remain separate pending work; this request covers P15 only.

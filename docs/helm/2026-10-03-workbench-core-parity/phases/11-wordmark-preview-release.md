# P11 — Wordmark and consolidated preview

goal_ref: ../goal.md
created: 2026-10-04T01:47:51+08:00
updated: 2026-10-04T01:51:51+08:00
revision: 2

## Outcome

左上角采用arxiv-daily纯文字标识，停止图标提案；把已确认模型/周末/标题修改统一打包为新预览版。

## Scope and verification

- 用户明确请求“改好了更新一版”；允许此次打包，先前暂不发版约束对此解除。
- Wordmark behavior: strict Red→Green，断言连字符标题且无旧图标/副标题。
- Typography proportionate check：系统sans字体、紧凑字距、加粗放大、弱化连字符，不引入外部字体。
- 全量CLI、DSH实际隔离Host、typecheck/boundaries/inventory/build/pack；不发送真实邮件/模型请求。

## Abort / reshape triggers

- 不启用任何待选图标、不影响用户当前会话/任务，不改变未授权运行设置。

## Acceptance

- 1edfed4: observed hyphenated-wordmark test Red, then full 32-file/270-test CLI suite Green. All 20 DSH tests pass, including isolated actual Host first-use/settings/read/generation/lifecycle flows.
- Typecheck/boundaries/product inventory/diff check/build/pack passed. Prepared dsh-arxiv-daily-0.1.9.tgz (linux/x64), then installed into existing Web profile via dsh plugin add. Verified installed version0.1.9 and bundle hash matches dist package.
- Existing DSH conversations/processes kept running; user must restart dsh web and refresh to use new in-memory assets. No new icon adopted. No live model or email tests or Computer Use.
- Installation scan had no findings in this project; broad global heuristics flagged unrelated tooling (WebSocket handshake base64/process.env and arXiv downloader), not this bundle. No global tools or hooks changed.

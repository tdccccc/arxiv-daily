# P7 — Secret reveal and model choices

goal_ref: ../goal.md
created: 2026-10-03T16:49:17+08:00
updated: 2026-10-03T17:03:11+08:00
revision: 2

## Outcome

用户点击Show能够查看刚输入或已保存的密钥，Get models后有明确可选模型列表。

## Assumptions

- 用户明确要求显示已存密钥；仅用户点击后通过受本机来源及revision保护的POST返回指定密钥，普通设置GET继续脱敏。
- 模型保留自由输入，同时提供显式select解决datalist不展开的问题。

## Chunks

- bug fix，strict Red→Green：HTTP显式show端点404/界面空值回归；校验field白名单、revision、Origin、no-store与普通GET不泄漏。
- bug fix，strict Red→Green：模型列表成功后select出现，可选择赋值，当前model不会被自动替换。失败提示在原行可见。

## Verification

- 设置UI/HTTP与已有设置回归，typecheck，build/pack，DSH tests。

## Abort / reshape triggers

- 不得把密钥加入常规GET状态、日志或持久浏览器缓存。

## Acceptance

- a4f9295: observed HTTP404 and empty displayed key Red, then missing explicit model selector Red; all corrected with focused UI and HTTP assertions.
- 49 tests across six relevant CLI files pass. All 20 DSH tests pass, including actual packed-host explicit key reveal and ordinary settings redaction. Typecheck, boundaries, inventory, diff check, build/pack pass.
- 0.1.7 linux/x64 archive generated; not installed into the user profile. No real model calls/email, no Computer Use or visual browser automation.
- Requested key reveal is intentionally available only by explicit POST with field allowlist, matching revision, existing Host/Origin/capability protection and no-store response. No secret values in status projections or logs.

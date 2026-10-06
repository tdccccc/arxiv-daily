# P9 — Weekend report state

goal_ref: ../goal.md
created: 2026-10-04T01:27:09+08:00
updated: 2026-10-04T01:28:18+08:00
revision: 2

## Outcome

工作台周末日报请求复用core日历规则显示跳过，不进入pipeline；旧的纯尚未发布周末错误在日历上显示无更新。

## Evidence and approach

Core自动tick跳过周末，runForDateNow手工日期入口刻意不做这一检查。工作台现有手工生成走后者。仅在workbench的日报入口复用core isWeekendDate，不修改CLI显式手工/force或core管线重试语义。

## Chunks

- bug fix / strict Red→Green：calendar旧周末缺bucket显示failed，API周末请求启动pipeline，先复现2条Red；用core日期helper提前返回skipped。
- UI回归：skipped中性标签；报告存在优先，历史真实网络/解析错误不掩盖；旧缺bucket兼容必须所有category都是相同尚未发布原因。

## Verification

- Calendar/API/读写/首次使用/UI相关测试，typecheck，boundaries/inventory。
- 不修改用户真实state，不生成安装包，等用户继续检查后统一版本。

## Abort / reshape triggers

- 不能把工作日所有抓取失败或混合来源失败笼统改成跳过。

## Acceptance

- Calendar legacy-weekend and API no-pipeline tests observed Red, then Green. UI skipped label regression passes; existing saved reports and genuine errors remain visible.
- 49 tests across calendar/server/onboarding/UI suites, CLI typecheck, boundaries and inventory pass. Existing failure/first-run tests now specify a weekday to avoid dependence on the real clock.
- No core scheduler semantic changes, no user-history writes, no build/pack/version increment. Pause further feature work while the user reviews the collected changes.

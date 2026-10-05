# P19 — Mainline acceptance

goal_ref: ../goal.md
created: 2026-10-05T12:16:11+08:00
updated: 2026-10-05T12:16:11+08:00
revision: 1

## Outcome

验证首版工作台/DSH可合入主线，更新README/安装/用户流程文档，并在主工作区干净且main未被外部推进时合并本地main。用户当前请求视为授权本地合并，不包含远程push/正式发布。

## Assumptions

- 主目录main初始a1cd713且干净；功能分支4d2b583且干净。先同步新增发布检查。
- 已验证核心日报/总结/阅读/设置闭环；P2/P3专用GUI及更完整任务管理仍后续，不宣称全产品完成。
- 文档必须区分已发布npm/Obsidian与源码可用的DSH本地包，不写不存在的安装命令。
- 保留主分支/功能分支备份；不丢弃其他worktree修改，不调用真实模型/邮件，不使用Computer Use。

## Chunks

1. 行为保持型集成：合入最新main、审查差异；使用既有Green基线与合并后完整检查。发现缺陷先复现Red再修。
2. 文档：中英文README、入门入口、DSH/CLI使用安装/限制；检查本地链接与命令对应实现，无需人为制造测试失败。
3. 验收：typecheck、lint、root/workspace测试、release tools、build/smoke与DSH/Claude打包检查。未能执行的环境门槛明确记录；不得把失败标成通过。
4. 验收通过后检查两工作区状态/main位置，备份并合并本地main；记录提交与未push/未发布状态。

## Abort / reshape triggers

主工作区出现其他人修改或main移动时重新核对，不自动stash/reset。真正合并阻断需要修复或明确说明；不要把文献库后续GUI当作已有功能。

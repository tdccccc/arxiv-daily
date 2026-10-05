# P19 — Mainline acceptance

goal_ref: ../goal.md
created: 2026-10-05T12:16:11+08:00
updated: 2026-10-05T12:29:28+08:00
revision: 2

## Outcome

验证首版工作台/DSH可合入主线，更新README/安装/用户流程文档，并在主工作区干净且main未被外部推进时合并本地main。用户当前请求视为授权本地合并，不包含远程push/正式发布。

## Assumptions

- 主目录main初始a1cd713且干净；功能分支4d2b583且干净。先同步新增发布检查。
- 已验证核心日报/总结/阅读/设置闭环；P2/P3专用GUI及更完整任务管理仍后续，不宣称全产品完成。
- 文档必须区分已发布npm/Obsidian与源码可用的DSH本地包，不写不存在的安装命令。
- 保留主分支/功能分支备份；不丢弃其他worktree修改，不调用真实模型/邮件，不使用Computer Use。

## Chunks

1. [x] 行为保持型集成：合入最新main、审查差异；使用既有Green基线与合并后完整检查。发现缺陷先复现Red再修。
2. [x] 文档：中英文README、入门入口、DSH/CLI使用安装/限制；检查本地链接与命令对应实现，无需人为制造测试失败。
3. [x] 验收：typecheck、lint、root/workspace测试、release tools、build/smoke与DSH/Claude打包检查。未能执行的环境门槛明确记录；不得把失败标成通过。
4. [x] 验收通过后检查两工作区状态/main位置，备份并合并本地main；记录提交与未push/未发布状态。

## Abort / reshape triggers

主工作区出现其他人修改或main移动时重新核对，不自动stash/reset。真正合并阻断需要修复或明确说明；不要把文献库后续GUI当作已有功能。

## Accepted result

- Merged origin/main a1cd713 into the feature branch before verification. Read-only review found no confirmed runtime/security/data-loss blocker in the inspected shared settings, library transactions, status, metrics, Markdown, history and DSH boundaries.
- A full release-tools gate exposed a real lockfile classification bug: apps/cli/node_modules/{commander,katex} were treated as unknown workspaces. Fixed only the checker, with scoped/deep nested dependency and unknown-workspace regression. Added tools/node-library-runtime/** to DSH push/PR path filters and product inventory after observed missing-path Red. Commits f7c8f2e and0977dea.
- Updated seven user docs: root English/Chinese README, English/Chinese getting-started, CLI, DSH and Claude integration README. Document current entry points, local-only DSH0.1.16 distribution, native build prerequisites, editable settings, topic directions, scientific reading, history, metadata and pending GUI scope.35 local links exist. Obsidian's persistent five-step guide semantics corrected. Docs commit d1ad169.
- Full root workspace run:3881 passed,2 skipped (real-corpus migration/retrieval opt-ins without user library samples); exit0. Release tools368 passed; DSH20 passed on installed0.2.0-rc.2; copied Claude package7 passed. All workspace typechecks, boundaries/inventory, root build, Obsidian submission, build smoke, offline native CLI install smoke and published-manifest check passed. npm audit reported0 vulnerabilities. Lint passed its configured gate with0 errors/22 warnings (threshold64); separate review-warning worktrees were not touched.
- No new Electron visual walkthrough or native cross-platform matrix. DSH CI retains its explicit0.1.7-alpha.1 baseline; local Host checks do not imply all versions are covered. No user corpus, paid model or email tests.
- Main checkout was still clean at a1cd713, and the feature commit was its descendant. Created backup/main-before-workbench-20261005 at a1cd713; fast-forwarded local main to d1ad169. Closing documentation is then fast-forwarded too. Feature worktree and backup/workbench-before-main-merge retained; no stash/reset or edits to other worktrees.
- No remote push, GitHub merge, version tag, npm publication or production plugin installation. Source/README are on local main; package0.1.16 remains a local artifact. Dedicated library/review GUI and fuller run management remain P2/P3/P4 pending, not acceptance blockers for this first workbench merge.

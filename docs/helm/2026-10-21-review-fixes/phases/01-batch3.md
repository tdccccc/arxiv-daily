# P1 — 第三批收尾

goal_ref: ../goal.md
created: 2026-10-23T22:19:51+08:00
updated: 2026-10-24T00:42:23+08:00
revision: 3

## Outcome

CLI、工作台、DSH、邮件和中继共 13 条修复经复核、全套检查及 CI 后合并。

## Assumptions

- origin/main 为 ebeec19，前两批已合入。
- 两个原 worktree 起点干净，往返测试实际已存在，交接文字需按代码复核。

## Approach

两个代理分别审阅 CLI 和邮件分支，协调者复核证据、集成、刷新索引、验证并完成 PR。

## Chunks

### Chunk 1 — CLI、工作台和宿主退出

- kind: bug fix
- change: 真实短密钥不泄露；保留 TOML 注释的依赖使用成熟版本；工作台首次报告与宿主退出符合预期。
- first signal: 短密钥回显测试失败；既有配置和 DSH 测试作为基线。
- checks: 针对性回归、gates.sh b3-cli-resume、DSH 构建及真实宿主 fixture 测试。
- [x] accepted on observed evidence

### Chunk 2 — 邮件恢复和中继

- kind: bug fix
- change: 补发邮件忠实读取已提交日报；失败提示、滑动续期及部署说明准确。
- first signal: 既有往返测试和中继测试基线；新增问题先失败回归。
- checks: 邮件和 pipeline 测试、gates.sh b3-mail-resume relay；只做部署 dry run。
- [x] accepted on observed evidence

### Chunk 3 — 集成与发布审查

- kind: non-behavioral
- change: 合并分支、更新问题索引，PR 通过检查并合入 main。
- first signal: n/a
- checks: 全套 gates.sh b3-resume relay vscode acceptance、DSH；54 条索引链接；远端 head、CI、threads 和 mergeable。
- [x] accepted on observed evidence

## Phase verification

- 全套本地检查、CI 全绿，review threads 全部解决，最新 main 已包含，PR 已合并。

## Abort / reshape triggers

- 若外部修改、配置/邮件持久化不一致或新增高风险行为，先停止集成并修正对应项。

## Observed evidence

- CLI 分支 24d3740 全套门禁通过；DSH 21/21，旧启动行为反向对照按预期失败。
- 邮件分支 2a33329 全套门禁通过；中继 227 测试通过。
- 集成 0e404fc 全套23项门禁通过：core 3104（2 skipped）、node-runtime 129、CLI 662、plugin 1220；release tools 927、acceptance tools 420、workbench 12/12；中继与 VS Code 检查通过，DSH 21/21。
- DSH 首次隔离运行因 loopback 未启用导致本机连接失败；启用隔离空间 loopback 后完整通过，外网保持隔离。
- 日志：/tmp/arxiv-fix-2026-10-21/gates-b3-integrated-resume/summary.txt、b3-integrated-dsh.log。
- 已核对原54条链接，状态36已修复/18未修复。远端main仍为ebeec19。PR/CI及合并待完成。

- 最终head a48d239：PR #108 的17项CI全部success；两条CodeQL线程已解决，分支无open扫描告警；main未变化，mergeStateStatus=CLEAN。PR已合并，提交e8b4838；第三批远端分支已删除。

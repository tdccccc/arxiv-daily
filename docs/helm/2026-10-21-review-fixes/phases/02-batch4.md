# P2 — 第四批剩余问题

goal_ref: ../goal.md
created: 2026-10-24T00:42:23+08:00
updated: 2026-10-24T01:08:30+08:00
revision: 2

## Outcome

原审查剩余18条经修复、归档、集成验证及CI后合入main。

## Assumptions

- 基线e8b4838为已合并第三批；四项额外发现继续独立跟踪。
- 不使用真实外部服务。真实CLI始终用隔离配置和unshare网络，loopback供夹具通信。
- 维护搜索索引必须尊重既有跨进程静默期约定，不能假设本进程空闲即全局安全。

## Approach

worktree a / b分别处理工具9条和核心产品6条；协调者在integration处理OBS-06、SCR-07、SCR-08并验收集成。每条独立提交、归档并记录进度；不得再派代理。

## Chunks

### Chunk 1 — 论文正文和日报可读性（PIPE-04/05/09）

- kind: bug fix
- change: 源码数学片段保真、清洗复杂度有界；不可信日报字段无法触发图片/嵌入，邮件往返保持可读。
- first signal: 数学源码/恶意长输入/Markdown图片等回归在旧实现失败。
- checks: source-extractor、日报渲染、digest往返及pipeline回归。
- [ ] accepted on observed evidence

### Chunk 2 — 索引进度与清理（LIB-06/07）

- kind: bug fix
- change: 中断索引复用已写入且匹配的论文文档；提供明确静默期清理入口，保留当前/备份/占用代，旧模型文件回收限制写明。
- first signal: 取消后重建重复嵌入；没有可调用清理入口。
- checks: 取消/失败/换模型/CAS冲突复用；多代清理和读者保护；CLI与插件入口回归。
- [ ] accepted on observed evidence

### Chunk 3 — 历史同步和真实设置输入（STATE-03/OBS-06）

- kind: bug fix
- change: 大小写不一致或整个目录暂不可见不清空索引；维度/旧版API密钥经事务保存，校验、回滚和索引状态刷新生效。
- first signal: 大小写目录清空索引、真实DOM输入绕过事务的用例失败。
- checks: 核心history-sync及插件路径测试；新旧设置页真实change/blur及保存失败测试。
- [ ] accepted on observed evidence

### Chunk 4 — 探索夹具与观察（SCR-02/04/05/09、EXP-04/05）

- kind: bug fix
- change: 全平台配置隔离；驱动符合现有提案契约；调度控件受限；PDF夹具可验证；活动笔记、焦点、被拦截外部打开可观察。
- first signal: 各缺口的确定性驱动/fixture测试失败。
- checks: acceptance-fixtures、runner、workbench、exploration-host/obsidian/labels；固定workbench验收。
- [ ] accepted on observed evidence

### Chunk 5 — 发版工具守卫（SCR-01/03/07/08）

- kind: bug fix（退休脚本删除为non-behavioral）
- change: 按宿主构建并记录产物证据；相对跨层导入不可绕过；删除退休诊断脚本入口；模型超时可配置并有限重试。
- first signal: 构建选择、相对导入、模型超时预算回归失败；删除脚本无需行为测试。
- checks: runner、build-deploy、boundaries、exploration预算测试及release-tools。
- [ ] accepted on observed evidence

### Chunk 6 — 依赖审计与公式渲染（SCR-06）

- kind: bug fix
- change: 6个lockfile动态纳入更新和审计；KaTeX升级消除已知低危公告，工作台公式和资源仍可用。
- first signal: 漏配lockfile检测失败；升级前保留公式渲染基线。
- checks: security-ci覆盖测试、所有lockfile audit；markdown/math回归、构建字体资源及真实浏览器公式显示。
- [ ] accepted on observed evidence

### Chunk 7 — 集成与合并

- kind: non-behavioral
- change: 54条索引全部有验收证据并归档；第四批通过检查、CI和审查后合并。
- first signal: n/a
- checks: gates.sh b4-final relay vscode acceptance及DSH；全部索引链接，CI/threads/main/mergeable；清理已合并worktree。
- [ ] accepted on observed evidence

## Phase verification

- 全套本地门禁及CI通过；review threads全解决；最新main已包含；PR确认合并。
- 原54条结果完整，额外4条不算作已修复；部署中继留给用户。

## Abort / reshape triggers

- 复用或清理破坏旧索引、设置事务失效、夹具可访问真实数据/服务时，停止对应集成，先补失败证据并修正。

## Observed evidence

- OBS-06 c607bb0：16个真实DOM回归先失败，完整设置278项、插件类型和lint通过；归档accc7d5。
- SCR-07 d2fa7fd：4个退役脚本移除，历史说明不再提供可运行入口；引用和diff检查通过，无需运行时测试；归档c06f0c9。
- SCR-08 5a4797b：3个新增回归先失败，探索/runner完整108项通过。
- 协调复核PIPE-09发现裸TeX环境仍可生成img，已把实际渲染器复现交回产品代理修正，尚未验收。

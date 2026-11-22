# 2026-11-22 独立复审发现的回归与后续修复

状态：已修复（分支 `fix/2026-11-22-review`，PR #180）

> 来源：对 PR #180（`fix/2026-11-22-review` vs `origin/main` `2557b90a`）的独立只读复审，基线 `43ea9ab5`
> 结果：9 组问题（1 阻断、2 高、4 中、2 低）全部修复；修复后的全 workspace typecheck、core / node-runtime / CLI / plugin 全量测试与 lint/边界/提交检查均通过

## 阻断

### R2-01 日报已提交但索引未提交时，重跑会永久丢失当天论文（PIPE-04 回归）

某次运行在 `writeDaily` 成功之后、`upsertManyFromDailyPapers` 之前取消或索引写入失败时，日报已落盘而 `papers.json` 没有当天条目。重跑进入“日报已存在”的修复路径，而 `repairExistingDaily` 只更新已存在的条目，`addDailyReports`/`setSummaries`/`reconcilePaperDetails` 对未知 id 一律跳过，于是当天被永久标记完成、论文再也不会进入索引。

修复：`PaperIndexStore.createMissingDailyPapers` 只补建缺失条目、绝不覆盖已有条目的用户状态；修复路径从已提交的日报 Markdown 构造最小条目（标题、作者、日期、主题、fallback 摘要、已验证精读路径），再执行原有回填。主路径的检查点清理移到索引提交与精读回链刷新之后；修复路径只有真正完成才清理检查点。修复时的精读回链优先复用索引中已有条目，保留论文更早的日报历史与首次发布日期。

提交：`7da16250`、`b41865a6`。测试：写完日报后立即取消 → 重跑补建条目并回链；只建缺失条目、已有状态/优先级/路径不被覆盖；重复修复不重复创建。

## 高

### R2-02 Get models / 旧版 Send test 仍与异步设置保存竞态（OBS-01 / OBS-04 回归）

设置输入在 blur/change 时异步持久化，`SettingsChangeService` 只有在持久化完成后才更新 `plugin.settings`。输入后直接点 Get models 会读到旧 key/URL；若保存随后完成，拉取结果还会因缓存 key 不匹配被丢弃。旧版 `<1.13` 的 Send test / Send verification 同样没有冲刷刚粘贴的密钥/收件人。

修复：设置页按 key 注册草稿保存句柄（`registerDraftSave`/`flushDraftSave`），Get models 在快照端点和构造客户端前等待 API key + Base URL 的保存；旧版 Send test 等待收件人与当前模式密钥，Send verification 等待收件人。Base URL 的保存与掩码输入一样合并去重。

提交：`13cbfddd`。测试：延迟持久化 + 真实 DOM 事件，覆盖声明式/旧版 Get models、旧版 Resend key、hosted token、收件人五条路径；去掉修复后全部失败。

### R2-03 跨重跑与无 paperIndex 时精读笔记日报回链仍缺失（PIPE-03 回归）

回链刷新只接收本轮新写的 `confirmedPaths`；第一次运行写完精读笔记后摘要失败，第二次运行不会重写该笔记，因此永远不会回填；未配置 `paperIndex` 时所有论文都没有 `indexEntry`，刷新直接跳过。

修复：主路径使用“本轮新写 + 已在磁盘验证”的合并路径集；缺少索引条目时由论文自身构造最小索引视图；修复路径同样回填回链。

提交：`ecf5da09`。测试：首轮写笔记后摘要失败、次轮成功仍回链；无 `paperIndex` 时用真实 MarkdownWriter 断言 `published` 为日报 wikilink。

## 中

- **R2-04 本地嵌入模型运行中被 dispose**：`dispose()` 与在途 `load()`/`embed()` 竞态会把 `ready` 反向置真并释放正在使用的 extractor；插件离开本地模式时立即释放。修复：只在仍是当前 load 时置 ready；被释放的 embed 抛出可重试错误；dispose 等待在途 embed 结束；插件把释放推迟到最后一个全文索引任务结束，卸载时先取消再释放（`bb4e60a9`）。
- **R2-05 嵌入配置变更不取消远程索引**：只有显式撤销授权会取消；切换模式/端点/模型/密钥时后台远程索引仍向外发送并可能提交不匹配的 manifest。修复：候选设置变更时同步取消 `personal-library-fulltext-index`；运行启动时快照模式与模型身份，提交前不一致则以取消结束（`69368a3a`）。
- **R2-06 自适应思考预算仍被 4096 锁死**：自适应 Claude 分支提前返回，`maxCompletionTokens: 4096` 时总 `max_tokens` 仍为 4096。修复：自适应路径同样在可见预算上叠加 effort 预算（`bdb41b2a`）。
- **R2-07 准备失败错误永不恢复 / 配置不完整暴露旧模型论文**：切回已就绪的本地索引后 `"Library preparation failed..."` 不在清除白名单；远程配置未填完时把旧本地论文当作当前索引。修复：命名并纳入校验通过后的清除条件；配置不完整时仅在同模型/同维度（或空 manifest）下暴露论文（`3f179186`）。

## 低

- **R2-08 Dashboard 过早重绘 / 超时提示残留**：首次索引扫描完成前，前台 recent-dates 刷新可能用空态覆盖 Loading；超时后的后台回调仍与超时前基线比较。修复：首次索引就绪前不重绘；超时提示渲染后更新比较基线（`566f9464`）。
- **R2-09 其余边界**：零来源论文分支补检查点清理（`f6552481`）；hosted 验证邮件取消原样透传（`3f48fbfc`）；存储 `stat` 权限/IO 错误不再视为文件不存在、无 stat 时目录回退 `rmdir`（`b09bdd5d`）；提案冲突重读后重算新论文计数并记录重读错误（`4252fa26`）；旧版重复自定义分类拒绝 + sidecar 死代码清理（`5aa21622`）。

## 验证

- `npm run typecheck`（全部 workspace）
- `npm test`（core / node-runtime / CLI / plugin；plugin 64 文件 1327 项）
- `npm run lint:obsidian`、`npm run lint:review`、`npm run check:boundaries`、`npm run check:obsidian-submission`

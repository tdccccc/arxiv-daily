# 旧版 data.json 中的 runState 迁移失败后，第一次保存设置就会把它永久删除

> 发现日期：2026-10-18 · 来源：main `21fe2ce` 深度代码审查 · 编号 P7
> 范围：插件设置持久化（从旧版本升级的用户） · 严重程度：低 · 状态：已修复（分支 `fix/2026-10-18-review`，提交 a38d2be） · 置信度：已复现

## 问题

从旧版本（运行状态保存在 data.json 里）升级的用户，启动时会把已完成/失败/跳过的历史复制到新的运行状态存储。如果复制失败（只读或磁盘满、输出路径错误、锁冲突），插件只记录一条"legacy run state was not migrated: ..."。紧接着的任意一次设置保存（切换开关、编辑主题、文献库操作）都会重写不含 `runState` 键的 data.json，永久销毁唯一的副本。结果：日历/历史显示没有任何完成记录，调度器把回溯期内每一天都重跑一遍，"首份报告完成"的引导也被重置。被 `isReadableEntry` 拒绝的条目同样只打警告，随后被从磁盘抹掉。

## 原因

- `plugin/src/services/legacy-run-state.ts:17-37` `migrateLegacyRunState` 吞掉 `replaceAll` 的错误并正常返回。
- `plugin/main.ts:441-446` 只把旧对象保存在内存中。
- `main.ts:2818-2833` `persistSettings()` 只用设置、`libraryProposalAcceptances`、`libraryConnection` 构建 PersistedData，没有透传未迁移的 `runState`（对比之下，解码失败的 `libraryProposalAcceptances` 会原样保留）。

## 可能的解决方案

推荐：在迁移成功之前一直携带旧数据——保留一份待迁移副本（含被判为不可读的条目），只要非空，`persistSettings` 就把它写回 `runState`；`migrateLegacyRunState` 返回成功与否，只有 `replaceAll` 成功后才清空。

备选：第一次重写前把 data.json 复制为 `data.json.legacy-runstate.bak`。

## 修复后应补的测试

- 迁移失败后保存设置，data.json 仍含 `runState`。
- 之后迁移成功，下一次保存移除该键。
- 不可读条目保留在待迁移副本中。

## 修复情况

迁移失败时 `migrateLegacyRunState` 返回旧数据，`persistSettings` 持续写回，直到迁移成功。

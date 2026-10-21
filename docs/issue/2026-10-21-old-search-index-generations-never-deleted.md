# 文献库每次重新建立索引都会在 Vault 里留下一整份新的搜索索引，旧的从不清理

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 LIB-07
> 范围：个人文献库搜索索引（"代"目录，插件、CLI 共用） · 严重程度：低 · 状态：未修复 · 置信度：已复现

## 问题

文献库有论文新增、删除或修改后重新建立索引，会生成一份新的不可变搜索索引"代"（generation），保存在 Vault 的 `.index/personal-library-search-index/<库范围>/<识别规则>/generations/` 下。旧的代在新的代提升为当前版本之后就不再使用，但没有任何生产代码会删除它们：清理入口 `FullTextGenerationIndexStore.runHostAuthorizedMaintenance()` 只被插件的 `maintainPersonalLibraryFullTextGenerations()`（`plugin/main.ts:2127`）包了一层，而这个方法除测试外没有调用方：没有命令、没有设置页按钮，CLI 和 Workbench 也没有对应命令。

结果是 Vault 里的索引目录随每次变更单调增长，而且它在 Vault 内，会被 Obsidian Sync、iCloud、Git 等同步。

复现：用 `LibraryWorkflow` 在内存存储上连续做 5 次"新增一篇论文后 scan + index"，磁盘上的代目录数依次为 1、2、3、4、5，字节数 84,675、180,013、286,074、403,109、531,092。夹具里每篇论文每代约 10.8 KB（向量维度 2；真实模型 384 维每篇再多约 1.5 KB），按此估算，1000 篇论文的库每次变更约增加 12 MB。该估算没有用真实库验证。

另外按代码推断，更换嵌入模型后，旧模型的逐篇文档目录（`papers/models/<命名空间>/`）也没有清理入口：`removePaper` 只删除当前已提交 manifest 对应路径，`removeAll` 没有生产调用方。

## 原因

- `packages/core/src/library/fulltext/generation-index-orchestration.ts:130-149`：只有 `sourceRevision` 与 manifest 完全一致且派生版本相同时才复用当前代，否则总是构建新的一代；构建不会回收旧代。
- `packages/core/src/library/fulltext/generation-index-store.ts:311-345`：回收只能通过"宿主授权的静默期维护"完成，设计上不自动执行（跨进程读者无法观察）。
- `plugin/main.ts:2127`：唯一的宿主封装没有入口；`apps/cli`、`extensions` 里没有任何调用。

## 可能的解决方案

推荐：给维护找一个安全的自动触发点，而不是增加用户手动步骤。插件在索引运行结束后、没有其它操作且调度器空闲时调用已有的维护封装（它已经要求调度器空闲并停止准入）；CLI 的 `library index` 在拿到 `personal-library-workflow` 锁、索引成功之后调用同样的维护。维护本身保留"当前代和备份代、有读者的代、有声明的代都不删"的规则，所以只会回收确定无用的旧代。

备选：保留手动入口，但在设置页的索引行或命令面板加一个"清理旧搜索索引"命令，并在 CLI 增加 `library maintain`；至少让用户有办法回收空间。

## 修复后应补的测试

- 连续多次变更并重建索引后（有维护触发），磁盘上最多保留当前代和备份代。
- 有打开的读者（正在搜索）时，维护不删除它所在的代。
- 更换嵌入模型并成功提交后，旧模型的逐篇文档目录被回收（或明确文档化为不回收）。

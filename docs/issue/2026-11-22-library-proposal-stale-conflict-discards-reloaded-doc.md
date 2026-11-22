# 文献库提案持久化冲突重读时丢弃结果导致内存状态过期，且切换库目录加载 catalog 失败未记录错误

状态：未修复

> 来源：2026-11-22 main `2557b90a` 代码审查（Obsidian 个人文献库状态管理）
> 严重程度：中 · 置信度：代码推断

## 问题与依据

`plugin/main.ts` 中有两处个人文献库持久化/重载状态不同步的问题：

1. **`mutatePersonalLibraryProposal()` 在版本冲突（stale）后重读磁盘提案却把结果丢弃了**：
   - `plugin/main.ts:2748-2753`：
     ```ts
     } catch (error) {
       if (this.isReviewPersistenceConflict(error)) {
         await this.loadPersonalLibraryProposalDocumentDirect(guard, stores);
       }
       throw error;
     }
     ```
   - `loadPersonalLibraryProposalDocumentDirect()` 只负责从存储读取并返回 `PersonalLibraryDirectionProposal | null`，**并不修改 `this.libraryProposal`**（对比 `generatePersonalLibraryDirections` 在第 1358 行通过 `reloadPersonalLibraryProfileDocuments()` 更新内存字段）。
   - 结果：当外部客户端（CLI / Workbench）更新了提案文件、插件内保存编辑/重命名/决策触发 stale 冲突时，第 2750 行虽然多做了一次磁盘 I/O 读出了最新提案，但返回值直接被丢弃，`this.libraryProposal` 仍停留在旧的过期版本。UI 刷新快照时仍看到旧 revision，用户再次操作会继续冲突，必须手动重新打开或重启。
2. **`selectLibraryRoot()` 加载 catalog 失败时未设置 `this.libraryCatalogLoadError`**：
   - `plugin/main.ts:699-702`：切换文献库根目录并加载持久化 catalog 失败时，`catch` 块只打印了日志并置空 `this.libraryCatalog = null`，遗漏了赋值 `this.libraryCatalogLoadError = this.safeProfileLoadError("catalog", error)`（对比其他加载路径均会设置 `libraryCatalogLoadError` 以在界面提示加载损坏而非“尚未扫描”）。

## 原因

- `mutatePersonalLibraryProposal` 误以为 `loadPersonalLibraryProposalDocumentDirect` 会顺带更新实例成员 `this.libraryProposal`。
- `selectLibraryRoot` 的异常处理未走统一的 profile load error 记录路径。

## 建议和验收

- 在 `plugin/main.ts:2749-2751`：捕获持久化冲突时改为调用 `await this.reloadPersonalLibraryProfileDocuments(guard, stores)`，或将 `loadPersonalLibraryProposalDocumentDirect(guard, stores)` 的结果在校验 `this.assertPersonalLibraryReviewGuard(guard)` 后写回 `this.libraryProposal` 并清空 `this.libraryProposalLoadError`。
- 在 `plugin/main.ts:699-702`：`catch (error)` 中补充记录 `this.libraryCatalogLoadError = this.safeProfileLoadError("catalog", error)`。
- 验收：在磁盘上并发推进提案 revision 后在插件内执行一次提案修改（如重命名主题），抛出冲突后 `plugin.getPersonalLibraryProfileSnapshot().proposal` 立即反映磁盘上的新版本。

## 修复后应补的测试

- `plugin/tests/personal-library-interest-profile-modal.test.ts` 或 `plugin/tests/library-connection-lifecycle.test.ts`：
  - 模拟外部将提案文件推进到新 revision 后调用 `renamePersonalLibraryProposedTopic()`，断言抛出冲突错误后 `plugin.getPersonalLibraryProfileSnapshot().proposal.revision` 已更新为磁盘最新 revision；
  - 模拟 `selectLibraryRoot()` 指向包含损坏 catalog 的库，断言 `getPersonalLibraryProfileSnapshot().catalogLoadError` 非空。

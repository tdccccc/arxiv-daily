# ObsidianStorageAdapter.remove 仅支持存在的文件，导致 Obsidian 下旧搜索索引清理必然失败且运行历史超 512 KB 后永久停写

状态：已修复（分支 `fix/2026-11-22-review`）

> 来源：2026-11-22 main `2557b90a` 代码审查（Obsidian 存储适配器 + core 索引维护与运行历史）
> 严重程度：高 · 置信度：已复现（核心路径按 Obsidian `DataAdapter.remove` 语义离线复现）

## 问题与依据

`StorageAdapter.remove(path)`（`packages/core/src/core/adapters.ts:147`）在两个宿主实现上语义完全不一致：

- Node 宿主 `packages/node-runtime/src/storage-adapter.ts:501-503`：调用 `fs.rm(path, { recursive: true, force: true })`（**允许路径不存在，且可递归删除非空目录**）。
- Obsidian 宿主 `plugin/src/hosts/obsidian/storage-adapter.ts:148-150`：直接调用 `await this.vault.adapter.remove(this.normalizePath(path))`。根据 Obsidian 官方类型定义（`node_modules/obsidian/obsidian.d.ts:3333-3344`），`DataAdapter.remove` 是**仅删除单个文件**的接口（删除目录需用 `DataAdapter.rmdir(path, true)`），在桌面端底层对缺失文件抛 `ENOENT`、对目录抛 `EISDIR`/`EPERM`。连 `ObsidianStorageAdapter.writeTextAtomic` 自己（`storage-adapter.ts:103,107,113,117`）都不得不给 `vault.adapter.remove()` 加上 `.catch(() => undefined)`。

核心层有两处生产代码依赖了 `NodeStorageAdapter.remove` 的宽容/递归语义，因此在 Obsidian 桌面端全部静默失效：

1. **“清理旧个人文献库搜索索引”命令永远清不掉任何旧代目录**：
   - `packages/core/src/library/fulltext/generation-index-store.ts:442-452`（`runGenerationMaintenance`）：直接 `await this.storage.remove(generation.directory)`，而 `generation.directory` 是包含 `descriptor.json` 和 `objects/*.bin` 的非空目录。
   - 失败被第 449-452 行捕获并记为 `retained.push({ generationId, reason: "remove-failed" })`。用户在 Obsidian 运行命令 `Clean up old personal library search indexes…`（`plugin/src/commands.ts:525`）时，虽然流程“成功结束”，但永远只提示移除了 0 个旧索引，Vault 中旧代目录持续堆积。
2. **运行历史 `run-history.jsonl` 超过 512 KB 触发轮转后永久停止记录**：
   - `packages/core/src/services/run-history.ts:290-312`（`rotateHistoryFiles`）：第 299 行无条件调用 `await storage.remove(\`${path}.${maxRotations}\`)`（以及第 304、309 行对目标槽位无条件 `remove`），未检查 `storage.exists(...)`。
   - 首次触发 512 KB 轮转时，`run-history.jsonl.3` 必然不存在，在 Obsidian 下立即抛 `ENOENT`；上层 `safeAppend` 将其吞为 warning，文件既没轮转也没截断，此后**每一次**运行记录追加都再次超过 512 KB、再次轮转、再次抛 `ENOENT`，运行历史日志在 Obsidian 中永久停写。

## 原因

1. `StorageAdapter.remove` 接口未明确约定是否幂等忽略不存在的路径、以及是否支持删除目录。
2. 测试替身掩盖了宿主差异：`plugin/tests/obsidian-adapters.test.ts:90-92` 的 `realFilesystemAdapter` 假实现把 `remove` 写成了 `fs.rm(..., { recursive: true, force: true })`，与真实 Obsidian `DataAdapter.remove` 行为相反；`packages/core/tests/run-history.test.ts:38-42` 的内存存储也对不存在路径静默成功。

## 建议和验收

- 明确 `StorageAdapter.remove(path)` 契约（并在接口注释中写明），同时做双向防御：
  1. 在 `ObsidianStorageAdapter.remove(path)` 中：先检查是否存在（或忽略 `ENOENT`）；若目标是目录（可通过 `stat` / `list` 探测），使用 `this.vault.adapter.rmdir(normalized, true)` 递归删除，文件则调用 `this.vault.adapter.remove(normalized)`。
  2. 在 `packages/core/src/services/run-history.ts:290-312` 的 `rotateHistoryFiles` 中，每次 `storage.remove(...)` 前先检查 `await storage.exists(...)`（与 `paper-index.ts`、`state-store.ts` 等其他 store 保持一致）。
  3. 在 `packages/core/src/library/fulltext/generation-index-store.ts:442-452` 删除孤儿代目录时，考虑复用类似 `knowledge-base-store.ts` 的递归删子文件再删目录逻辑，避免任何严格文件级 `StorageAdapter` 踩坑。
- 验收：
  - 使用严格贴近 Obsidian `DataAdapter`（`remove` 仅删存在的文件、`rmdir(p, true)` 删目录）的适配器测试：`RunHistoryStore` 跨 512 KB 多次轮转正常保留 `.1`~`.3` 且持续追加；`FullTextGenerationIndexStore.runHostAuthorizedMaintenance()` 成功删除非空旧代目录并返回对应 `removedGenerationIds`。

## 修复后应补的测试

- `plugin/tests/obsidian-adapters.test.ts`：修正/新增严格模拟 Obsidian `DataAdapter`（`remove` 对目录抛 `EISDIR`、对缺失文件抛 `ENOENT`，另提供 `rmdir(path, recursive)`）的测试，验证 `ObsidianStorageAdapter.remove` 能删除非空目录且对不存在路径幂等返回。
- `packages/core/tests/run-history.test.ts`：使用对缺失路径抛 `ENOENT` 的 `StorageAdapter` 验证 `RunHistoryStore` 首次与连续轮转均成功。
- `packages/core/tests/fulltext-generation-index-store.test.ts`：使用文件级 `remove` + 目录 `rmdir`（或通过 `ObsidianStorageAdapter`）验证 `runHostAuthorizedMaintenance()` 能彻底清理含 `descriptor.json` 和 `objects/` 的非空旧代目录。

## 修复情况

`StorageAdapter.remove` 契约明确为递归且幂等；Obsidian 适配器先 `stat`，目录走 `rmdir(path, true)`，缺失路径与 stat/remove 之间的 ENOENT 竞态均静默返回。`rotateHistoryFiles` 在每次 remove 前检查 `exists`；generation 维护改为自底向上删除子树。测试替身改为严格 `DataAdapter` 语义（`remove` 对目录抛 `EISDIR`、对缺失文件抛 `ENOENT`，另提供 `rmdir`），新增目录删除、幂等删除、连续轮转与非空旧代目录清理用例。验证：`obsidian-adapters` 31、`run-history` 22、`fulltext-generation-index-store` 79 项通过。

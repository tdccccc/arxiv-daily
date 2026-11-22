# Reading Dashboard 三处交互与状态缺陷：未连接文献库打开相似论文报错、Show All 未同步页签状态、搜索失败永久缓存

状态：已修复（分支 `fix/2026-11-22-review`）

> 来源：2026-11-22 main `2557b90a` 代码审查（Obsidian Reading Dashboard）
> 严重程度：中 · 置信度：代码推断

## 问题与依据

`plugin/src/dashboard/view.ts` 中存在三处直接影响阅读面板交互的状态 bug：

### 1. 未连接个人文献库时点击“相似论文”默认选中 Library 页签并报错
- `plugin/src/dashboard/view.ts:2374-2398`（`buildLibrarySimilarOption`）只判断论文标题/摘要是否为空，**没有检查 `this.plugin.getLibraryConnectionStatus().kind !== "disconnected"`**（也没有检查索引是否已就绪）。
- 而 `plugin/src/dashboard/similar-papers-modal.ts:111` 在收到 `options.library` 时会默认选中 `"library"` 页签并立即执行 `load()`。
- 结果：未连接个人文献库的用户（文献库是可选功能）每次在论文行点击相似论文按钮，弹窗都会先默认打开 `Library similar` 页签并报错 `"Library full-text search unavailable: Choose a personal library first"`，必须手动切到 `Daily similar` 才能看日报内相似论文。主搜索框同类问题曾在 `docs/issue/archive/2026-10-18-dashboard-library-search-error-without-library.md` 修复，但相似论文弹窗漏改。

### 2. Starred 空状态点击 "Show All" 按钮未同步页签高亮、计数和分页
- `plugin/src/dashboard/view.ts:724-727`：
  ```ts
  this.createEmptyActionButton(actions, "list", "Show All", () => {
    this.query = { ...this.query, tab: "all" };
    this.renderCurrentResults();
  });
  ```
- 对比正常的页签切换回调（`view.ts:872-878`）：这里漏掉了 `this.currentPage = 0`、`this.updateTabButtonState(tabs)` 和 `this.updateToolbarFilterCounts(tabs)`。
- 结果：点击 "Show All" 后虽然列表显示了全部论文，但顶部页签栏依然高亮停留在 `Starred`、工具栏筛选计数未按 `All` 更新，若之前停留在非第一页还会落在错误的页码上。

### 3. 文献库搜索的瞬时错误被永久写入 `librarySearchCache`，点刷新也无法重试
- `plugin/src/dashboard/view.ts:2442-2478`（`refreshLibrarySearch`）：
  - 当 `searchPersonalLibraryFullText()` 失败（如远程嵌入 API 临时超时/503 或断网），第 2470 行生成 `{ kind: "error", message }`，第 2475 行**无条件**将其写入 `this.librarySearchCache = { key, state }`。
  - 缓存键 `key` 仅包含 `[query, index.lastRun.updatedAt, index.lastRun.papers, embedding.mode]`（第 2437 行）。
  - 此后即使用户恢复网络、点击 Dashboard 刷新（`reloadIndex()`）或切页再切回，第 2442 行都直接命中错误的缓存结果并原样渲染错误块，完全不会重新发起搜索，必须改动查询文本或关闭重开 Dashboard 才能重试。

## 原因

1. `buildLibrarySimilarOption` 遗漏了与 `refreshLibrarySearch`（`view.ts:2423`）一致的文献库连接状态守卫。
2. 空态快捷按钮直接修改 `this.query.tab` 而未复用页签切换 helper。
3. 文献库搜索缓存未区分可复用的成功/空结果与瞬时失败态。

## 建议和验收

- `buildLibrarySimilarOption()`：当 `this.plugin.getLibraryConnectionStatus().kind === "disconnected"` 时返回 `undefined`（使 `SimilarPapersModal` 直接展示 `Daily similar` 列表，不渲染无效的 `Library similar` 错误页签）；若已连接但未建立索引，可按主搜索框约定展示中性提示而非默认报错，或默认选中 `Daily similar`。
- Starred 空态的 "Show All" 按钮复用统一的切页方法（重置 `this.currentPage = 0`，更新页签 `aria-pressed`/激活样式和筛选计数后再 `renderCurrentResults()`）。
- `refreshLibrarySearch()`：当 `state.kind === "error"` 时**不要**写入 `this.librarySearchCache`（或至少在 `reloadIndex()` 显式刷新时清空错误缓存，并在错误块提供重试路径）。
- 验收：
  - 未连接文献库时打开相似论文弹窗，无错误提示、直接展示 `Daily similar`；
  - Starred 为空时点 "Show All"，`All` 页签高亮、计数正确、`currentPage` 归零；
  - 远程嵌入搜索首次超时报错后，恢复网络点击 Dashboard Refresh 能重新发起搜索并显示结果。

## 修复后应补的测试

- `plugin/tests/dashboard-view.test.ts` / `plugin/tests/dashboard-library-search.test.ts`：
  - 未连接文献库时触发相似论文操作，断言传给 `SimilarPapersModal` 的 `library` 为 `undefined`、不调用 `searchPersonalLibraryFullText`。
  - Starred 为空时点击 "Show All"，断言 `All` 页签处于激活状态（`aria-selected` / active class）、`Starred` 取消激活、分页归零。
  - 模拟 `searchPersonalLibraryFullText` 第 1 次 reject、第 2 次 resolve，验证报错后重新刷新 Dashboard 会再次发起查询并展示匹配结果。

## 修复情况

未连接文献库时不再向相似论文弹窗提供 Library 选项，默认展示 Daily similar；Starred 空态的 **Show All** 复用统一 `switchTab`（重置分页、页签高亮与筛选计数）；文献库搜索的错误状态不再写入缓存，刷新或重新渲染会重新发起查询。测试：`dashboard-view` + `dashboard-library-search` 73 项通过。

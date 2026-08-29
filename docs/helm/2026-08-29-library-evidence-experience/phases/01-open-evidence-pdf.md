# P1 — open-evidence-pdf

goal_ref: ../goal.md
updated: 2026-08-29

## Outcome

用户在 Dashboard 或 Similar papers 点击「Open PDF at page N」时，Vault 内的文献 PDF 打开到该页；打不开时通知带具体原因。

## Assumptions

- 失败发生在插件证据打开路径，不是 Obsidian 的 `#page=N`。隔离会话里 `openLinkText("small_library/1706.03762.pdf#page=4")` 能落到第 4 页，同一 PDF 走 `openPersonalLibraryFullTextEvidence` 抛 `Cannot read properties of undefined (reading 'basePath')`。
- 根因是 `desktopVaultRoot` 把 `adapter.getBasePath` 抽成无绑定函数再调用，FileSystemAdapter 内部读 `this.basePath` 时 `this` 丢失。现有 lifecycle 测试用 `getBasePath: () => "/vault"`，不依赖 `this`，所以绿了。
- 打开前的一字节 `readBinary` 对 2.11 MiB 的 Attention PDF 不是这次失败原因。

## Approach

先写会失败的测试：mock adapter 的 `getBasePath` 必须通过 `this.basePath` 返回路径。再改为以方法调用（或 `call(adapter)`）读取 vault root，并让失败通知带上 `error.message`。用同一隔离会话复验插件打开路径。

## Test strategy

- change kind: bug fix
- strategy: strict Red–Green–Refactor
- Red / baseline signal: method-style `getBasePath` mock 在修复前抛 `basePath` TypeError；现有 arrow-function mock 仍绿
- Green / regression checks: 该测试与 `library-pdf-opener` / fulltext lifecycle 通过；隔离 Obsidian 上 `openPersonalLibraryFullTextEvidence` 打开 Attention PDF 第 4 页，对照第 2 页
- exception: 无

## Tasks

- [x] 用依赖 `this` 的 FileSystemAdapter mock 复现 `desktopVaultRoot` / 证据打开的 `basePath` TypeError。
- [x] 以绑定 `this` 的方式读取 vault root，使 Vault 内目标解析为 `small_library/<file>#page=N`。
- [x] Dashboard 与命令入口的失败通知带上具体错误信息。
- [x] 隔离 Obsidian 复验插件打开路径（不是裸 `openLinkText`）。

## Verification

- Red：method-style mock 抛 `Cannot read properties of undefined (reading 'basePath')`。
- Green：同一 mock 返回 vault root；lifecycle 仍把 vault 内路径解析为相对 `#page=N`。
- 桌面：插件打开 Attention PDF 第 4 页报 4，第 2 页报 2。
- 通知：失败文案含原始 `error.message`。
- Observed Red (2026-08-29): method-style `getBasePath` mock 抛 `TypeError: Cannot read properties of undefined (reading 'basePath')` at `desktopVaultRoot`；arrow-function mock 仍绿。
- Observed Green: 同一测试与 32 项相邻 plugin 测试通过。隔离 Obsidian（`/tmp/arxiv-daily-open-vault`，selectedRoot 在 vault 内）上 `openPersonalLibraryFullTextEvidence` 打开 `1706.03762.pdf` 第 4 页报 4、第 2 页报 2；`outcome=page-targeted`；0 renderer 错误。

## Abort / reshape triggers

- 如果绑定 `this` 之后真实 Obsidian 仍不能打开，停止并改查 `readBinary` 预检或路径映射，而不是改 UI。
- 如果失败变成「打开了文件但页码不对」，那是宿主 fragment 问题，保留页码降级，不要假装高亮成功。

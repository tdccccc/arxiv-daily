# 扫描时 arXiv 标题检索临时失败（限流、503、断网）后，这份 PDF 被永久记为"未识别"，文件不变就再也不会重试

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 LIB-04
> 范围：个人文献库扫描与重新扫描（插件、CLI、Workbench 共用的 core 逻辑） · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-21-batch1`，提交 c6202fd、cd8aaeb） · 置信度：已复现

## 问题

文件名里没有 arXiv 编号的 PDF，扫描时会读 PDF 标题并向 arXiv 按标题检索。这一步的任何失败（HTTP 429、503、超时、离线）都被当成"没有匹配"，该文件被记为 `unresolved`，并且带着当前识别规则版本。之后只要文件的大小和修改时间没变，每次重新扫描都直接复用这条记录，不会再问 arXiv。

只有两种办法能让它重新识别：修改（或 touch）这个文件，或者代码升级识别规则版本。用户在设置页点"Retry preparation"、在 CLI 里重复 `library scan` 都不会重试。与此相对，元数据请求失败（已知编号、取元数据失败）会记成 `failed`，下次扫描会重试；标题检索没有这个待遇。

影响：这份论文只能作为"仅本地标题和摘要"的条目参与检索和方向提案，没有 arXiv 的分类、作者、发表日期（方向预览的分类覆盖判断显示"未知"），用户也不知道为什么，因为界面把"标题检索失败"和"这篇论文确实不在 arXiv 上"混为一谈（`plugin/src/library/index-completion.ts:20-24` 的注释自己承认了这一点）。

触发可能性：识别循环逐个文件、没有间隔地向 `export.arxiv.org` 发标题检索，并且不经过 `ArxivFetcher` 的节流和 429 冷却（后者对同一个主机设了请求间隔和 10 分钟冷却，说明该主机限流是实际存在的）。按代码推断，数百个未识别 PDF 的首次扫描很可能中途撞上限流；本次没有联网验证限流行为。离线首次扫描则必然全部落入这一状态。

复现（桩 HTTP）：同一份未改动的 PDF，第一次扫描时标题检索返回 503，文件记为 `unresolved`；第二次扫描时 arXiv 已恢复，结果仍是 `unresolved`，`reusedFileCount` 为 1。

## 原因

- `packages/core/src/library/pdf-library-file-identifier.ts:34-46`：标题检索的异常被吞掉（只检查取消），随后返回 `directId` 或 `null`；"确定没有匹配"和"这次没问成"无法区分。
- `packages/core/src/library/personal-library-reconciliation.ts:211-234`：`identify` 返回 `null` 就写 `unresolved`，并带上 `contentIdentificationVersion`。
- `packages/core/src/library/personal-library-reconciliation.ts:188-197`：复用条件是"指纹相同、状态不是 `failed`、识别版本相同"，`unresolved` 不会重试。`detectPersonalLibraryFolderChange` 对这类文件同样视为"未变化"，所以索引前的自动检查也不会触发重扫。

## 可能的解决方案

推荐：让识别器区分两种结果。标题检索遇到传输错误或 HTTP 错误时返回"未能判断"，reconcile 把它记为 `unresolved` 但不写 `contentIdentificationVersion`。复用条件要求版本相同，没有版本的记录下次扫描会重新识别；不需要改目录 schema，也不影响老版本读取。同时把标题检索接到与 `ArxivFetcher` 相同的节流（请求间隔、429 冷却）上，减少自己制造的限流。

备选：新增 `failed` 原因 `identification-lookup-failed`，行为与 `metadata-fetch-failed` 一致；但目录解码器按原因枚举严格校验，新旧版本交替使用时旧版本会把目录判为无效，需要额外的兼容处理。

## 修复后应补的测试

- `reconcilePersonalLibraryCatalog`：第一次标题检索返回 503 -> 第二次（同一未改动文件）重新调用识别器并得到 `ready`。
- 标题检索明确无匹配 -> 第二次复用记录，不再调用识别器（`reusedFileCount` 为 1）。
- 取消仍然向外传播，不会被记成临时失败。
- 识别器遇到 429 时遵守冷却，不再对同一主机连续发请求。

## 修复情况

扫描时 arXiv 标题检索没能完成（HTTP 429/503、超时、离线）的 PDF，不再被永久记成"未识别"：对账把它记为 `unresolved` 但不写 `contentIdentificationVersion`，下一次扫描会重新识别（目录 schema 不变）；只有 arXiv 明确回答"没有匹配"才会被记住。标题检索改走与 `ArxivFetcher` 相同的请求协调器：请求之间至少间隔 3 秒，429 让 `export.arxiv.org` 进入冷却，冷却期内不再发请求；同一次扫描里连续 3 次检索失败后停止检索，避免离线扫描被间隔拖慢。证据版本升到 4 后重新识别时若暂时无法判断，未变化的文件保留原有识别结果。代价：首次扫描大量没有编号的 PDF 会变慢，每个需要标题检索的文件约 3 秒。

测试：`personal-library-reconciliation.test.ts`（失败不被记住、明确无匹配被记住、逐文件独立、取消向外传播、重新识别失败时保留旧结果）、`pdf-library-file-identifier.test.ts`（失败时抛出、429 冷却、请求间隔、连续失败后暂停、取消）、`arxiv-title-search.test.ts`（HTTP 错误类型、间隔、冷却）、`library-workflow.test.ts`（503 之后下一次扫描变为 `ready`，之后不再询问）。

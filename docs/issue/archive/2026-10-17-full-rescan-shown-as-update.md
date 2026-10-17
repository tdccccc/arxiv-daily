# 全量重新分析被误当成增量更新，审阅页显示错误的"本次新增"和"已覆盖的新论文"

> 发现日期：2026-10-17 · 来源：main `e6eaed5`（#93–#95 合并后）代码审查
> 范围：插件文献库方向审阅页、CLI Workbench 文献库审阅页 · 严重程度：中低（界面数字错误，不影响提案内容、方向接受和日报） · 状态：未修复 · 置信度：已复现

## 问题

已有方向建议时再做一次全量分析，例如插件里的 "Generate again"、Workbench 的"重新生成候选"、CLI 的 `library propose`。之后打开审阅页会看到：

- 插件 "Library overview" 页第一行显示 "N papers analyzed on <日期>. M added in this update."。这里 M 等于本次全量分析的全部输入篇数，N 是历次分析"看过"的论文数，不是本次分析的篇数。
- 插件 "Topics from your library" 页显示 "New papers already covered by your existing directions: K"。K 是本次输入里已被现有方向覆盖的论文数。
- Workbench 概览显示"分析了 N 篇论文 · 本次新增 M 篇"，候选视图显示"其中 K 篇新论文已被你现有的方向覆盖"。

全量分析没有"新论文"或"本次新增"可言，这些数字都是错的。`countNewPapersCoveredByExistingDirections` 自己的注释（`packages/core/src/library/personal-library-change.ts:90-96`）也写明：全量扫描时它应返回 0。

触发条件，满足任一即可：

1. 文献库里至少有一对同名文件被扫描合并。例如同一篇论文的两个不同 PDF，或者一份能识别出 arXiv 编号的文件加一份只能按标题识别的本地副本。
2. 上一份提案看过的论文中，有仍是 ready 状态、却不在本次全量分析输入里的论文。例如库里 ready 论文超过单次分析上限（1000 篇），最旧的被挤出窗口。按代码推断，ready 但没有分块的论文也会触发，这一情形未单独复现。

CLI 的 `library directions` 只输出 `newPapersSinceSuggestions`，不显示这些文字，不受影响。

影响：只影响审阅页上的提示数字和"分析了多少篇"的说法。新论文计数、新论文提醒和增量更新实际发送的论文都不受影响。但用户会误以为刚做的是增量更新，或者以为有一批"新论文"已被现有方向覆盖，从而误判文献库状态。

## 原因

三处界面各自用"已统计的论文比本次输入多"来推断"这是一次增量更新"：

- `packages/core/src/library/personal-library-change.ts:101`：`countNewPapersCoveredByExistingDirections` 里的 `if (accountedPaperKeysOf(proposal).length <= analysed.size) return 0;`。插件在 `plugin/src/library/interest-profile-modal.ts:987` 调用它，Workbench 在 `apps/cli/src/workbench/web/library-review.ts:136` 调用它。
- `plugin/src/library/interest-profile-modal.ts:537`：`const added = proposal.catalogInputPapers.length < accountedKeys.size ? proposal.catalogInputPapers.length : 0;`。第 538 行把它渲染成 "M added in this update."，并用 `accountedKeys.size` 作为 "papers analyzed"。
- `apps/cli/src/workbench/web/library-review.ts:201`：`analyzed.size < accounted.size ? … t("本次新增 {0} 篇", analyzed.size) …`。

全量分析同样会让已统计集合大于本次输入。`proposeClusteredPersonalLibraryDirections` 在 `packages/core/src/library/personal-library-direction-proposer.ts:564-568` 调用 `mergeAccountedPapers`（同文件第 676-693 行）。后者把扫描时合并掉的同名文件（`mergedKeys`）和上一份提案看过、仍为 ready 的论文（`previous`）都并入 `accountedPaperKeys`。这是 ADR 0015 第 3 节有意的设计，用来避免这些论文在全量分析后又被当成"新论文"。所以"已统计多于输入"区分不了全量和增量。

提案里没有字段记录这次是全量还是增量。唯一能区分两者的是 `generationContractFingerprint`：

- 全量用 `createPersonalLibraryClusteredDirectionGenerationContract`，契约里是 `scope: "full-library"`（`personal-library-direction-proposer.ts:401-409`）。
- 增量用 `createPersonalLibraryNewPapersDirectionGenerationContract`，契约里是 `scope: "new-papers"`（同文件第 416-422 行）。

两者的契约字符串不同，指纹也不同。

## 复现

用临时测试复现过，跑完已删除，工作区没有留下改动。做法是复制 `packages/core/tests/library-new-papers-proposal.test.ts` 里的全量扫描用例 "includes the ungrouped paper and the same-title duplicate it merged away, and keeps inherited papers"，在原有断言后打印几个数值。这个用例构造了一对同名文件和一篇继承论文，并用一个把所有分组都判为已覆盖的模拟模型。

| 项目 | 数值 |
| --- | --- |
| 本次全量分析输入（`catalogInputPapers`） | 8 |
| 已统计论文（`accountedPaperKeys`） | 10 |
| `countNewPapersCoveredByExistingDirections` 的返回值，即界面上的"已覆盖的新论文" | 7 |
| 按插件第 537 行、Workbench 第 201 行算出的"本次新增" | 8 |

该用例本身已断言：`accountedPaperKeys` 包含合并掉的重复文件和继承论文，而 `catalogInputPapers` 不包含。可见这种形状就是全量扫描的正常输出，不是边角情况。

## 可能的解决方案

### 方案 A（推荐）：在提案里记录扫描范围

给 `PersonalLibraryDirectionProposal` 增加可选字段，例如 `scope?: "full" | "new-papers"`：

- `assembleProposal`（`personal-library-direction-proposer.ts`）由两个提议器分别写入 `"full"` 或 `"new-papers"`。
- `decodeDirectionProposal`（`packages/core/src/library/personal-library-interest-profile.ts:345`）照 `coveredPaperKeys`、`accountedPaperKeys` 的现有写法，把它作为可选键接受，并校验取值。审阅编辑会先解码再输出提案（`personal-library-proposal-review.ts`），所以会自然保留这个字段。
- 在 core 新增一个判断函数，例如 `isNewPapersUpdate(proposal)`，即 `proposal.scope === "new-papers"`。`countNewPapersCoveredByExistingDirections`、插件第 537 行、Workbench 第 201 行都改用它，不再各自比较集合大小。Workbench 前端已经从 `@arxiv-daily/core` 引入 `accountedPaperKeysOf` 等函数，可以直接引入这个新函数。
- 没有该字段的提案一律按全量处理，不显示"本次新增"和"已覆盖的新论文"。
- 可顺带处理插件概览的 "N papers analyzed"：全量时改用本次输入篇数，避免把合并掉的重复文件和继承论文也算作"本次分析过"。增量时仍可显示已统计数，并附上"本次新增"。

代价：

- 不需要提升 `PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION`。`personal-library-interest-profile.ts:356-359` 的注释约定：v6 已发布，新字段一律作为可选新增。
- `decodeDirectionProposal` 用 `isExactObject` 严格拒绝未知键。降级到旧版本的插件或 CLI 后，带新字段的提案会解码失败，需要重新生成。#93 加入 `accountedPaperKeys` 时也是同样的处境。
- 修复前已保存的增量提案没有该字段，会被当成全量，"已覆盖的新论文"提示不再显示，直到下一次增量更新。这是安全方向的退化，不会显示错误数字。

### 方案 B：比较生成契约指纹

不改数据格式，只在 core 新增 `isNewPapersUpdate(proposal)`：

```ts
proposal.generationContractFingerprint ===
  createPersonalLibraryGenerationContractFingerprint(
    createPersonalLibraryNewPapersDirectionGenerationContract(),
  )
```

三处调用方都改用它。

代价：

- 契约字符串包含 `PERSONAL_LIBRARY_CLUSTERED_DIRECTION_PROPOSER_VERSION`、`PERSONAL_LIBRARY_DIRECTION_ORGANIZATION_PROMPT_VERSION` 等版本号。以后任何一次版本升级，都会让已保存的增量提案不再匹配而被当成全量，提示随之消失。这同样是安全方向的退化，但每次升级都会发生。
- 判断依赖"指纹只由契约决定"这一实现细节，可读性和长期稳定性都不如显式字段。
- 优点是零格式变更，不影响降级兼容，改动最小。可以作为方案 A 落地前的临时修复。

### 不推荐：缩小已统计集合

不推荐让全量分析不再把合并掉的重复文件和继承论文并入 `accountedPaperKeys`。这会违背 ADR 0015 第 3 节的目的：这些论文会在下一次计数时重新被当成新论文（`listUnaccountedLibraryPapers` 依赖该集合），触发错误的新论文提醒。

## 修复后应补的测试

- core：`countNewPapersCoveredByExistingDirections` 在"全量扫描、带合并重复文件或继承论文"的提案上返回 0。可直接用 `library-new-papers-proposal.test.ts` 里上述全量用例的结果来断言。现有 `packages/core/tests/personal-library-change.test.ts:217` 的 "is zero for a scan of the whole library" 只覆盖了已统计等于输入的情形，没有覆盖这种情况。
- core，若采用方案 A：解码器接受 `scope` 的两种取值，拒绝其他取值，缺省时按全量处理；`personal-library-proposal-review.ts` 的各种编辑之后 `scope` 仍然保留；两个提议器写入正确的取值。
- 插件：`plugin/tests/personal-library-interest-profile-modal.test.ts` 增加全量提案用例，提案里已统计多于输入。断言 "Library overview" 页没有 "added in this update"，"Topics from your library" 页没有 "already covered" 那一行。
- Workbench：`apps/cli/tests/workbench-library-review-ui.test.ts` 用同样的提案，断言概览没有"本次新增"，候选视图没有"其中 K 篇新论文已被你现有的方向覆盖"。

# P1d — inline-remote-consent

goal_ref: ../goal.md
updated: 2026-08-30

## Outcome

Library 行的主按钮只要选了文件夹永远是 Build index，"Review & authorize" 按钮不再存在。远程嵌入的全文授权改为「切到 remote 时就地确认」：确认即一步完成切换 + 授权，取消则下拉弹回 local、什么都不改。Revoke 走 A1：确认后撤销授权并切回本地嵌入，因此不存在「remote 但未授权」的死状态。

授权本身没有取消 —— 仍是全文深度、仍披露具体接口地址、仍绑定接口指纹、仍可撤销。ADR 0005 §5 与 ADR 0008 §2 的决定不变，本阶段只改询问时机。

## Assumptions

- 选目录只表达「可以本地读」，把全文发到远程接口仍需一次明确同意（ADR 0005 §5）。
- 同意的时刻可以移动，同意的内容不能缩水：披露必须能写出文件夹与目标接口，否则不算有效披露。
- 披露不能名不副实：没有文件夹（或没有配置远程接口）时弹框写不出目的地，只能推迟。
- declarative（1.13+）与 legacy display() 必须走同一套判断，Embedding 下拉与远程字段两条路径各写一份渲染。

## Approach

一个共享的 `requestRemoteFullTextConsent()` 承担全部询问，返回 granted / declined / undisclosable，被四个时机调用，先到先弹：

1. **切到 remote**（`applyEmbeddingModeChange`）：先用 `embeddingMode: "remote"` 的覆盖参数算出披露（此时 mode 还没改），确认后才落库 mode 再写授权；取消则一个字节都不改。
2. **选完文件夹**（`chooseLibraryRoot` → `offerEmbeddingModeChoice`）：切换时没有文件夹可写，推迟到这里。
3. **改动已授权的接口**（`saveEmbeddingEndpointField`）：字段落库后指纹若失配就重新询问，取消把字段恢复成已授权的值。
4. **remote 未授权时点 Build index**（`ensureRemoteEmbeddingConsent`）：升级上来的遗留配置的兜底，取消则中止建索引、状态不变。

`librarySetupNextStep` 删掉 `authorize` action，改为 `index` + `remoteConsentPending` 标志：按钮层再也拿不到「授权」这个分支，描述文字仍能区分远程未授权与已授权。`libraryAuthorizationDisclosure` 的 depth 改为取自 scope 而非既有 grant，否则首次远程授权前会把 full-text 披露成 metadata-only。

## Test strategy

- change kind: behavior change（同意时机与设置页交互）
- strategy: strict Red–Green on connection / modal / tab / declarative rows / legacy rows
- Red / baseline signal: 旧测试锁死 remote 未授权时主按钮为 "Review & authorize"、`librarySetupNextStep` 返回 `authorize`
- Green / regression checks: library-connection、library-connection-lifecycle、library-modal、library-inline-remote-consent、settings-declarative-tab、settings-tab、settings-definitions

## Tasks

- [x] `librarySetupNextStep` 去掉 authorize action，保留 `remoteConsentPending` 供描述文字使用。
- [x] `renderLibraryConnectionControls` 不再产出授权按钮（declarative + legacy 共用一处）。
- [x] 切到 remote 就地弹披露，确认一步完成切换 + 授权，取消什么都不改。
- [x] 未选文件夹时推迟到选完文件夹再弹。
- [x] 接口指纹变化后重新询问，取消恢复已授权的值。
- [x] Revoke 先确认，确认后撤销授权并切回 local。
- [x] remote + 未授权点 Build index 时先弹披露，取消中止建索引。
- [x] 披露文案补上用途（相似度向量）、其它内容不离开本机、可随时撤销。
- [x] 披露里的嵌入接口地址改成真正会被 POST 的 `{baseUrl}/embeddings`，不再显示 chat/completions。

## Verification

- 任何状态下 Library 行按钮数 ≤ 3 且没有含 "authoriz" 的文案；主按钮只会是 Choose folder / Build index。
- 切 remote 确认 → mode=remote 且已授权；取消 → mode=local 且授权状态不变、未写盘。
- 未选文件夹切 remote → 不弹；随后选文件夹 → 弹一次并授权。
- 已授权改接口取消 → 字段回到旧值且仍为 authorized；确认 → 按新指纹重新授权。
- Revoke 确认 → 撤销且 mode=local；取消 → 两者都不变。
- 遗留 remote+未授权 → 按钮仍是 Build index，点击先弹，取消不建索引、状态不变。
- 披露里的嵌入接口地址等于 `{baseUrl}/embeddings`，凭据与查询参数仍被脱敏。
- Observed Green (2026-08-30): plugin 668 tests、typecheck、build 通过；boundaries OK；lint 18 warnings（上限 64）。
- 已知无关红：`packages/core/tests/pipeline/pipeline-novelty-stage.test.ts` 单跑即 OOM（本分支已提交的 core 测试，本阶段未碰 packages/）。

## Abort / reshape triggers

- 如果用户觉得「切下拉就弹框」太突兀，下一步是把披露并进 Build index 一个时机，而不是把授权按钮加回来。
- 如果推迟披露导致有人长期停在 remote+未授权，把 Library 行描述写成更明确的一句提示，而不是新增按钮。

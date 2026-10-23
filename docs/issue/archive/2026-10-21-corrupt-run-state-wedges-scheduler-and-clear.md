# run-state.json 一旦损坏，所有日报运行都以含糊的 JSON 报错失败，连"清除运行状态"也无法修复

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 STATE-02
> 范围：运行状态存储 `run-state.json`（Obsidian 插件、CLI、工作台共用） · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-21-batch1`，提交 284ace6） · 置信度：已复现

## 问题

`<输出目录>/.index/run-state.json` 是按日期记录"已完成 / 失败 / 运行中"的文件。如果它变成无法解析的内容，例如：

- Vault 被 git 管理，两台机器各自提交后合并，文件里留下 `<<<<<<<` / `=======` / `>>>>>>>` 冲突标记；
- iCloud、Dropbox、坚果云之类同步工具生成了冲突副本并覆盖，或把文件截断；
- 用户手动编辑时写坏了 JSON；

那么此后：

1. **每一次日报运行都失败。** 调度器在每次运行前都要从磁盘读取"权威"状态，解析失败就直接返回 `failed_transient`，原因文字是裸的 JSON 解析错误，例如 `Unexpected token '<', "<<<<<<< HE"... is not valid JSON`，没有文件名，也没有说明怎么办。定时 tick 每 20 分钟重复一次同样的失败；因为这类失败发生在状态写入之前，不会累计到重试上限，也不会变成永久失败，只是永远在重试。
2. **界面看起来一切正常，只是历史是空的。** 面板和工作台读取走另一条"宽松"路径：主文件损坏且没有备份时，直接当作空状态并只在日志里警告。日历上没有任何已完成的日期，看不出文件已损坏。
3. **用户能想到的所有修复入口都不好使。** 命令"Clear arXiv Daily run state"（`clearAll()`）也要先读取权威状态，所以同样抛出这条 JSON 错误，文件保持原样。强制重跑、重试失败同理。唯一的办法是自己找到被隐藏的 `.index` 目录并手动删除文件。
4. 平时也没有备份可以回退：`writeAtomic` 在每次成功提交后会删除 `.bak`，只有在两次改名之间崩溃的极短窗口里 `.bak` 才存在，所以绝大多数损坏发生时都没有 `.bak` 可以接管。

"主文件损坏时拒绝写入、以免把历史清空"本身是有意的保护（已有测试），问题在于保护之后没有出口：用户明确要求"清除"时，旧内容本来就不需要，却仍然因为解析旧内容而失败。

## 原因

- `packages/core/src/services/state-store.ts:383-398` `loadAuthoritativeRunState`：主文件存在但无法解析时直接抛出 `JSON.parse` 的原始 `SyntaxError`（`parseRunState` 在 444-477 行），没有带上路径，也没有区分"损坏"和"暂时读不到"。
- `state-store.ts:235-250` `enqueueMutation`：每个变更（包括 `clearAll` 173-178 行、`replaceAll` 180-186 行）都先 `authoritativeLoadFn()`，所以"清除"和"整体替换"这两个本不依赖旧内容的操作也被损坏的旧内容挡住。
- `state-store.ts:575-599` `writeAtomic`：提交成功后 `storage.remove(bak)`（594 行），稳态下没有备份。
- `state-store.ts:400-442` `loadRunStateWithFallback`（界面用的宽松读取）：主文件损坏且无 `.bak` 时返回空状态，只 `warn`，与上面的严格路径行为不一致，造成"界面空白但运行失败"。
- `packages/core/src/services/scheduling/scheduler-driver.ts:585-589,1042-1065`：`loadAuthoritative()` 失败被包装成 `failed_transient`，原因直接取 `error.message`。
- `plugin/src/commands.ts:161-176` `clearRunState`：直接调用 `plugin.stateStore.clearAll()`，失败只弹出同样的 JSON 错误。

## 复现

临时 vitest（`packages/core/tests/zz-review-state-corrupt-runstate.test.ts`，用后已删除）：用内存存储放一个含冲突标记的 `arxiv-daily/.index/run-state.json`，没有 `.bak`：

```
lenient load -> dates: 0 | .bak exists: false
manual run -> {"kind":"failed_transient","reason":"Unexpected token '<', \"<<<<<<< HE\"... is not valid JSON"}
clearAll (the 'Clear arXiv Daily run state' command) -> SyntaxError: Unexpected token '<', "<<<<<<< HE"... is not valid JSON
file still conflicted: true
```

流水线一次也没有被调用；`clearAll` 之后文件原样保留。

## 可能的解决方案

推荐：给"保护"加一个不丢数据的出口，并让错误可读。

1. 严格读取遇到无法解析（语法错误或根结构非法）的主文件时，抛出带路径的专用错误，例如 `run-state.json 已损坏（<路径>）：请使用"清除运行状态"重置，原文件会被保留为 run-state.json.corrupt-<时间戳>`。未知 schema 版本仍保持现在的"拒绝写入"，那是升级兼容保护，不是损坏。
2. `clearAll()` / `replaceAll()` 不再依赖旧内容：遇到无法解析的主文件时，先把它改名为 `run-state.json.corrupt-<时间戳>`（保留证据，不丢数据），再写入新状态。
3. 宽松读取遇到损坏主文件且无有效备份时，除了返回空状态，也要让宿主能得知"已损坏"（返回标志或日志级别提升为 error），面板/工作台据此显示提示，而不是显示一个看似正常的空日历。

备选：在严格读取失败时自动把损坏文件隔离为 `.corrupt-<时间戳>` 并从空状态继续。最省事，但会在用户不知情时清空历史显示；建议至少同时给出明确的日志和面板提示。

## 修复后应补的测试

- 主文件含冲突标记、没有 `.bak`：`clearAll()` 成功，原文件以 `.corrupt-*` 名称保留，随后的运行可以正常开始。
- 同样的损坏文件下，运行返回的失败原因包含 `run-state.json` 的路径和修复指引。
- 未知 `schemaVersion` 仍然拒绝写入，`clearAll()` 对它也保持拒绝（或要求显式确认），不会悄悄丢掉新版本写入的状态。
- 宽松读取遇到损坏主文件时，宿主能拿到"已损坏"信号。

## 修复情况

`run-state.json` 无法解析（冲突标记、被截断、手改坏）时，运行不再以裸 JSON 报错失败，而是返回带文件路径和处理办法的错误（`CorruptRunStateError`）；"Clear arXiv Daily run state" 现在能成功：把损坏文件改名为 `run-state.json.corrupt-<时间戳>`（不删除，也不覆盖已有副本；只有备份坏时则隔离备份）后从空状态开始，之后的运行恢复正常。未知 schema 版本和非法可选字段仍然拒绝清除，以免丢掉新版本写下的数据。`StateStore.damage()` 让宿主得知宽松读取遇到了损坏：插件启动时弹出提示并写 error 日志，`arxiv-daily status` 输出 `runStateDamaged`。与 write-up 的差别：`replaceAll()` 没有改成隔离损坏文件，因为它只服务旧版数据的自动迁移，不该在用户不知情时自己丢弃文件；工作台网页也没有加损坏横幅。

测试：`state-store.test.ts`（11 例）、`scheduler-corrupt-run-state.test.ts`、`apps/cli/tests/cli-inspect.test.ts`、`plugin/tests/run-state-damage.test.ts`。

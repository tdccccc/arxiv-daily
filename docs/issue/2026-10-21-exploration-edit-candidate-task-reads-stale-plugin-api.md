# Obsidian 探索任务 library-edit-candidate 永远阻塞，发版探索的 Obsidian 一端无法得到绿灯

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 SCR-02
> 范围：`scripts/acceptance/exploration-obsidian.mjs`、`exploration-tasks/obsidian.mjs`、`scripts/tests/acceptance-exploration-host.test.mjs` · 严重程度：中 · 状态：未修复 · 置信度：已复现

## 问题

跑 `npm run test:acceptance -- --suite exploration --host obsidian --release` 时，21 个目标任务里的 `library-edit-candidate`（编辑个人文献库的一个候选研究方向）**每次都会被阻塞**，0 次模型调用，几百毫秒内结束，报错：

```
library-edit-candidate: prepare()'s real generation did not produce any proposed topic/direction to edit
```

整次探索因此以 `blocked`、退出码 2 结束，发版用的 Obsidian 证据永远拿不到全绿。2026-10-21 协调者的真实运行就是这个结果（`passed: 20, blocked: 1`，阻塞的正是这个任务，`usage.modelCalls = 0`、耗时 720 ms）。报错信息会把人引向“固定模型响应没产生提案”或“聚类失败”，而实际上提案已经生成并保存成功了（同一次运行里夹具服务器收到了该任务发出的 `proposal` 请求，状态 200）。

这不是偶发的：2026-10-14 的发版运行里这个任务还是通过的，之后插件的接口改了，探索宿主没有跟着改，而单元测试里的假驱动恰好还模拟着旧接口，所以 `npm run test:release-tools` 一直是绿的。

## 原因

- `plugin/main.ts:1203-1331`（提交 `00a41eb`，2026-10-15，“Nudge library changes and rescan only new papers”）把 `generatePersonalLibraryDirections()` 的返回值从提案文档本身改成了包装对象 `{ outcome: "generated", proposal, scope, fellBackToFull }`（没有新论文时是 `{ outcome: "nothing-new" }`）。
- `scripts/acceptance/exploration-obsidian.mjs:1126-1133`：探索驱动把这个返回值原样 `JSON.stringify` 后交给调用方，没有取 `.proposal`。
- `scripts/acceptance/exploration-tasks/obsidian.mjs:1246-1251`：`prepare()` 读 `proposal?.topics?.[0]`。拿到的是包装对象，`topics` 不存在，于是抛出上面的错误；`libraryEditCandidateBaseline` 也就没有机会被记录。
- `scripts/tests/acceptance-exploration-host.test.mjs:1765-1806`：`createFakeLibraryTaskDriver().generatePersonalLibraryDirections()` 直接返回 `{ topics: [...] }`（旧形状），所以“prepare 能拿到提案”的测试在真实接口下并不成立，却一直通过。
- 同一类问题的另一个调用点 `scripts/acceptance/obsidian-library-change.mjs:111` 忽略了返回值，不受影响。

复现：用真实的 `library-edit-candidate` 任务代码（`goalTasks()` 里的 `prepare`），驱动的 `generatePersonalLibraryDirections` 返回插件现在真实返回的形状，`prepare()` 立即抛出上述错误；换成旧形状（`{ topics: [...] }`）则通过。

## 可能的解决方案

推荐：在探索驱动 `generatePersonalLibraryDirections()` 里按真实契约解包——`outcome === "generated"` 时返回 `.proposal`；`outcome === "nothing-new"` 时抛出带明确原因的错误（“没有新论文，未生成提案”），而不是让调用方去猜。同时让 `prepare()` 在拿不到 `topics[0].directions[0]` 时把拿到的 `outcome` 一并写进错误信息，避免再次误导。

备选：任务 `prepare()` 不用返回值，改为生成后调用已有的 `driver.readLibraryDirectionProposal()` 从磁盘读取提案（这条路径读的是插件真正保存的文件，对返回值形状不敏感）。

## 修复后应补的测试

- 把 `createFakeLibraryTaskDriver` 的 `generatePersonalLibraryDirections()` 改成返回真实形状 `{ outcome: "generated", proposal, scope: "full", fellBackToFull: false }`，让 `library-edit-candidate: prepare()` 的现有用例在真实形状下通过。
- 为探索驱动本身加一个测试：用一个假的 `session.evaluate`，让它返回 `JSON.stringify` 后的真实包装对象，断言驱动方法返回的是 `proposal`；再加 `{ outcome: "nothing-new" }` 的用例，断言得到明确的错误。
- 长期：给该驱动方法加一个类型/契约检查（例如从 `plugin/main.ts` 导出的 `GeneratedPersonalLibraryDirections` 形状里取字段名做断言），让插件接口再变时测试先红。

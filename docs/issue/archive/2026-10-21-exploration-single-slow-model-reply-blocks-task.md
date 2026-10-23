# 探索里模型只要有一次回复超过 45 秒，该任务就被判为阻塞，整次发版运行退出码变成 2，且超时时间无法从命令行调整

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 SCR-08
> 范围：`scripts/acceptance/exploration.mjs`（`runExplorationLoop`）、`scripts/acceptance/run.mjs`（命令行选项） · 严重程度：低 · 状态：已修复（分支 `fix/2026-10-21-batch4`，提交 5a4797b） · 置信度：代码推断

## 问题

探索循环对每一次模型调用设了固定的 45 秒期限（`modelCallTimeoutMs` 默认 45000）。一次调用超过 45 秒就被中止，这个任务立刻以“阻塞”结束，没有重试，也不会再给模型第二次机会；只要整次运行里有一个任务是阻塞的，`report.json` 的总状态就是 `blocked`，退出码为 2，发版证据就不是全绿。

这个期限只能通过代码里的函数参数改，`run.mjs` 的命令行（`--max-steps`、`--max-api-calls`、`--max-duration-ms`、`--task-max-*`）和 `README.md` 里都没有对应的选项。换用推理更慢的模型、把推理强度调高，或者服务商某一刻延迟抖动，都会直接撞上这个固定值。

已经发生过：`docs/helm/2026-10-13-exploration-safety-and-coverage/journal.md` 记录了 2026-10-14 的真实发版运行中，Obsidian 的 `completed-date-idempotent` 任务“在任何动作之前”就因为 `Model API call timeout after 45000ms` 被阻塞，单独重跑通过，当时的结论是“服务商瞬时延迟，不处理”。2026-10-21 当天四次本地探索（约 400 次模型调用，平均每次 5–6 秒）没有出现超时，所以对当前的服务商这不是常态问题；但每次发版都要靠“碰巧没有一次慢回复”才能拿到绿灯，而补救办法只有“单独重跑那个任务”。

回答协调者的问题：高推理强度的模型是否会因此卡住任务？取决于单次回复延迟的尾部。当前这家服务商的平均延迟远低于 45 秒，没有观察到；只要尾部延迟（不是平均值）超过 45 秒，任务就会阻塞，没有缓冲。

## 原因

- `scripts/acceptance/exploration.mjs:275`：`const callTimeout = limit(modelCallTimeoutMs, 45000, 300000, "modelCallTimeoutMs")`。
- `scripts/acceptance/exploration.mjs:400`：调用模型时 `withDeadline(…, Math.min(callTimeout, deadline - Date.now()), "Model API call", signal)`，超时后抛出 `ExplorationTimeoutError`。
- `scripts/acceptance/exploration.mjs:471`：外层 `catch` 把任何模型调用错误（包括超时）一律写成 `scenario.status = "blocked"` 并结束该任务，没有“同一步重试一次”的逻辑；超时也不区分“这次调用慢”和“服务不可用”。
- `scripts/acceptance/run.mjs:213`：`runAcceptance` 传给探索的选项里没有 `modelCallTimeoutMs`，`parseArgs` 也没有对应的开关，尽管 `runExploration`（`exploration.mjs:545`）本来就接受这个参数。

## 可能的解决方案

推荐：新增 `--model-call-timeout-ms N`（范围与现有校验一致，1 到 300000），文档和 `--help` 写明默认 45000；同时对“超时”这一种错误允许同一步重试一次（重试仍计入模型调用和 API 调用预算，被中止的上一次请求不再计费以外的影响），仍超时才判阻塞。

备选：只加命令行选项，不加重试。能解决“换了慢模型”的情况，但解决不了偶发的单次抖动。

## 修复后应补的测试

- `scripts/tests/acceptance-runner.test.mjs`：`parseArgs` 接受并校验 `--model-call-timeout-ms`，并传给探索运行器。
- `scripts/tests/acceptance-exploration.test.mjs`：模型第一次回复超过期限、第二次正常时，任务继续并最终可以通过；连续两次超时才阻塞；重试计入步数和 API 调用预算。

## 修复情况

可用 `--model-call-timeout-ms N` 为慢模型调整单次等待时间，默认45000毫秒、最多300000毫秒，帮助和探索说明已同步。一次回复超时后取消该请求，并在重新观察界面后最多重试一次；连续两次超时仍明确阻塞。重试消耗同一份任务及全局步数、API和时间预算，预算耗尽不会继续请求，用户取消和认证错误不重试。超时记录仍保留在逐步证据中。

验证：3个新增恢复/参数用例先失败；修复后探索及runner完整108项通过，覆盖第一次超时后通过、连续超时、真实本机HTTP请求取消、各预算耗尽、父级取消及认证错误；全部模型均为stub或本机HTTP夹具。

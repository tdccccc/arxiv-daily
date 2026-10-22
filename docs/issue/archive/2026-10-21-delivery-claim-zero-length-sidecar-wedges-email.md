# 崩溃留下的一个空 claim 文件，会让之后每一天的自动日报邮件都发不出去，且没有恢复办法

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 MAIL-01
> 范围：自动邮件投递状态（`delivery-state.json.claims/`，插件与 CLI 共用） · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-21-batch1`，提交 a2e9b83） · 置信度：已复现

## 问题

自动日报邮件在发送前，要在 Vault 的 `arxiv-daily/.index/delivery-state.json.claims/` 里为“日期 + 收件人”建一个 claim 文件。建文件是两步：先用 `O_EXCL` 创建一个空文件，再写入 JSON 内容（Linux 回退路径和原生存储都是如此）。如果进程恰好在这两步之间被杀（SIGKILL、OOM、断电、关机时 Obsidian 被强退），目录里就会留下一个 0 字节的 `<hash>.g0.claim.json`。同样，decision / result 文件也是先建后写，也可能留下 0 字节文件。

之后只要目录里有这样一个文件，**所有日期**的自动邮件都会失败，不只是出问题的那一天：

1. 新的一天运行日报后，`claimAutomaticDelivery` 成功创建自己的 claim；
2. 紧接着 `rebuildDeliveryStateFromClaims` 要读取目录里**每一个**已有的 claim，空文件在 `JSON.parse` 处抛错；
3. 这个新 claim 被立刻释放，结果是 `failed / delivery_state_update_failed`，邮件没有发出；
4. 第二天重复同样的过程，每次还会再多留下一对 claim + decision 文件。

用户看到的现象：日报文件正常生成，但邮件再也不来了。插件界面没有任何提示（`plugin/main.ts` 只对 `delivery_storage_unsupported` 弹通知），CLI 只在日志里写一行 `email: cannot claim <date>: delivery_state_update_failed`，既没有说是哪个文件坏了，也没有任何命令或文档告诉用户怎么恢复。只有手动进到 `.index/delivery-state.json.claims/` 里找出那个 0 字节文件并删除，才能恢复。

已用内存存储复现（一次性测试，已删除）：目录里放一个 0 字节的 `g0.claim.json`（模拟崩溃残留），随后依次对 2026-07-25、07-27、07-28 三个日期调用 `deliverDailyEmailIfEnabled`，三次都返回 `{"kind":"failed","reason":"delivery_state_update_failed","attempts":0}`，Resend 请求次数为 0，每个日期又各多留下一对 claim/decision 文件。把坏文件换成 0 字节的 `g0.decision.json`（对应的 claim 正常）结果相同。

按设计，损坏的 sidecar 应当“失败即关闭”（宁可不发，也不能重复发），这一点没有问题。问题在于两点：

- 0 字节 claim 文件是这套“先建后写”协议自己会留下的崩溃残留，而不是外部损坏；它证明这个 claim 从未写完，不可能有任何 provider 调用发生过（provider 调用之前必须先读回并校验自己的 claim），按同样的“可证明未调用”规则，恢复是安全的。代码对“陈旧但完整的 claim”有恢复路径（5 分钟后创建 `recovered_before_provider_attempt` decision），对不完整的文件却没有。
- 影响范围没有被隔离：一个坏文件让所有日期失败，而不是只阻断它自己的那个日期。

## 原因

- `packages/core/src/delivery/delivery-state.ts:859-883` `readLatestGenerations` 与 `:907-926` `readGeneration`：对目录里每个 stem 读取 claim、decision、result；`:1040-1050` `readJsonDocument` 对不可解析的内容直接抛错，没有按 stem 隔离，也没有区分“空文件”和“内容被改坏”。
- `delivery-state.ts:657-692` `rebuildDeliveryStateFromClaims` 在 `:673` 读取全部 stem；任何一个 stem 抛错都会让这一次重建失败。它在 claim 创建后（`:422-426`）、provider 调用前就被调用，所以所有日期都在发送前失败。
- `delivery-state.ts:440-463`：重建失败后把刚建的 claim 释放，返回 `delivery_state_update_failed`；该原因码对用户没有任何指向性。
- `packages/node-runtime/src/native-private-storage.ts:55-62`、`packages/node-runtime/src/storage-adapter.ts:426-457`：`createTextExclusive` 先 `createFile`/`O_EXCL` 再写内容，两步之间没有任何崩溃保护（正常错误会 `removeBestEffort`，被杀则不会）。

## 可能的解决方案

推荐：把“不可解析”按 stem 隔离，并区分两类情况。

1. `readGeneration` 遇到 0 字节（或只含空白）的 claim：当作“从未写完的 claim”，该 stem 视为已释放（可以安全进入下一个 generation），不抛错。
2. 遇到 0 字节的 decision / result：decision 说明 provider 调用尚未发生（先于调用写入且要先成功），可视为“未写完”并忽略；result 为空则按“provider 调用已开始、结果不明”处理，保持阻断，但只阻断这一个 stem。
3. 其他不可解析的内容（真正被改坏）：只把对应 stem 标记为“阻断 + 需要人工处理”，不要让 `rebuildDeliveryStateFromClaims` 整体失败；并给出带文件名的日志，插件对这类原因码弹一次通知（与 `delivery_storage_unsupported` 的通知方式一致）。

备选：保持“任何坏文件都整体失败”，但把原因码拆成 `delivery_claim_corrupt`，日志与通知里写明目录和文件名，并在 CLI 增加 `email status` 对坏文件的检查与提示，让用户知道删哪个文件。

## 修复后应补的测试

- 目录里有一个 0 字节的 `g0.claim.json`（任意日期）：其他日期仍能正常 claim、发送一次、记为 delivered。
- 0 字节的 decision 文件：对应日期按“未调用”恢复或只阻断该日期；其他日期不受影响。
- 0 字节的 result 文件：对应日期保持阻断（不重发），其他日期不受影响。
- 内容被改坏（非空、非 JSON）的 sidecar：只阻断对应 stem，日志指明文件名。
- 同一个 `createTextExclusive` 在写入前被中断的集成测试（`afterTargetCreated` 钩子里抛错/终止），之后的 `claimAutomaticDelivery` 能恢复。

## 修复情况

崩溃留下的 0 字节（或写到一半）的 claim / decision / result 文件，不再让所有日期的自动邮件失败：读取 sidecar 时坏内容不再抛错，重建主状态会跳过这样的日期（保留它原有的记录）并在日志里写出文件名，每个日期只读最新的一代。空的或无法解析的 claim（且没有 decision/result）按"从未写完的 claim"处理，下一次发送同一日期时用 decision 文件仲裁（`recovered_before_provider_attempt`）并进入下一代：这样的 claim 不可能发出过邮件，原 owner 之后也发不出。无法判断是否已发的情形一律失败即关闭且只影响该日期：decision 不可读、没有 decision 的 result 不可读、未知 schema 的 claim 返回 `failed / delivery_claim_unreadable`，日志写出完整路径（插件会弹一次通知）；decision 为 `provider_attempt_started` 而 result 不可读仍按已开始发送阻断。与 write-up 的差别：0 字节 decision 没有按"未写完"忽略，因为它可能属于正要调用 provider 的活着的 owner；`createTextExclusive` 本身仍是先建后写，没有改。

测试：`packages/core/tests/delivery/delivery-damaged-sidecars.test.ts`（23 例，22 例在旧代码下失败）、`packages/node-runtime/tests/delivery-crash.test.ts`（在 `createTextExclusive` 里真正 SIGKILL 一个投递进程，覆盖 claim / decision / result 三种文件、原生存储与 Linux 描述符回退两种后端，旧代码下全部失败）。

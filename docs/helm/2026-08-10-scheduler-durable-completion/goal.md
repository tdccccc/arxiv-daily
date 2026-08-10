# Scheduler Durable Completion

status: active
updated: 2026-08-10
owner: zcode-main-session

## Intent

只有 run-state 的完成记录被确认持久化后，Scheduler 才向调用方返回 completed 并触发完成通知、历史和邮件回调。

## Success criteria

- [ ] `setCompleted` 无法确认时返回 `failed_transient`，不显示完成、不记录 completed history、不调用 completion callback。
- [ ] StateStore 保存成功前不向读取者发布候选状态，失败后内存与 durable state 一致或保守恢复旧快照。
- [ ] 保存抛错但 durable candidate 已落盘时能通过重新读取确认，避免虚假失败。
- [ ] batch 在单日 commit 失败后继续处理其他日期，后续重试可以完成。
- [ ] Run history 保持 best effort，既有 Plugin/CLI 结果契约不变，完整门禁和技术报告同步。

## Non-goals

- 不新增 `PipelineResult` kind，也不建立 run-state 与 JSONL history 的跨文件事务。
- 不解决 Plugin 与 CLI 同时调度同一 Vault 的跨进程锁。
- 不修改邮件投递本身的幂等实现。

## Constraints

- `run-state.json` 是本事项唯一硬完成边界；history 是可修复 observability。
- 已提交日报可能存在，因此 commit 失败不得伪装成重新运行 pipeline 的永久失败。
- 所有 StateStore mutation 使用同一候选提交原则。

## Phases

1. P1 — Scheduler 只在 durable completion commit 后公开完成 — status: active

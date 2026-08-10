# Filter Response Validation

status: done
updated: 2026-08-10
owner: zcode-main-session

## Intent

确保论文过滤模型的响应只有在通过严格契约验证后才能产生成功零结果；非法响应应成为可诊断、可重试的失败，且不触发下游写入或投递。

## Success criteria

- [x] 非严格 JSON 和不符合过滤契约的模型响应映射为 `failed_transient`，并保留安全、明确的失败原因。
- [x] 严格 `{"papers":[]}`、合法 `skip` 和全部 ignored 的既有成功零结果语义保持不变。
- [x] 非法响应不保存过滤 checkpoint，不更新 Paper Index，不抓取正文，不写日报，也不进入完成后的邮件回调。
- [x] 定向测试、Core 类型检查、边界检查和现有 release 测试门禁通过。
- [x] 当前实现的技术报告与接受后的代码保持同步。

## Non-goals

- 不改变过滤结果契约允许省略部分输入论文的现有规则。
- 不为过滤响应新增内部即时重试或新的 `PipelineResult` kind。
- 不修改邮件幂等、Paper Index 持久化或 Scheduler 完成态提交语义。
- 不迁移或猜测历史 `completed` 零结果状态。

## Constraints

- 成功路径 API `filterPapers(): Promise<FilteredPaper[]>` 保持兼容。
- 不记录或嵌入原始、可能不可信的模型响应。
- Filter checkpoint schema、fingerprint 和 result contract version 不变。
- 本事项只在 `fix/filter-response-validation` 分支与对应独立 worktree 中实施。

## Phases

1. P1 — 非法过滤响应 fail closed，合法零结果保持兼容 — status: done

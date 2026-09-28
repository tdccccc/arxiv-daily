# Journal

## 2026-09-28 — frame and start P1

- evidence: 用户明确要求新开 helm 修复 F19、跨进程一致性、跨平台自动邮件和 F29–F32。旧 initiative 的记录延期不能作为新 initiative 的完成证据；基线 6de54d9 已在最终提交上完整复测通过，工作树干净。
- change: 创建独立目标与 `fix/review-followups` 分支，全部七项纳入成功条件；只展开 P1（CLI cron 与异步边界），其它阶段保持结果索引。
- disposition: 保留既有代码/测试和旧 helm 历史，不沿用这些条目的豁免。缺真实 provider 或原生平台时记录未验证，不将其标成修复完成。已询问用户验收接口与环境，P1 可独立推进。
- next: F29 真实 shell 回归 Red → 最小编码修复 → Green 与 CLI 回归。

## 2026-09-28 — P1 chunk 1 accepted

- evidence: F29 had 11 expected failures (eight executable path round-trips and three invalid-path checks); all 12 focused cases now pass. CLI schedule/config/main 39 tests and CLI typecheck pass. The shell is real; cron percent preprocessing is modeled from its command grammar, and no actual crontab is installed.
- checkpoint: On track; isolated implementation and tests committed.
- next: F30 invalid schedule window regression.

## 2026-09-28 — native-platform validation sequencing

- evidence: 用户选择先完成实现和 CI 验收配置，保留平台验收待办。F19 的接口问题已用产品语言解释；到该阶段先辨认当前配置，再说明可验证的服务。
- change: P5 保留跨平台实现与 CI 配置，新增 P6 作为原生平台验收待办；不预建未来阶段文件，P1 继续执行。
- disposition: 原生验收仍未完成，不用 Linux 模拟替代；不因此停止可独立进行的实现。
- next: 完成 F30，接着 F31。

## 2026-09-28 — P1 chunk 2 accepted

- evidence: F30 had two expected failures (TOML accepted the reversed window; install returned success and touched crontab). After validation, focused tests 23 and full CLI 95 pass, along with CLI typecheck. Valid single-run, equal-boundary and recurring windows retain their behavior.
- checkpoint: On track; isolated F30 implementation and tests committed.
- next: F31 async command errors and handler lifetime.

## 2026-09-28 — P1 done, start P2

- evidence: F31 had 12 expected failures across command rejection, configuration exit codes and email signal-handler lifetime. All 37 focused tests and 107 CLI tests now pass; root typecheck, build and boundaries pass. F29/F30 retain their accepted evidence.
- checkpoint: On track; P1 complete after its three isolated fix commits.
- change: check F29–F31 criteria, mark P1 done and activate P2 for relay request JSON shape validation.
- next: F32 failing request-boundary tests; no real mail or live cutover operations.

## 2026-09-28 — P2 done, start P3

- evidence: F32 produced 12 expected failures among 20 request-shape cases. Shared object parsing now passes all 20 cases and the full 161-test relay suite; typecheck passes. No provider call, gate dispatch or state write for invalid public input. Fix committed as 2b49954.
- checkpoint: On track; P2 accepted and P3 active.
- provider evidence: fetched official OpenAI Chat Completions reference, DeepSeek thinking guide, Anthropic OpenAI SDK compatibility and Zhipu thinking guide. SDK extra_body must be flattened; OpenAI accepts reasoning_effort without a thinking field; DeepSeek accepts thinking plus effort; Anthropic accepts top-level thinking through its compatibility endpoint; the currently preset GLM models use thinking without reasoning_effort. Transport changes must also invalidate generation checkpoints created with the old wire contract.
- sources: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/ ; https://api-docs.deepseek.com/guides/thinking_mode ; https://platform.claude.com/docs/en/api/openai-sdk ; https://docs.bigmodel.cn/cn/guide/capabilities/thinking
- environment: installed plugin selects DeepSeek but names a GPT model through a private gateway; do not infer the gateway protocol from the preset or mutate the user's configuration. A concise synthetic live-test question is pending; no live request has been sent.
- next: HTTP payload contract Red/Green plus checkpoint invalidation, then the selected live-interface acceptance when authorized.

## 2026-09-28 — L1 adjust P3 for current Anthropic models

- evidence: all 171 focused tests and the full workspace suite passed after request correction and checkpoint versioning. User explicitly authorized the existing private gateway; one synthetic arithmetic call returned HTTP 200 and the expected digit 2 with thinking enabled/low effort. No plugin config modification or paper data was sent. Official Claude extended-thinking documentation states 4.7+ rejects manual budget_tokens, which includes the project's default Opus 4.7 preset.
- change: retain the provider fix, add model-aware adaptive thinking and a native Messages request/parser for the official Anthropic host so effort can reach its documented output_config. Third-party compatibility endpoints retain their route. Add tests before this production extension; generation endpoint hashing follows the selected route.
- source: https://platform.claude.com/docs/en/build-with-claude/extended-thinking
- disposition: do not treat gateway success as a live test of Anthropic. Keep original local tests, add native response-contract coverage; adjust endpoint fingerprint fixtures only for the changed actual route. No additional live calls are needed for the unchanged private-gateway path.
- next: observe native Anthropic contract Red, implement, and rerun focused and affected regressions before accepting P3.

## 2026-09-28 — P3 done, start P4

- evidence: F19 committed as bb3307e. Observed 11 payload Reds, four old-checkpoint reuse Reds, and four native Anthropic Reds. Final full workspace regression: core 2074 passed / two existing skips, node-runtime 45, CLI 107, plugin 751; root lint/typecheck/build/boundaries/submission passed. The user's selected gateway accepted the synthetic request (HTTP 200, answer 2). Native Anthropic is locally contract-tested; no claim of live verification with an Anthropic key.
- checkpoint: On track; request and checkpoint semantics accepted together, including deterministic fingerprint fixture regeneration and adaptive checkpoint round-trips.
- change: P3 done, P4 active. Current index code already rereads disk inside each mutation; the remaining defect is the lack of a shared lock around that full transaction, not an indefinitely cached inbox.
- next: implement host-local locks with atomic immutable ownership/decision records and OS liveness checks, then wire daily generation and index transactions. Do not reclaim a live owner merely because time elapsed. Coordination records are local to the host and keyed by the canonical Vault root, outside exportable research data.

## 2026-09-28 — P4 shared-lock primitive accepted

- evidence: missing-module contract baseline followed by four actual behavioral Reds with a no-op lock. Real OS subprocess tests now prove exclusion, serialized counter updates, recovery after SIGKILL with competing contenders, and no takeover of a live holder on timeout/cancellation. Full node-runtime 49 tests and typecheck passed.
- checkpoint: On track; shared lock port and Node implementation committed separately from host wiring. Coordination uses the same OS PID namespace and local hard-link-capable filesystem; uncertain liveness remains busy.
- next: reproduce and repair RunLock/PaperIndexStore host wiring.

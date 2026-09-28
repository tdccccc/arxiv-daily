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

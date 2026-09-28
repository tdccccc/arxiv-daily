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

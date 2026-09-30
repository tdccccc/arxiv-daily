# Review follow-ups: provider, shared Vault and delivery reliability

status: active
created: 2026-09-28T22:22:37+08:00
updated: 2026-09-30T09:10:20+08:00
revision: 11
owner: claude-root-session-2026-09-29

## Intent

修复上一轮 review 明确遗留的 F19、跨进程一致性、跨平台自动邮件及 F29–F32，使 CLI、Plugin product 与邮件中继在这些场景下具有正确且可验证的行为。本轮以实际修复和验收为完成条件，不能沿用上一轮“仅记录延期”的处置代替修复。

## Success criteria

- [x] F29：含空格、引号、shell 特殊字符和 `%` 的 CLI 路径生成可正确执行的 cron 命令；无法安全表示的路径在写 crontab 前被拒绝。
- [x] F30：非法或倒置的重复运行时间窗被拒绝，旧 crontab 不被覆盖或删除；有效时间窗及单次运行保持正常。
- [x] F31：异步子命令失败统一返回正确退出码和脱敏错误；信号处理器在操作真正结束后清理。
- [x] F32：中继拒绝非对象 JSON（包括 null/数组），返回明确客户端错误，不访问投递服务或修改状态。
- [x] F19：服务商收到其支持的推理参数，不再发送 SDK 专用 extra_body 包装；非推理模式、流式兼容回退和输出上限保持正确。请求契约与所选真实接口验收均有记录。
- [x] 同一台机器多个 Plugin/CLI 进程使用同一 Vault 时，运行互斥且 PaperIndexStore 更新不丢失；进程退出/崩溃和争用路径有真实多进程验证。
- [ ] Linux/macOS/Windows 的 Plugin/CLI 自动邮件使用可靠的持久化投递占位，不因平台本身被拒绝；重复发送、崩溃、路径置换与恢复的保护保持有效，并有原生平台验收记录。
- [ ] 所有行为修复分别提交并附 Red/Green 与相关回归证据；root lint/typecheck/test/build/boundaries/submission 和 relay 独立测试/类型检查通过，技术报告同步。

## Non-goals

- 本轮不扩展到 F23 的网络增量流式改造、F24 代理对截断及未确认的 topics 迁移疑点。
- 不承诺多机器经网盘同步或网络文件系统的分布式互斥。
- 不执行线上 relay cutover、不发送真实邮件、不合并或发布；推送和 PR 仅限下述本次原生验收授权。

## Constraints

- 从已验收的 6de54d9 创建本地分支 `fix/review-followups`，旧 helm 保持 done 与原历史。
- 沿用不使用子代理的偏好；每个阶段由当前会话串行推进。
- 每次只展开当前阶段；后续阶段到启动时才写具体方案与测试策略。
- 2026-09-28 用户决定：先完成跨平台实现和 CI 验收配置，原生平台验收单独保留 P6 待办。
- 2026-09-29 用户选择 1A：允许随应用增加系统原生支持组件；先验证可行性，再接入和配置分发，安全要求与 P6 原生验收边界不变。
- 2026-09-30 用户选择 1A：本次允许推送 `fix/review-followups` 并创建验收 PR，运行原生 CI 及同一验收任务的修复重跑；不合并、不发布、不发真实邮件，其他推送/PR 仍需另行授权。
- 不以普通 write-before-check、缩短占位寿命或仅凭墙钟超时破坏现有防重复投递与路径边界。
- 用户本轮明确点名的项目未经用户重新决定不豁免；缺真实接口或原生平台证据时保留未完成状态并写明阻碍。
- commit 使用 Conventional Commits 英文动词主题，多个 -m 分别写 Why/What/Validation；提交前检查 staged diff。

## Phases

1. P1 — CLI 的 cron 安装与异步命令边界正确（F29–F31） — status: done
2. P2 — 邮件中继拒绝非法 JSON 对象形状（F32） — status: done
3. P3 — 服务商推理参数按实际 HTTP 契约发送并验收（F19） — status: done
4. P4 — 同机多个进程共享 Vault 时运行互斥、索引更新不丢失 — status: done
5. P5 — 跨平台自动投递实现与 CI 验收配置就绪 — status: done
6. P6 — macOS/Windows 原生平台验收（CI 第一关已获准，桌面验收单列） — status: active

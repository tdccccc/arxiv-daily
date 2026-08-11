# Hosted Relay Transition Compatibility

status: done
updated: 2026-08-11
owner: zcode-main-session

## Intent

让 0.4.2 客户端在 relay v2 cutover 前安全识别当前生产 Worker 的有限成功响应，避免 provider 已接受邮件后因响应体代际差异被误判为 ambiguous。

## Success criteria

- [x] Hosted 客户端接受精确 `{ "ok": true }` 及两种严格白名单的旧 Worker 成功体。
- [x] 旧响应中的 provider ID 仅用于契约识别，验证后立即丢弃，不进入日志、状态或公开结果。
- [x] 其他额外字段、非法类型、空或超长 ID 继续 fail closed，并阻断 automatic 自动重发。
- [x] Core 定向测试、完整门禁和技术报告同步通过。

## Non-goals

- 不修改 relay Worker、部署生产服务、发送真实邮件或发布 0.4.2。
- 不放宽非白名单 2xx 响应，不把 provider ID 暴露给调用方。

## Constraints

- 兼容层必须严格、有界且可在 relay 完成迁移后独立移除。
- 不记录 token、邮件正文、未哈希收件人或 provider 响应正文。

## Phases

1. P1 — 严格识别并丢弃旧 Worker 成功响应 — status: done

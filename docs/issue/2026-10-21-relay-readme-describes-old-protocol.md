# 中继 README 里的 `/v1/deliver` 协议说明已经过时：按文档构造的请求会被 400 拒绝，部署清单也缺少必需的 secret

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 MAIL-06
> 范围：`services/email-relay/README.md`（与 `docs/technical-report.md` 对照） · 严重程度：低 · 状态：未修复 · 置信度：已复现

## 问题

`services/email-relay/README.md` 仍然描述第一代中继，和现在的代码行为不一致：

1. `POST /v1/deliver` 的请求头说明（`README.md:87`）：`Idempotency-Key: YYYY-MM-DD|you@example.com`（optional but recommended）。实际上这个头是**必需**的，并且只接受 `arxiv-daily:auto:<64 位十六进制>` 或 `arxiv-daily:test:<32 位十六进制>` 两种格式（`services/email-relay/src/deliver-logic.ts:30-43,80-88`）。已复现（用中继测试夹具，一次性测试已删除）：用 README 里的 key 或者不带 key 发请求，都得到 `400 {"error":"Idempotency-Key must be a supported bounded auto or test key"}`。按 README 自己写客户端的人（或者想排查问题而手工 curl 的运维）会直接碰壁；README 也没有说明自动 key 与测试 key 走不同的路由与额度。
2. “Concurrency”一节（`README.md:113-118`）说中继先在 KV 里保留 `pending:… → done:…`，并按 Idempotency-Key 选 Durable Object。现在的实现是：自动请求路由到 `recipient-v2:<recipientIdentity>`，测试请求路由到 `device-v2:<deviceIdentity>`（`src/index.ts:248-251`），账本、额度都在 Durable Object 里；旧的 KV `idemp:*` 只作为 cutover 时的历史证据被扫描。
3. Setup 一节（`README.md:31-32`）只列出 `RESEND_API_KEY`、`TOKEN_SECRET` 两个 secret。要让自动投递可用，还需要 `IDENTITY_SECRET`（至少 16 个字符且长期稳定）、`DELIVERY_V2_CUTOVER_TOKEN`（operator 令牌），以及部署时注入的 `BUILD_IDENTITY`，并且要走完 cutover 才会 `ready`（见 `docs/runbook-cutover.md`；`BUILD_IDENTITY` 的部署问题见 MAIL-03）。只按 README 部署会得到一个 `/ready` 永远是 503 的中继：自动投递 503，验证链接 503。
4. 缺少对 `GET /ready`、`GET|POST /internal/delivery-v2/cutover` 的说明，也没有说明 device token 的有效期（见 MAIL-04）。

这些信息只在 `docs/technical-report.md` 和 runbook 里是新的，README 作为服务目录里的第一入口会误导读者。

## 原因

- `services/email-relay/README.md:31-32,53,87,113-118,129`：写于第一代中继，随 v2 cutover（Durable Object 账本、recipient/device 路由、cutover 门禁）没有同步更新。
- 没有任何检查把 README 与代码的协议常量绑定起来（`product-units.json` 的 `compatibilityInputs` 只列了客户端文件，不含 README）。

## 可能的解决方案

推荐：重写 README 的 Setup、API、Concurrency 三节，使之与 `technical-report.md` 的 Worker 小节和 runbook 一致，并在 README 顶部指向 runbook：

- API：列出四个公开/内部路由与各自的认证；`/v1/deliver` 的两种 key 格式、路由方式、状态码表（401/403/409/413/422/429/503/502 与 `ambiguous` 字段的含义）。
- Setup：四个 secret、`BUILD_IDENTITY` 的注入方式、cutover 之前 `/ready` 为 503 是预期。
- Concurrency：改为 Durable Object 账本与 `VerifyRateLimiter` 的现状，删掉 KV `pending→done` 的描述。

备选：把 README 缩成一个指向 `technical-report.md` 对应章节和 runbook 的索引，避免两处各写一份。

## 修复后应补的测试

- 文档一致性测试（可放在 `services/email-relay/tests`）：README 中出现的 `Idempotency-Key` 示例必须通过 `deliveryKeyKind`；README 的 secret 清单与 `scripts/cutover-preflight.mjs` 的 `REQUIRED_SECRET_NAMES` 一致。

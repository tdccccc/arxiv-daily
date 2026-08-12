# Email Relay V2 生产 Cutover Runbook

单用户、单版本、一次性迁移操作清单。**每一步都是人工检查点**：只执行你已明确授权、且已在下方确认的操作；任何结果不明（网络超时、响应丢失、状态异常）时停止，先通过只读 `GET /internal/delivery-v2/cutover` 确认实际状态，再决定下一步，绝不盲目重试。

- 服务：Cloudflare Worker `arxiv-daily-email-relay`（`services/email-relay/`）
- 公开地址：`https://mail.arxiv-daily.top`
- operator 令牌：`DELIVERY_V2_CUTOVER_TOKEN`（Worker secret，隐藏输入，不写入 argv/日志/PR）
- 所有 mutation（部署、凭据、cutover action、验证、邮件）都发生在工具之外，由你手工执行。

## 0. 准备

```bash
cd services/email-relay
npm ci
```

- 确认要部署的精确 commit SHA：`git rev-parse HEAD`（40 位 hex）。该 SHA 会成为永久 cutover build binding，之后不能静默更换。
- 确认 `IDENTITY_SECRET` 与 `TOKEN_SECRET`：前者必须长期稳定（轮换会锁定 automatic 并丢失已签发 token 的 identity），后者可轮换。
- 准备一个新的 64 位 hex 操作 ID（每次 action 一个，用于幂等收敛）：`openssl rand -hex 32`。

## 1. Preflight（只读）

```bash
npm run preflight            # 本地检查 + 远程 /health、/ready
node scripts/cutover-preflight.mjs --skip-login   # 无 wrangler 登录环境时
node scripts/cutover-preflight.mjs --check-readonly  # 静态自检（CI 亦会执行）
```

逐项确认：

- `gitHead`、`wranglerToml`、`wranglerDryRun` 必须 PASS；
- `wranglerLogin` 未登录时先 `npx wrangler login`；
- `wranglerSecrets` 必须列出全部四个 secret 名称：`RESEND_API_KEY`、`TOKEN_SECRET`、`IDENTITY_SECRET`、`DELIVERY_V2_CUTOVER_TOKEN`；
- `remoteReady` 为 **locked (503)** 是预期初始状态；若为其他失败，先修复再继续。

任何 FAIL 未解决前，不进行第 2 步。

## 2. 部署新 Worker（以 automatic-locked 启动）

```bash
npx wrangler deploy --var "BUILD_IDENTITY:email-relay-v2-$(git rev-parse HEAD)"
```

- 部署后 `GET https://mail.arxiv-daily.top/ready` 必须返回 503 locked（新 Worker 不允许直接开放 automatic）。
- **确认点**：只读 status 显示 `phase: locked` 或 `inventoried` 之前的状态；若显示 ready，停止并调查（不允许在未完成 cutover 前 ready）。

## 3. 撤销旧 Resend 凭据（人工检查点）

在 Resend/凭据管理侧撤销旧 Worker 使用的 `RESEND_API_KEY`。撤销完成是建立跨 Worker provider fence 的前提；只有你能确认撤销事实，工具无法替你证明。

- **确认点**：确认旧凭据已失效（例如尝试一次已知会失败的调用），再继续。

## 4. 配置新凭据

```bash
npx wrangler secret put RESEND_API_KEY            # 新凭据，只交给仍锁 automatic 的新 Worker
npx wrangler secret put TOKEN_SECRET
npx wrangler secret put IDENTITY_SECRET           # 长期稳定
npx wrangler secret put DELIVERY_V2_CUTOVER_TOKEN # operator 令牌，隐藏输入
```

- **确认点**：`npx wrangler secret list` 名称齐全；不把 secret 值写入任何文件、argv、日志或 PR。

## 5. Cutover actions（每步一个 64-hex operationId，逐步确认）

只读查询：

```bash
curl -s -H "Authorization: Bearer $CUTOVER_TOKEN" \
  https://mail.arxiv-daily.top/internal/delivery-v2/cutover
```

按顺序执行（每个 action 单独确认后再发下一个）：

```bash
# 1) 盘点 legacy evidence
curl -s -X POST -H "Authorization: Bearer $CUTOVER_TOKEN" -H "Content-Type: application/json" \
  -d '{"action":"inventory","operationId":"<64-hex-1>"}' \
  https://mail.arxiv-daily.top/internal/delivery-v2/cutover

# 2) 声明 provider fence（必须精确引用 attestation 短语）
curl -s -X POST -H "Authorization: Bearer $CUTOVER_TOKEN" -H "Content-Type: application/json" \
  -d '{"action":"provider-fence","operationId":"<64-hex-2>","attestation":"old-resend-credential-revoked"}' \
  https://mail.arxiv-daily.top/internal/delivery-v2/cutover

# 3) observe 第一次：扫描 legacy evidence 并写入 audit marker（进入 sealed）
curl -s -X POST -H "Authorization: Bearer $CUTOVER_TOKEN" -H "Content-Type: application/json" \
  -d '{"action":"observe","operationId":"<64-hex-3>"}' \
  https://mail.arxiv-daily.top/internal/delivery-v2/cutover

# 4) observe 第二次：marker 观察，距上一次 ≥60 秒（seal 要求至少 2 次观察）
sleep 60
curl -s -X POST -H "Authorization: Bearer $CUTOVER_TOKEN" -H "Content-Type: application/json" \
  -d '{"action":"observe","operationId":"<64-hex-4>"}' \
  https://mail.arxiv-daily.top/internal/delivery-v2/cutover

# 5) observe 第三次：再次观察（每次间隔 ≥60 秒）
sleep 60
curl -s -X POST -H "Authorization: Bearer $CUTOVER_TOKEN" -H "Content-Type: application/json" \
  -d '{"action":"observe","operationId":"<64-hex-5>"}' \
  https://mail.arxiv-daily.top/internal/delivery-v2/cutover

# 6) seal：观察次数满足后直接进入 ready
curl -s -X POST -H "Authorization: Bearer $CUTOVER_TOKEN" -H "Content-Type: application/json" \
  -d '{"action":"seal","operationId":"<64-hex-6>"}' \
  https://mail.arxiv-daily.top/internal/delivery-v2/cutover
```

- 每个 action 响应丢失或结果不明时，用上面的只读 GET 查询实际 phase；已完成的 action 不必重发（幂等收敛于 operationId）。
- **确认点**：每步 status 中的 phase 依次为 `inventoried → observing → sealed → ready`；`seal` 前确认 observations 已达 2 次以上且每次间隔 ≥60 秒；任何一步出现 `blocked`/`identity-locked` 或 binding 不一致，停止并走第 8 节。

## 6. Readiness 确认

```bash
curl -s https://mail.arxiv-daily.top/ready
```

必须返回 200，且 `automatic: "ready"`、`phase: "ready"`、`readyGeneration > 0`、`buildIdentity` 与第 2 步部署的 SHA 一致。

## 7. 工具外人工验证

- 触发一次验证邮件（插件设置页或 `/v1/verify/start`），打开 magic link 获取 device token，填入插件；
- 用插件发一封测试邮件，确认收件与内容正确；
- 确认 `delivery-state.json` 与插件侧无重复/阻断状态。

## 8. 异常与恢复

- **任何 fail closed**（`identity-locked`、binding 不匹配、marker 不一致、结果不明）：停止所有操作，只读查询 status 与 `/ready`，记录现场，按 fix-forward 修复（如重新注入正确 build identity、repair action），不盲目重试 mutation。
- **恢复**：所有恢复路径都以只读 status 判定为先；不能通过状态确认的操作一律视为未完成，需要你重新授权。
- **回滚**：不支持自动 rollback；旧 Worker 的凭据已撤销，恢复的唯一路径是 fix-forward 完成 cutover。

## CI 边界

CI（`.github/workflows/email-relay.yml`）只执行测试、typecheck 和 Wrangler `--dry-run`，并运行 `cutover-preflight.mjs --check-readonly` 证明 preflight 脚本只读。任何 CI 步骤都不部署、不读写生产 KV、不调用验证/投递端点。

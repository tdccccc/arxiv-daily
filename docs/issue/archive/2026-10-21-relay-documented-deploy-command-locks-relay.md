# 中继文档里写的部署命令 `npm run deploy` 会丢掉 `BUILD_IDENTITY`，部署后验证邮件签发和所有自动日报邮件全部失效

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 MAIL-03
> 范围：`services/email-relay` 部署与升级流程（README、`package.json`、cutover runbook） · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-21-batch3`，提交 288e072） · 置信度：已复现

## 问题

中继（Official delivery）现在要求 Worker 变量 `BUILD_IDENTITY` 必须是 `email-relay-v2-<40 位小写十六进制提交 SHA>`，并且和第一次 cutover 时写入 Durable Object 的“永久 binding”一致。缺失或不一致时，中继整体锁定：

- 所有 `Idempotency-Key: arxiv-daily:auto:…` 的自动日报投递返回 `503 {"error":"automatic delivery is unavailable"|"delivery cutover is not ready","ambiguous":false}`；
- 用户点击邮件里的验证链接，得到 `503 Verification unavailable`（拿不到 device token）；
- `/ready` 返回 `503 {"automatic":"locked"}`。

但仓库里写的部署方式并不会设置它：

- `services/email-relay/README.md:53` 的“Deploy”一节只有 `npm run deploy`；`package.json:9` 的脚本就是裸 `wrangler deploy`；README 的 Setup 一节（`:31-32`）也只列出 `RESEND_API_KEY`、`TOKEN_SECRET` 两个 secret，没有 `IDENTITY_SECRET`、`DELIVERY_V2_CUTOVER_TOKEN`，也没有提 `BUILD_IDENTITY`。
- `wrangler.toml` 的 `[vars]` 里没有 `BUILD_IDENTITY`（有意如此，见 `docs/technical-report.md:383`）。
- 上一轮审查归档 `docs/issue/archive/2026-10-18-relay-verify-rate-limit-non-atomic.md:35` 的“只有用户部署后才生效”段落，明确告诉用户“在 `services/email-relay` 下运行 `npm run deploy`”来上线验证码限流的 Durable Object 修复。
- 只有 `docs/runbook-cutover.md:41` 写了 `npx wrangler deploy --var "BUILD_IDENTITY:email-relay-v2-$(git rev-parse HEAD)"`。

Wrangler 的行为是：不加 `--keep-vars` 时，每次 `wrangler deploy` 会先删除所有 vars，再设置配置文件和命令行里给出的那些（Cloudflare 文档对 `--keep-vars` 的说明）。所以先用 `--var` 部署过一次的人，只要之后按 README 或归档说明跑一次 `npm run deploy`，`BUILD_IDENTITY` 就被删掉了。

说明：“裸 `wrangler deploy` 会删除 `--var` 注入过的变量”这一步依据 Cloudflare 文档和仓库自己的 runbook（它特意用 `--var` 注入），没有在真实部署上验证（部署是用户自己的操作，审查中不触碰 Cloudflare）；下面的“缺失/不一致后中继的表现”则已在测试夹具里实际跑过。

已用中继的测试夹具复现（一次性测试，已删除；`cutover` 已走到 `ready`，device token 有效）：

1. 去掉 `env.BUILD_IDENTITY` 后：自动投递 `503 {"error":"automatic delivery is unavailable","ambiguous":false}`；`GET /v1/verify?token=<有效 pending token>` 返回 `503`、页面标题 “Verification unavailable”；`/ready` 返回 `503 {"automatic":"locked"}`。
2. 把 `BUILD_IDENTITY` 换成另一个合法值（`email-relay-v2-` + 另一个 40 位 SHA，即“发布新版本并按 runbook 注入新 SHA”）：自动投递 `503 {"error":"delivery cutover is not ready","ambiguous":false}`，`inventory`、`provider-fence`、`observe`、`seal`、`repair` 五个 cutover action 全部返回 `503` 且 `phase: "locked"`，状态查询也是 `503 locked`——没有任何 action 能重新绑定。只有把原来的 SHA 字符串重新注入，才恢复（实测 `200`）。
3. 上面两种锁定状态下，**测试邮件仍然成功**：`Idempotency-Key: arxiv-daily:test:…` 的请求返回 `200 {"ok":true}`（测试 key 不检查 cutover 就绪状态）。所以部署后用插件的“Send test email”按钮做冒烟验证会通过，而真正的自动日报和新用户验证全部不可用，问题要等到第二天没收到邮件才会发现。

叠加的后果：客户端对 `503` 一律记成“结果不明”（见 MAIL-02），锁定期间每个 hosted 用户当天的邮件都会被永久阻断，即使中继随后修好也不会补发。

另外，“永久 binding”意味着中继的任何后续代码发布都面临同一个问题：runbook 只描述了一次性 cutover（`runbook-cutover.md:17` 说该 SHA 之后不能静默更换，`:128` 的“fix-forward”只提到重新注入正确的 build identity），没有说明 cutover 之后怎样发布新代码——要么注入当初绑定的那个旧 SHA（此时 `BUILD_IDENTITY` 不再代表部署的代码），要么永久锁死。

## 原因

- `services/email-relay/src/kv.ts:390-416` `configuredBuildIdentity` / `automaticRuntimeConfigured`：缺 `BUILD_IDENTITY` 就认为自动投递未配置；`src/index.ts:188-193`（自动投递 503）、`src/cutover-control.ts:585`（`issueDevice` 返回 locked）依赖它。
- `src/cutover-control.ts:980-986`：binding 与当前 `BUILD_IDENTITY` 不一致时，所有 cutover action（包括 `repair`）直接 `identityLockedStatusResponse`，没有重新绑定的路径。
- `services/email-relay/README.md:31-32,53`、`package.json:9`：文档和脚本没有 `BUILD_IDENTITY`，与 runbook 脱节。
- `docs/issue/archive/2026-10-18-relay-verify-rate-limit-non-atomic.md:35`：给出的上线命令会触发这个问题。

## 可能的解决方案

推荐：

1. 把 `package.json` 的 `deploy` 脚本改成带注入的版本，例如 `node scripts/deploy-with-identity.mjs`：先要求工作区干净、读取 `git rev-parse HEAD`，再执行 `wrangler deploy --var BUILD_IDENTITY:email-relay-v2-<sha>`；并且在 cutover 已绑定后，先对比 `GET /ready`（或只读 status）里的 `buildIdentity`，不一致时拒绝部署并提示走 runbook。裸 `wrangler deploy` 不再是文档里的写法。
2. README 的 Setup/Deploy 一节补全四个 secret 和 `BUILD_IDENTITY`，并指向 `docs/runbook-cutover.md`；归档说明里的 `npm run deploy` 在新增的“部署后”清单里改为 runbook 的命令。
3. 在 runbook 里补一节“cutover 之后发布新代码”：明确两种做法之一并给出验证步骤——(a) 约定 `BUILD_IDENTITY` 只标识“协议/ledger 世代”，发布新代码时继续注入同一个字符串；(b) 增加一个受 operator token 保护、需要二次确认的 `rebind` action，把新的 build identity 写入 binding 并重新走 observe/seal。
4. 部署后的冒烟清单必须包含自动 key 的投递（或 `GET /ready` 为 200、`readyGeneration > 0`），不能只做测试邮件。

备选：把 `BUILD_IDENTITY` 写进 `wrangler.toml` 的 `[vars]`，由 CI 在部署时替换；缺点是仓库里会出现必须手动同步的占位值。

## 修复后应补的测试

- 部署脚本测试：工作区不干净、HEAD 不是 40 位 SHA、已绑定的 identity 与当前 HEAD 不一致时，脚本拒绝部署，且没有调用 `wrangler deploy`。
- 中继：`BUILD_IDENTITY` 缺失时 `/ready` 为 503 的同时，`GET /health` 仍为 200（现有行为，保持）；自动 key 与验证链接的 503 响应里带上稳定的、可区分“未配置 build identity”和“与 binding 不一致”的原因码（不泄露值），便于运维定位。
- 如果实现 `rebind`：缺少二次确认、缺少 operator token、build identity 格式错误时都拒绝；成功后需要重新经过 observe/seal 才 `ready`。
- 文档检查：`README.md` 的部署命令与 `docs/runbook-cutover.md` 一致（可加一个简单的脚本测试检查 README 不含裸 `wrangler deploy`）。

## 修复情况

`npm run deploy` 不再丢掉 `BUILD_IDENTITY`：它现在运行 `services/email-relay/scripts/deploy-with-identity.mjs`，要求 `services/email-relay` 工作区干净（含未跟踪文件）、`HEAD` 是 40 位 SHA，读取线上 `GET /ready`——中继已经绑定到另一个 identity 时拒绝（binding 是永久的，换了就会整体锁定）并打印应传给 `--identity` 的值——然后只执行一次 `wrangler deploy --var BUILD_IDENTITY:email-relay-v2-<sha>`，不需要任何 secret。读不到 `/ready`（不可达、已被锁定）时它只能警告，此时把 cutover 时记下的 identity 传给 `--identity`。README 与 runbook 不再出现不带 `--var` 的 `wrangler deploy` 命令；runbook 新增第 9 节“cutover 之后发布新代码”，采用写作里的方案 (a)：`BUILD_IDENTITY` 只标识 cutover 时绑定的协议/ledger 世代，之后发布继续注入它；没有实现 `rebind` action（那是协议变更）。`/ready` 在读不到 cutover 状态时多返回一个只有名字、不含值的 `reason`（`build_identity_missing`、`runtime_not_configured`、`cutover_locked`），preflight 会把它打印出来。

偏离写作的一点：没有给 `/v1/deliver` 的 503 响应加原因码，因为已安装的客户端只对恰为 `{error, ambiguous:false}` 的 503 判定“未发送”，多一个成员就会让它们把当天记成结果不明；诊断放在 `/ready`。需要用户自己部署：`npm run deploy` 脚本本身直接可用，`/ready` 的 `reason` 要重新部署中继之后才有。

测试：`services/email-relay/tests/deploy-with-identity.test.ts`（工作区不干净、HEAD 不是 40 位 SHA、与已绑定 identity 不一致时拒绝且没有调用 `wrangler deploy`；匹配时恰好一次且 `--var` 是唯一变量；`--identity`、`--dry-run`、`/ready` 读不到时的警告）、`readme-contract.test.ts`（文档命令里没有裸 `wrangler deploy`，`package.json` 的 `deploy` 指向该脚本）、`kv-idemp.test.ts`（`/ready` 的 `reason`，缺失 BUILD_IDENTITY 时 `/health` 仍为 200，恢复后 `/ready` 回到 200）、`cutover-preflight.test.ts`。其中 `/ready` 的 6 个用例在旧代码下失败。

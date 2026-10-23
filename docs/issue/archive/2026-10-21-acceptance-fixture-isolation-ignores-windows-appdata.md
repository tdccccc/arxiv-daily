# 在 Windows 上跑 workbench 验收或探索，隔离夹具不生效：产品会读写用户真实的 CLI 配置

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 SCR-05
> 范围：`scripts/acceptance/fixtures.mjs`、`workbench-session.mjs`、`run.mjs`（workbench 固定套件与 workbench 探索宿主） · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-21-batch4`，提交 f1faa04） · 置信度：代码推断

## 问题

验收夹具靠把 `XDG_CONFIG_HOME` / `XDG_CACHE_HOME` 指向临时目录来隔离 CLI 配置，并把夹具配置写在 `<root>/config/arxiv-daily/config.toml`。但 CLI 在 Windows 上完全不看 `XDG_CONFIG_HOME`，而是读 `%APPDATA%\arxiv-daily\config.toml`（`resolveCliConfigDir` 的 `win32` 分支）。夹具环境只覆盖了 XDG 变量，`APPDATA` 来自 `{ ...process.env }`，仍是开发者真实的目录。所以在 Windows 上：

- 被拉起的 workbench（`node apps/cli/dist/arxiv-daily-cli.cjs ui --no-open`）读到的是用户真实配置，包括真实的 `vault_root`、模型地址和密钥；夹具里写的那份配置没有被产品使用，只是被驱动用来做校验。
- 固定验收套件和模型探索对界面做的操作会写进真实配置：例如 `settings-persistence` 任务把“每日论文数量上限”改掉并等待保存，模型探索里的任何设置保存，都落在用户真实的 `config.toml` 上；点“生成”则会用夹具的假论文在用户真实的 Vault 里生成日报（所有 `fetch` 仍会被预加载脚本转给夹具服务器，所以不会真的访问外网，但文件写进了真实 Vault）。
- 驱动自己的校验读的是夹具那份配置，完全没变，于是任务以“验证失败/阻塞”结束，报告看起来只是“测试挂了”，没有任何提示说明真实数据已经被改过。
- `workbench-session.mjs` 里唯一的隔离检查是 `fixture.env.XDG_CONFIG_HOME === fixture.configHome`，在 Windows 上这个检查通过，给出了错误的安全感。

文档承诺“两个产品都在隔离的临时配置、Vault、缓存和端口中运行，不碰用户的真实数据”（`scripts/acceptance/README.md:57`），并且专门写了 Windows 的配置路径（README 第 39 行），`run.mjs` 里也有 `win32` 分支，说明作者认为 Windows 是支持的；CI 只在 ubuntu 上跑固定 workbench 套件，所以没人会发现。

用真实函数复现路径解析（不启动任何产品）：用夹具的真实环境（只含 XDG 覆盖）再加一个 Windows 开发机的 `APPDATA`，`resolveCliConfigPath(env, "win32")` 得到 `C:\Users\dev\AppData\Roaming/arxiv-daily/config.toml`，而 Linux/macOS 下得到的是夹具目录里的那份。

同一条路径上还有一个相关的 Windows 缺口（未验证）：`run.mjs` 构建步骤用 `spawn("npm.cmd", …)` 且不带 `shell: true`，在 Node 20.12 之后的版本里 Windows 对 `.cmd` 的无 shell 启动会直接抛 `EINVAL`；同一个仓库里的 `install-smoke.mjs` 已经为此绕开（用 `process.execPath` 加 `npm_execpath`）。所以 Windows 上默认带构建的验收根本跑不起来，开发者很可能会用 `--skip-build` 绕过去，正好走到上面那条危险路径。

## 原因

- `scripts/acceptance/fixtures.mjs:216`：夹具环境 `{ ...process.env, XDG_CONFIG_HOME, XDG_CACHE_HOME, ARXIV_ACCEPTANCE_FIXTURE_URL, NODE_OPTIONS }`，没有设置 `APPDATA`（也没有设置 `HOME`/`USERPROFILE`）。
- `apps/cli/src/config-path.ts:5-15`：`win32` 时只用 `APPDATA`（缺省 `~/AppData/Roaming`）。`apps/cli/src/config.ts:86`、`apps/cli/src/workbench/launch.ts:35` 都用这个函数。
- `scripts/acceptance/workbench-session.mjs:56-58`：隔离检查只看 `XDG_CONFIG_HOME`，没有用 CLI 自己的 `resolveCliConfigPath(fixture.env)` 去核对“产品实际会读哪一个文件”。
- `scripts/acceptance/run.mjs:125`：Windows 下 `spawn("npm.cmd")`。

## 可能的解决方案

推荐：让夹具环境同时隔离 Windows 的位置（`APPDATA`、`LOCALAPPDATA`、`USERPROFILE`/`HOME` 都指向夹具根目录下的目录），并把 `workbench-session.mjs` 的隔离检查改成“用 CLI 的 `resolveCliConfigPath(fixture.env)` 解析出的文件必须等于 `fixture.configPath`”，不相等就按 blocked 拒绝启动。这样任何一个平台上配置位置规则变化，守卫都会跟着失败，而不是继续放行。

另外：要么把 `run.mjs` 的构建改成和 `install-smoke.mjs` 一样通过 `process.execPath` 调 npm，要么在 README 明确写“工作台验收和探索目前只在 Linux/macOS 验证，Windows 上会被拒绝运行”，并让 `run.mjs` 在 `process.platform === "win32"` 时直接给出 blocked，避免走到上面的危险路径。

备选：只加文档说明，不改代码。风险是文档已经写了 Windows 路径，使用者会以为支持。

## 修复后应补的测试

- `scripts/tests/acceptance-fixtures.test.mjs`：断言 `createFixtureEnvironment().env` 在 `win32` 解析下 `resolveCliConfigPath(env, "win32")` 落在夹具根目录之内（可用注入的平台参数在 Linux 上跑）。
- `scripts/tests/acceptance-workbench.test.mjs`：当 `resolveCliConfigPath(fixture.env)` 指向夹具之外时，`startWorkbenchSession` 抛出 `WorkbenchBlockedError`，不启动子进程。

## 修复情况

Windows、macOS、Linux 验收都把实际配置、HOME 和缓存指向临时夹具；启动前用真实 CLI 路径规则核对配置位置，不匹配就拒绝启动。Windows 构建通过 Node 启动 npm，避免直接执行 cmd 的限制。2 项失败回归修复后，40 项相关测试与 CLI 类型检查通过。

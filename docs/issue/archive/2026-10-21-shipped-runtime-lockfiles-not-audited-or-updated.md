# 随 CLI 安装到用户机器上的 PDF/向量运行时依赖，既没有依赖更新提醒也不在 CI 的漏洞审计里

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 SCR-06
> 范围：`.github/dependabot.yml`、各工作流里的 `npm audit`、`scripts/tests/security-ci.test.mjs`、`tools/node-library-runtime/**`、`extensions/dsh-arxiv-daily` · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-21-batch4`，提交 759918d、8f3e862、b667688、a258e51） · 置信度：代码推断

## 问题

仓库里有 6 个被提交的 `package-lock.json`：根目录、`services/email-relay`、`extensions/vscode-arxiv-daily`、`extensions/dsh-arxiv-daily`、`tools/node-library-runtime/pdf`、`tools/node-library-runtime/embedding`。依赖安全这条线只覆盖了其中三个：

- Dependabot（`.github/dependabot.yml`）只配置了 `/`、`/services/email-relay`、`/extensions/vscode-arxiv-daily` 和 GitHub Actions；
- `npm audit --audit-level=moderate` 只在根目录（`lint.yml`、`release.yml`、`publish-cli.yml`、`publish-dsh-plugin.yml`）和邮件中继（`email-relay.yml`）里跑。

没有任何一条覆盖到的三个（前两个是随产品发布、在用户机器上安装并运行的依赖，第三个只是测试依赖）：

1. `tools/node-library-runtime/pdf`：固定 `pdfjs-dist` 5.4.624。这是 `arxiv-daily library prepare` 在用户机器上用 `npm ci --ignore-scripts` 安装、并用来解析用户本地 PDF 的解析器（`tools/node-library-runtime/README.md`）。
2. `tools/node-library-runtime/embedding`：固定 `@huggingface/transformers` 4.2.0，锁文件里带着真实的 `onnxruntime-node@1.24.3` 和 `sharp@0.34.5`。注意根目录的 `package.json` 用 `overrides` 把这两个包替换成了空包，所以根目录的 `npm audit` 在原理上就看不到实际随产品安装的这两个原生包。
3. `extensions/dsh-arxiv-daily`：它的锁文件只管测试用的 `react`/`react-dom`（客户端代码在运行时用宿主提供的 React，没有打包），风险最低，但同样没有任何更新或审计覆盖。

这些依赖都是精确版本固定的，没有 Dependabot 就永远不会自己升级；没有审计，针对它们的安全公告（PDF 解析器是这类公告的常见目标）不会让任何 CI 变红，也不会出现任何提醒。结果是发版门槛“`npm audit` 通过”的含义被高估：它说明被审计的三个锁文件没问题，而最直接面对不可信输入（用户的 PDF 文件）的依赖恰好在审计之外。

`scripts/tests/security-ci.test.mjs:31-51` 的测试名叫“Dependabot covers every lockfile and GitHub Actions”，但断言里把四个目录写死，所以它守住的是现状而不是“每个锁文件”。

## 原因

- `.github/dependabot.yml`：只有 4 个 `updates` 项，没有 `/extensions/dsh-arxiv-daily`、`/tools/node-library-runtime/pdf`、`/tools/node-library-runtime/embedding`。
- `.github/workflows/*.yml`：`npm audit` 全部在仓库根目录或 `services/email-relay` 下执行；没有任何步骤对其余锁文件执行 `npm audit --prefix …`。
- `scripts/check-product-units.mjs` 的扫描根目录是 `packages`、`apps`、`plugin`、`services`、`extensions`，`tools/` 不在其中，所以治理清单（`product-units.json`）也不会提示“有一个带锁文件的目录没人管”。

## 可能的解决方案

推荐：

1. 在 `dependabot.yml` 补上三个目录（优先 `/tools/node-library-runtime/pdf` 和 `/tools/node-library-runtime/embedding`，其次 `/extensions/dsh-arxiv-daily`）。运行时锁文件升级后需要按 `tools/node-library-runtime/README.md` 同步更新带版本号的运行时路径和烟雾测试，可以把这两个目录的更新设为只提醒（`open-pull-requests-limit` 小一些），由人工确认。
2. 在 `lint.yml` 的根目录审计之后，对这三个锁文件各加一步 `npm audit --audit-level=moderate --prefix <目录>`（只读，不安装）。
3. 把 `security-ci.test.mjs` 改成动态检查：用 `git ls-files '**/package-lock.json'` 枚举所有锁文件，断言每个所在目录都在 `dependabot.yml` 里，并且每个都在某个工作流的 `npm audit` 步骤里出现；确实不需要的目录要写进带理由的例外表。

备选：只做第 1 点。能收到升级提醒，但漏洞公告不会让 CI 失败。

## 修复后应补的测试

- 上述动态测试：新增一个带锁文件的目录而没有配置 Dependabot 或审计时，测试失败并指出目录名。
- 例外表里的每一项必须带理由，空理由被拒绝。

## 修复情况

CI 从 Git 跟踪的全部 package-lock.json 动态生成审计清单，Dependabot 覆盖同一集合的测试不再写死目录。新增 DSH、PDF 和真实 CPU 向量运行时更新提醒；任何锁的漏洞都让 CI 失败，只有注册表临时错误可有限重试。

覆盖扩展后实际发现 VS Code 传递依赖 6 个公告、本地向量运行时 4 个高危公告，分别用兼容传递更新与 Transformers 4.3.1/ONNX 1.30.0/sharp 0.35.5/adm-zip 0.6.1 消除。向量运行时使用新缓存目录，模型权重位置和 384 维 q8 约定不变；浮点舍入和排序不保证逐位一致，既有索引不自动重建。

KaTeX 固定到 2026-10-01 发布的 0.19.0并更新声明。2 项覆盖回归先失败；6 项安全 CI 测试、68 项公式/阅读界面测试、全部 CSS 字体资源构建检查通过；6 份锁文件审计均为 0 漏洞。实际 CPU 推理与真实浏览器公式视觉对照由集成验收单独执行，合并仍以其结果为准。KaTeX 的命令行依赖 commander 要求 Node 22.12，但不进入产品 bundle，Node 20.19 构建和产品运行仍纳入兼容性门禁。

原生验收补充：使用此次锁文件安装真实运行时，并复用任务临时缓存中的公开 multilingual-e5-small q8 权重，在无外网的独立网络空间分别用 Node 22.22.2 和 Node 20.19.0 执行产品实际加载器：PDF夹具正文解析正确，两条固定文本均得到384维有限数且L2范数约1。Node 20发行包已对照官方SHA-256校验。未读取用户文献库或调用真实模型API。

浏览器验收补充：升级前后用同一份含行内公式、分式、矩阵、代码和美元金额的日报进行真实工作台对照；KaTeX0.19在Node22和Node20.19宿主上均显示3个公式/3个MathML结构、正确加载KaTeX字体，无页面或请求错误，截图目视无回归。所有浏览器与CLI均在隔离网络和临时配置中运行。

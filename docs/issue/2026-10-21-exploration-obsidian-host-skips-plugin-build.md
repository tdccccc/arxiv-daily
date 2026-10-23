# 文档里的发版探索命令不构建 Obsidian 插件，会悄悄探索一份过期的插件

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 SCR-01
> 范围：`scripts/acceptance/run.mjs`、Obsidian 探索宿主（`exploration-obsidian.mjs` → `desktop-acceptance` 部署） · 严重程度：中 · 状态：未修复 · 置信度：已复现

## 问题

`scripts/acceptance/README.md` 里的发版命令是：

```sh
npm run test:acceptance -- --suite exploration --host obsidian --release
```

这条命令（以及日常调试用的 `--suite exploration --host obsidian`）**只构建 `apps/cli`，不构建 `obsidian-arxiv-daily` 插件**。随后 Obsidian 探索会把 `plugin/main.js`、`plugin/manifest.json`、`plugin/styles.css` 当前磁盘上的内容原样部署到临时 Vault 里探索。如果这三个文件是上一次（甚至上一个分支）构建出来的，整次发版探索就是在一份过期插件上跑的：

- 运行结果可以是全绿、退出码 0，报告里只有 `buildRequested: true`，看不出实际探索的是哪一份插件。
- 部署后的“版本守卫”（`assertVersionUnderTest`）只比较 `manifest.json` 的版本号字符串，而期望值和实际值都来自刚部署进去的同一份 `manifest.json`，所以只要版本号没变（日常开发中几乎总是如此），过期的 `main.js` 照样通过。
- 同样的缺口也出现在 `--explore --suite workbench --host obsidian` 这类组合里：只要没有选中固定的 `obsidian` 套件，插件就不会被构建。

实际发生过：协调者在这台机器上发现 `plugin/main.js` 还是 2026-10-07 的构建，而插件源码直到 2026-10-18 还在变化；如果当时直接跑文档里的发版命令，Obsidian 一端探索的就是 11 天前的插件，发版证据却会被当成“当前代码”的证据。（2026-10-21 当天的探索前协调者已手动重新构建，当天证据不受影响。）

复现（不构建任何东西：用一个只记录参数的假 `npm` 放在 PATH 最前面，运行真实的 `parseArgs` + `runAcceptance` + `buildProducts`）：

| 命令行 | 解析出的套件 | 实际执行的构建 |
|---|---|---|
| `--suite exploration --host obsidian --release` | `["exploration"]` | 只有 `npm run build --workspace apps/cli` |
| `--explore --suite workbench --host obsidian` | `["workbench","exploration"]` | 只有 `apps/cli` |
| `--explore --host obsidian` | `["workbench","obsidian","exploration"]` | `apps/cli` 和 `obsidian-arxiv-daily`（因为默认套件里带了 `obsidian`） |
| `--suite exploration --host workbench` | `["exploration"]` | `apps/cli`（正确） |

## 原因

- `scripts/acceptance/run.mjs:116-119`：`buildProducts(suites, …)` 只根据套件名选择工作区——`workbench` 或 `exploration` 就构建 `apps/cli`，只有套件名里有 `obsidian` 才构建 `obsidian-arxiv-daily`。探索套件的宿主（`--host`）没有传进来，`run.mjs:196` 调用 `build(selected, …)` 时也没有传。
- `scripts/acceptance/exploration-obsidian.mjs:813`、`scripts/desktop-acceptance/preflight.mjs:186-194`：Obsidian 探索把 `plugin/` 当作构建产物目录，预检只检查三个文件“存在”，不检查它们是否比源码新。
- `scripts/desktop-acceptance/build-deploy.mjs:62-70,76-88`、`session.mjs:74,139`：版本守卫的期望值读自被部署的 `manifest.json`，实际值是插件从同一份 `manifest.json` 报告的版本，因此对“`main.js` 过期”没有任何检测能力。
- `scripts/acceptance/README.md:49-65`：发版命令和前置条件里没有提到需要先构建插件，使用者无从得知这条命令不会替他构建。

## 可能的解决方案

推荐：让 `buildProducts` 同时按套件和探索宿主选择工作区——选中 `exploration` 且 `--host obsidian` 时构建 `obsidian-arxiv-daily`（`--host workbench` 时构建 `apps/cli`；`--explore` 同理）。把这条选择逻辑抽成一个可单测的纯函数（例如 `workspacesToBuild(suites, host)`），`runAcceptance` 把 `options.host` 传给 `build`。

补强（避免 `--skip-build` 或手工构建带来的同类问题；`npm run test:desktop` 和独立脚本 `scripts/acceptance/obsidian-library-change.mjs` 同样直接部署磁盘上的 `plugin/` 而不构建，所以检查应放在三者共用的 `deployBuildUnderTest` 里）：在 `report.json` 的 `metadata` 里记录被探索产物的指纹（`plugin/main.js`、`manifest.json`、`styles.css` 以及 `apps/cli/dist/arxiv-daily-cli.cjs` 的 sha256 和修改时间），这样发版证据能说明“探索的到底是哪一份构建”；`--skip-build` 时若 `plugin/main.js` 比 `plugin/main.ts`、`plugin/src/**`、`packages/core/src/**` 中最新的文件还旧，打印醒目的警告。

备选：只改文档，在发版命令前加一行 `npm run build`。代价是仍然依赖人记住，而且 `--explore --suite workbench --host obsidian` 这类组合不会被提醒到。

## 修复后应补的测试

- 在 `scripts/tests/acceptance-runner.test.mjs` 中断言：`--suite exploration --host obsidian` 和 `--explore --suite workbench --host obsidian` 会要求构建 `obsidian-arxiv-daily`；`--suite exploration --host workbench` 不会；固定的 `obsidian` 套件行为不变。
- 断言 `--skip-build` 时不触发任何构建，并且报告元数据里带有产物指纹。
- 若加入过期检测：构造一个比源码旧的 `plugin/main.js`，断言 `--skip-build` 的 Obsidian 探索会给出警告（或按约定拒绝运行）。

# 功能与交互自动验收

运行真实 Reading workbench 和 Obsidian，检查用户操作、页面反馈及保存结果。论文、业务模型和邮件接口使用固定响应；这里不评估筛选相关性或摘要质量。

## 运行

使用 Node.js 22，先安装开发依赖：

```sh
npm ci
npx playwright install chromium
```

工作台优先使用本机 Chrome/Chromium，也支持 Playwright 安装的浏览器。可用 `--browser /path/to/browser` 或 `ARXIV_DAILY_ACCEPTANCE_BROWSER` 指定可执行文件。

完整桌面验收目前在 Linux 验证，需要 Obsidian、`xvfb-run`、`Xvfb` 和 `xauth`。Obsidian 默认路径为 `/opt/Obsidian/obsidian`，可通过 `OBSIDIAN_BINARY` 指定。无需准备或指定自己的 Vault。

```sh
# 构建当前工作树，运行两个产品的固定界面验收
npm run test:acceptance

# 只验证独立工作台
npm run test:acceptance -- --suite workbench

# 只验证 Obsidian
npm run test:acceptance -- --suite obsidian

# 查看选项及可用套件
npm run test:acceptance -- --help
npm run test:acceptance -- --list
```

每次运行创建独立的临时配置、Vault、缓存和 PDF。工作台测试操作已打包 CLI，并使用真实服务端、生成子进程、调度器、pipeline 和文件存储；Obsidian 测试部署当前构建到临时 Vault，并使用真实插件和宿主。

`--skip-build` 仅用于本地调试已有构建；正式验收保留默认构建步骤。`--headed` 显示工作台浏览器。`--output PATH` 指定新的或空的证据目录，已有证据不会被混入下一次运行。

## 模型 API 探索

显式选择探索套件后，模型会读取当前页面、选择允许的控件操作，并记录可疑交互。默认只从现有 CLI 工作台配置读取 `[llm]`，不启动该配置对应的产品运行时、不打开其中的用户 Vault。

```sh
# 固定验收通过后，再由模型自动探索
npm run test:acceptance -- --explore --max-steps 40 --max-api-calls 50

# 单独运行模型探索
npm run test:acceptance -- --suite exploration --max-steps 40 --max-api-calls 50

# 使用另一份模型配置；该文件可以只有 [llm]
npm run test:acceptance -- --suite exploration --model-config /path/to/model.toml
```

模型配置使用工作台已有字段：`base_url`、`api_key`、`model`，以及可选的 `provider`、`thinking_mode`、`reasoning_effort`。支持现有 `LlmClient` 的供应商协议。密钥不需要写进命令行。

默认任务为：

1. 修改每日论文数量上限，等待保存，刷新并重新打开设置验证。
2. 收起与展开侧栏，分别刷新验证持久化。
3. 生成固定日期日报，等待完成，再打开对应论文。

通过条件由程序检查页面和文件确认。模型说“完成”本身不算通过；模型提出的疑点显示为“待复核”，不自动当作已证实的产品故障。

默认最多 50 次模型决策、100 次模型 HTTP 请求（含客户端重试）、10 分钟。可使用 `--max-steps`、`--max-api-calls`、`--max-duration-ms` 调整。实际调用数及 token 用量写入探索报告。普通固定验收不读取真实模型密钥，也不产生探索模型调用。

模型只能使用当前观察中的控件引用进行点击、输入、选择、返回、刷新、滚动及短暂等待。路径、密钥、外部连接、邮件、调度等设置由测试环境控制。探索会保留每一步观察、动作、错误及截图；预算耗尽或无法继续会明确标为未验证。

## 看结果

证据默认写入 `output/playwright/acceptance/run-*/`，命令结束时打印报告路径：

- `report.html`：展开查看各场景的操作、断言、错误和证据链接。
- `report.json`：套件、场景、通过数量、待复核项及运行元数据。
- `workbench/`：截图、失败 DOM 文本、浏览器 trace、CLI 输出及外部请求记录。
- `obsidian/`：截图、宿主诊断、状态和产物证据、外部请求记录。
- `exploration/`：模型逐步观察与操作记录、任务判据、截图和使用量。

运行结束后清理临时产品数据与自有进程，保留证据目录。Ctrl-C 会取消已接线的工作并回收自有进程；被中断的工作不会记为通过。

| 结果 | 含义 | 退出码 |
|---|---|---|
| `passed` | 所选场景实际执行并满足断言 | 0 |
| `failed` | 产品行为、验收执行或证据保存失败 | 1 |
| `blocked` | 环境、模型服务或前置条件阻止验证 | 2 |
| `not-run` | 尚未执行，或中断/预算结束后未开始 | 整体不会返回 0 |

只运行工作台时，报告仅代表工作台。缺少 Obsidian 不会悄悄变成“全部通过”。空断言、缺失套件、未完成的模型任务均不能通过。

## 当前覆盖范围

| 功能 | 本版真实界面覆盖 | 其他已有测试与边界 |
|---|---|---|
| 首次使用与设置 | 草稿、保存、模型列表、配置冲突、刷新/重启恢复 | 并非逐项操作所有设置字段；见 `apps/cli/tests/workbench-settings*.test.ts`、`plugin/tests/settings*.test.ts` |
| 主题与方向 | 修改方向，并确认真实生成请求使用保存的模型和方向 | 新增/删除/排序等组合主要由现有设置测试覆盖 |
| 日报 | 完整生成、文件/索引/运行状态一致、重复启动、取消重跑、认证失败恢复 | 零匹配、公告等待、断点及持久化异常主要由 Core pipeline/scheduler 测试覆盖 |
| 单篇总结 | 工作台生成、保存、当前页面更新和打开阅读 | PDF 回退和其他错误路径由现有 Core/CLI 测试覆盖 |
| 阅读 | 搜索、概览、来源日报、收藏、待读、返回导航、重启恢复；Obsidian PDF 页码 | 大规模分页、全部键盘路径及移动端未做本版真实浏览器走查 |
| 文献库 | 工作台连接、浏览及方向审核入口 | 本版未实跑完整扫描、索引、检索、提议生成与接受；已有 `cli-personalized-workflow.test.ts`、`workbench-library*.test.ts` 和 Core library 测试 |
| 调度 | 检查设置操作与重启不会意外开启自动任务 | 实际定时触发、时间窗口及外部调度安装由现有 scheduler/CLI 测试覆盖 |
| 邮件与数据迁移 | 固定响应支持不发送真实邮件的测试环境 | 本版未操作邮件/导入导出 UI；保留 `cli-email.test.ts`、`cli-data.test.ts` 等已有测试 |

这套验收补充既有单元、集成和组件测试。日常改动仍保留项目原有 `npm test`、`npm run typecheck` 等检查；自动验收减少重复手点，覆盖清单也用于明确下一步该补哪些用户任务。

## 维护用例

- 固定场景在 `workbench.mjs`、`obsidian-scenarios.mjs` 中维护；先写用户步骤及能判错的结果检查，再实现操作。每个通过场景至少有一个实际断言。
- 模型任务在 `exploration-browser.mjs` 的 `createDefaultExplorationTasks()` 中维护。新增任务需要自然语言目标和独立 `verify`，不能让模型自行决定测试通过。
- 新的外部响应放在 `fixtures.mjs`，未知请求会失败；关键协议使用真实解析器或客户端做契约验证。
- 将确认过的探索发现转换成固定回归。保留失败截图和复现步骤，不通过忽略错误或放宽预期来变绿。

基础设施测试无需启动浏览器或 Obsidian：

```sh
npm run test:acceptance:tools
```

它们也包含在既有 `npm run test:release-tools` 中。

## CI

`.github/workflows/ui-acceptance.yml` 在 PR、main 推送和手动触发时运行固定工作台验收，安装对应版本的 Playwright Chromium，并上传报告、截图和 trace。失败会保留为失败，证据上传步骤会始终尝试执行。

公共 CI 不调用用户的模型 API，也不安装 Obsidian；Obsidian 和真实模型探索通过前述本机命令执行。此分支中的 CI 配置已经过本地契约检查，推送后才会在远程运行。

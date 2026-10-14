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

显式选择探索套件后，模型会读取当前页面、选择允许的控件操作，并记录可疑交互。默认只从现有 CLI 配置（Linux/macOS `~/.config/arxiv-daily/config.toml`，Windows `%APPDATA%\arxiv-daily\config.toml`）读取 `[llm]` 表，不启动该配置对应的产品运行时、不打开其中的用户 Vault、密钥不出现在命令行或任何输出里。

### 发版命令

两端各跑一次即可留下一次发版用的完整证据：

```sh
# Reading workbench
npm run test:acceptance -- --suite exploration --host workbench --release

# Obsidian（先满足下面的前置条件）
npm run test:acceptance -- --suite exploration --host obsidian --release
```

`--release` 在不显式传 `--persona` 时等价于连同全部用户角色一起跑；运行预算已按两端实测用量设置好发版默认值（见“预算”一节），通常不需要再加 `--max-steps` 等参数。要连续跑、互不混证据，各自带一个新的 `--output` 目录。

这条命令做的事：

- 两个产品都在隔离的临时配置、Vault、缓存和端口中运行，不碰用户的真实数据；论文源、模型列表、邮件等外部请求全部走固定响应，不产生真实网络副作用。
- 模型只能看到并操作插件（Obsidian）或工作台自己界面上的控件：点击、输入、选择、返回、刷新、滚动、短暂等待，以及回到插件主界面的 `home` 动作；文件路径选择、API 密钥字段、外部链接、发送邮件、清除运行状态、调度开关等危险或越权操作在驱动层直接拦截，不会真的执行。
- 每个目标任务由独立程序 `verify` 判定通过/失败/阻塞，模型自称“完成”不算数；用户角色自由探索不做判定，只产出“待复核”疑点。

前置条件：

- workbench 宿主：无额外条件，复用上面“运行”一节的 Node.js / Playwright 环境。
- obsidian 宿主：目前只在 Linux 验证，需要 `Obsidian`、`xvfb-run`、`Xvfb`、`xauth`（见上文“完整桌面验收”段）；默认 Obsidian 路径 `/opt/Obsidian/obsidian`，可用 `OBSIDIAN_BINARY` 覆盖；不需要准备或指定自己的 Vault，探索会用临时 Vault。
- 两端都需要一份可用的模型配置：现有 CLI 配置文件里的 `[llm]` 表（`base_url`、`api_key`、`model`，可选 `provider`、`thinking_mode`、`reasoning_effort`，兼容现有 `LlmClient` 支持的供应商协议），或用 `--model-config /path/to/model.toml` 指定一份只含 `[llm]` 的文件。密钥本身不需要、也不应该出现在命令行参数里。

### 日常调试：只跑部分任务或角色

```sh
# 固定验收通过后，再由模型自动探索（仅默认目标任务，不含角色）
npm run test:acceptance -- --explore --max-steps 40 --max-api-calls 50

# 单独运行模型探索
npm run test:acceptance -- --suite exploration --max-steps 40 --max-api-calls 50

# 使用另一份模型配置；该文件可以只有 [llm]
npm run test:acceptance -- --suite exploration --model-config /path/to/model.toml

# 只运行指定的目标任务或角色（可重复传参，也可用逗号分隔）
npm run test:acceptance -- --suite exploration --task settings-persistence,generate-and-read
npm run test:acceptance -- --suite exploration --persona keyboard-only

# 切换到 Obsidian 宿主
npm run test:acceptance -- --suite exploration --host obsidian
```

用户角色（`--persona`）只产出“待复核”疑点，没有独立判定，默认不随目标任务一起运行。

### 当前任务与角色清单

跑 `npm run test:acceptance -- --list` 可以拿到当前实际生效的清单（和 `--task`/`--persona` 的合法取值一一对应）；下表是同一清单的中文说明，任务库增减后记得一起更新。

workbench 目标任务（`--host workbench --task ...`，共 12 个）：

| id | 内容 |
|---|---|
| `settings-persistence` | 设置自动保存并刷新复查 |
| `sidebar-persistence` | 侧栏折叠和展开状态持久化 |
| `generate-and-read` | 生成固定日期日报并打开论文 |
| `star-filter-restore` | 收藏后切换筛选并恢复阅读 |
| `generate-paper-note` | 为单篇论文生成详细总结 |
| `external-config-conflict` | 设置冲突后放弃草稿并重新打开 |
| `library-build-index` | 构建个人文献库搜索索引 |
| `library-revoke` | 撤销个人文献库的处理授权 |
| `library-catalog-filter` | 筛选个人文献库目录 |
| `library-index-search` | 在索引检索方式下检索个人文献库 |
| `library-accept-direction` | 生成并采纳一个来自个人文献库的候选研究方向 |
| `library-edit-candidate` | 重命名并丢弃个人文献库的候选研究方向 |

obsidian 目标任务（`--host obsidian --task ...`，共 14 个）：

| id | 内容 |
|---|---|
| `daily-paper-limit` | 修改每日论文数量上限 |
| `star-paper-from-report` | 在阅读面板收藏一篇论文 |
| `add-topic-direction` | 新增研究方向 |
| `remove-topic-direction` | 删除研究方向 |
| `add-arxiv-category` | 新增 arXiv 分类 |
| `run-for-date` | 按日期生成日报 |
| `cancel-then-rerun` | 运行中取消后重跑 |
| `auth-failure-retry` | 认证失败后重试 |
| `completed-date-idempotent` | 对已完成日期重复运行不重复生成 |
| `dashboard-tabs-paging` | 切换到 All 标签并翻到第二页 |
| `mark-to-read` | 将论文标记为待读 |
| `create-paper-note` | 为指定论文生成论文总结 |
| `summarize-by-id` | 按 arXiv 编号生成论文总结 |
| `open-today-report` | 从日历打开日报 |

用户角色（`--persona`，两端共用同一组 id，自身文案按宿主调整）：`first-time-user`（第一次使用的新用户）、`impatient-clicker`（没耐心快速乱点的用户）、`keyboard-only`（只用键盘操作、动作被限制为按键和等待的用户）。`--release` 在不显式传 `--persona` 时，等价于连同以上全部角色一起选中。

### 预算

整次运行默认最多 90 次模型决策、15 分钟，模型 HTTP 请求（含客户端重试）默认为步数预算的两倍；`--release` 在不显式传 `--max-steps`/`--max-duration-ms`/`--max-api-calls` 时分别改用 220 步、20 分钟（1,200,000 ms）、440 次调用这组更大的默认值。这组发版默认值按两端各两次真实模型发版运行的实测最大用量（约 80–106 次模型调用、约 6–9 分钟）留出约两倍余量定的；日常只跑单个或少数任务不需要这么大的预算，所以非 `--release` 场景仍用原来较小的默认值。显式传的 `--max-steps`、`--max-api-calls`、`--max-duration-ms` 始终优先于 `--release` 的默认值，不论两者在命令行里谁先出现。

每个任务自己还有一份默认预算（目标任务 20 步；用户角色因为能覆盖的操作密度不同分别是 18/16/12 步，`keyboard-only` 每步只按一个键，预算最紧），API 调用上限为步数预算的两倍，超出只阻塞该任务、不影响其余任务；可用任务对象自身的 `maxSteps`/`maxApiCalls`，或全局 `--task-max-steps`、`--task-max-api-calls` 调整。整次运行的预算耗尽（含模型适配器自身的调用上限）会结束整次运行，未执行的任务记为 `not-run`。实际调用数、输入输出 token 及耗时按任务记录并汇总写入探索报告。普通固定验收不读取真实模型密钥，也不产生探索模型调用。

模型只能使用当前观察中的控件引用进行点击、输入、选择、返回、刷新、滚动及短暂等待（`keyboard-only` 角色进一步限制为按键和等待）。路径、密钥、外部连接、邮件、调度等设置由测试环境控制。探索会保留每一步观察、动作、错误及截图；预算耗尽或无法继续会明确标为未验证。

### 看探索报告

探索的证据在 `exploration/` 子目录；总报告的看法见下一节“看结果”，这里补充探索专属的几点：

- 目标任务的状态仍是 `passed`/`failed`/`blocked`/`not-run`；用户角色没有判定，统一是 `explored`（报告里显示为“已探索（仅待复核）”），即使角色找到很多疑点也不算失败。
- **只跑角色、不跑任何目标任务时退出码仍是 0**（因为没有失败/阻塞的场景）——这不代表任何功能被验证过，只代表角色探索顺利跑完；`report.html` 在这种“全是 explored、没有任何 passed”的情况下会额外提示一行，但看退出码或 `report.json` 里的 `counts` 时要自己确认 `passed` 不是 0。
- 模型提出的疑点列在 `report.html` 的“待复核的交互观察”里，和报告里每项，附复现步骤摘要、`evidence` 文字和对应截图链接；这些是“待复核”，不是已证实的产品故障，需要人工看证据和截图后再下结论（见下一小节）。
- `exploration/exploration.json` 是逐任务机器可读明细（状态、断言、`modelUsage`/`usage` 的调用数与 token、`durationMs`）；`exploration/exploration-events.jsonl` 是逐步事件流（观察、动作、错误），配合同目录下的截图可以完整复盘某一步发生了什么。

### 实测成本与耗时

两端各两次真实模型发版运行（2026-10-12，模型为某供应商的 `gpt-6-luna`，经由现有 CLI `[llm]` 配置）：

| 宿主 | 目标任务数 | 角色数 | 模型调用 | 输入 token | 耗时 |
|---|---|---|---|---|---|
| workbench | 6 | 3 | 80 | 约 57 万 | 约 4.7–6.0 分钟 |
| obsidian | 14 | 3 | 105–106 | 约 51–70 万 | 约 8.1–8.9 分钟 |

成本和耗时会随换用的模型、`base_url` 和实际响应速度明显变化；上表只是用于校准“预算”一节发版默认值的一组实测参考，不是对所有模型的承诺。要知道某一次具体花了多少，看那次报告的 `exploration/exploration.json` 里的 `modelUsage`/`usage` 字段。

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
| `explored` | 仅探索套件的用户角色场景；自由探索完成，不代表任何判定 | 不单独影响退出码（见下） |

只运行工作台时，报告仅代表工作台。缺少 Obsidian 不会悄悄变成“全部通过”。空断言、缺失套件、未完成的模型任务均不能通过。探索套件如果只跑了用户角色（没有任何目标任务），`explored` 不算失败也不算阻塞，整体仍会报 `passed`、退出码 0——这不代表验证过任何功能，只代表角色探索顺利跑完，参见上文“看探索报告”。

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
- 新增一个探索目标任务：在对应宿主的 `exploration-tasks/workbench.mjs` 或 `exploration-tasks/obsidian.mjs` 里的 `goalTasks()`/`GOAL_TASK_IDS` 加一项，给出模型能看懂的自然语言 `goal` 和一个独立、可信的 `verify`（程序检查页面/文件状态，不能让模型自称完成就算过）。落地前至少确认两件事：`verify` 在任务真正完成时判通过（正向），以及故意不完成或弄错一步时 `verify` 真的判不通过（负向，写成受控模型场景或单测）；再用 `node scripts/tests/acceptance-exploration*.test.mjs` 或 `node --test scripts/tests/acceptance-exploration-host.test.mjs` 之类的受控模型回归跑通，有条件的话用真实模型跑一次该任务（`--task 新任务id`）确认可完成。新任务要同步补进本文件“当前任务与角色清单”一节（或改为引用 `--list` 的输出）。
- 新增一个用户角色：角色的 id/标题/按宿主调整的 Chinese 文案/每任务步数预算都在 `exploration-tasks/personas.mjs` 里的 `PERSONA_TASK_IDS`/`BRIEFS`/`MAX_STEPS` 维护，两端共用；角色没有 `verify`，不做判定，只允许在 brief 里提出“发现就记一条 finding”的要求（可以像 `keyboard-only` 一样额外约束允许的动作类型）。同样要同步更新“当前任务与角色清单”。
- 新的外部响应放在 `fixtures.mjs`，未知请求会失败；关键协议使用真实解析器或客户端做契约验证。
- 把探索报告里“待复核”的疑点转成固定回归的流程：先打开 `report.html` 的“待复核的交互观察”，对着 `evidence` 文字和截图、再翻 `exploration/exploration-events.jsonl` 定位具体哪一步、哪个控件；能复现的，在 `workbench.mjs`/`obsidian-scenarios.mjs` 里补一个固定场景重放相同操作并加断言（断言要能在问题不修的情况下真的报错，不要写一个总是通过的断言），如果是产品缺陷就先改产品代码再让新场景变绿；判断是误报或不值得改的，在 journal 里记录结论和理由，不要悄悄删掉证据。不通过放宽断言、忽略错误或删掉截图来让回归变绿。

基础设施测试无需启动浏览器或 Obsidian：

```sh
npm run test:acceptance:tools
```

它们也包含在既有 `npm run test:release-tools` 中。

## CI

`.github/workflows/ui-acceptance.yml` 在 PR、main 推送和手动触发时运行固定工作台验收，安装对应版本的 Playwright Chromium，并上传报告、截图和 trace。失败会保留为失败，证据上传步骤会始终尝试执行。

公共 CI 不调用用户的模型 API，也不安装 Obsidian；Obsidian 和真实模型探索通过前述本机命令执行。此分支中的 CI 配置已经过本地契约检查，推送后才会在远程运行。

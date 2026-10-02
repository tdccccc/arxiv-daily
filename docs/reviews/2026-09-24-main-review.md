# main 分支 review — 2026-09-24

> Branch: `fix/onboarding-review`（从 `origin/main` 切出）
>
> Baseline: `976c12b`（0.4.6 + personal library setup path）
>
> Scope: 设置页与新手引导（重点）、插件运行时、core 报告流水线、core 状态与投递、CLI 与邮件中继

## Executive verdict

用户反馈的“引导里有些按钮点了没反应”已在真实 Obsidian 1.13.7 中复现，根因明确：新版（1.13+）声明式设置页上，引导的定位按钮找不到目标，点击后什么也不发生（F1）。同一轮复现还确认了三个同源问题：设置完成后编辑研究主题每输入一个字符就整页重绘、输入框失焦（F2）；“Daily reports folder”等文本框每按一个键就提交一次输出目录切换（F11）；引导卡片标题重复、被挤在右侧控件栏（F8）。

引导流程的其余问题集中在“看似完成、实际跑不起来”：首份报告固定用“今天”，周末或 arXiv 当天未公告时必然失败（F3）；新主题标签会重复，引导前三步都显示完成却没有生成按钮（F4）；引导显示“Setup complete”但每日自动运行默认关闭（F10）。

core 的批次 A/B 由早期后台 agent 审查；CLI、node-runtime 与邮件中继的批次 C 已于 2026-09-28 由接续主会话完成。最后新增的高优先级配置权限问题 F28 已修；其余问题的本轮处置见下文与 Helm journal。

| 区域 | 结论 | 建议强度 |
| --- | --- | --- |
| 新手引导（1.13+） | 两个主要按钮失效、编辑失焦，是用户反馈主因 | 高，先修 |
| 设置页其它控件 | 文本控件逐键提交、sidecar 地址改不了、旧版逐键重绘等行为缺陷 | 中 |
| 插件运行时 | 未发现崩溃级问题；个别状态不刷新、旧数据导致加载失败 | 低 |
| core / CLI / relay | 见下文 | 见下文 |

## Method and limitations

- 主会话读代码审查设置页（`plugin/src/settings/*`、`plugin/src/onboarding.ts`）与插件运行时（`plugin/main.ts`、`commands.ts`、`dashboard/`、`services/`、`hosts/`）；另派一个只读 agent 审查 core、CLI 与邮件中继，其高严重度结论由主会话复核后才写入。
- 使用仓库自带的桌面验收框架（`scripts/desktop-acceptance/`）在虚拟显示器里启动隔离的 Obsidian 1.13.7（从本机 `~/.config/obsidian/obsidian-1.13.7.asar` 复制，隔离配置目录，一次性 vault 位于 `/tmp`），用 CDP 直接点击引导按钮、在输入框里输入、关闭设置窗口，确认 F1、F2、F8、F9、F11 以及 C2 的实际行为。探针脚本不在仓库内。
- 1.13.7 的设置页运行在独立窗口里（`containerEl.ownerDocument` 不是主窗口的 `document`），后续写测试或探针时需要注意。
- 探针用程序派发的 `input` / `change` 事件模拟输入，不等同于真人键盘输入（例如程序赋值不会触发 `change`）；关闭设置窗口只测了 `app.setting.close()` 这一种方式。
- 未进行真实 LLM 调用、arXiv 抓取或邮件投递。

## Verification results

| 检查 | 结果 | 观察 |
| --- | --- | --- |
| `npm run check:boundaries` | 通过 | |
| `npm run lint` | 通过 | 0 errors、21 warnings（上限 64） |
| `npm run typecheck` | 通过 | core、node-runtime、CLI、plugin |
| `npm run build` | 通过 | plugin `main.js` 约 1.5 MB |
| `npm run check:obsidian-submission` | 通过 | |
| `NODE_OPTIONS=--max-old-space-size=8192 npm test -- --maxWorkers=1` | 通过 | 2035 passed、2 skipped |
| `plugin`: `npm test` | 通过 | 43 files、702 tests |
| email relay 单独测试 | 通过 | 141 tests |
| 真实 Obsidian 1.13.7 探针 | 复现 F1、F2、F8、F9、F11；C2 未复现 | 见各条 |

现有测试全部通过，但没有一条覆盖 1.13+ 路径下引导按钮能否找到目标，也没有覆盖“设置完成后编辑主题”的场景；插件测试里 `TextComponent.onChange` 的 mock 绑定在 `change` 事件上，而真实 Obsidian 在每次 `input` 时触发，所以旧版逐键行为（F13）在测试里看不到。

## Onboarding and settings findings

### F1 — 1.13+ 上引导的 “Connect AI” / “Describe interests” 按钮点了没反应

**Priority: P0 · 用户报告主因 · CONFIRMED（真实 Obsidian 1.13.7）**

- 按钮回调 `scrollToSection` 用 `[data-arxiv-daily-section="…"]` 查找目标，找不到就静默返回：`plugin/src/settings/tab.ts:2244-2258`。
- 这个标记只在旧版 `display()` 的 `sectionHeading` 里设置：`plugin/src/settings/tab.ts:397-410`。新版声明式定义里的 LLM 分组、arXiv categories 列表、Research topics 列表都没有任何可定位的标记：`plugin/src/settings/definitions.ts:203-265`。
- 真实复现：页面上 `[data-arxiv-daily-section]` 数量为 0；点击两个按钮后滚动位置不变、焦点不变。三个分组都渲染为 `containerEl` 内的 `.setting-group` 元素（列表额外带 `mod-list`），所以按分组定位是可行的。
- “Choose sources” 同理失效，只是默认分类 `astro-ph` 让第二步一开始就完成，按钮不出现。

**建议**：三个分组用 1.13 支持的 `cls` 加上稳定 class，`scrollToSection` 在两条渲染路径上都能按 class / 标记找到目标；找不到时给出提示而不是静默。补一条 1.13+ 路径的测试。

### F2 — 设置完成后编辑研究主题，每输入一个字符就整页重绘、输入框失焦

**Priority: P0 · CONFIRMED（真实 Obsidian 1.13.7）**

- 主题卡片的名称、标签、描述每次 `input` 都保存并调用 `refreshSetupGuide`：`plugin/src/settings/tab.ts:2507-2555`。
- 1.13+ 上它走 `refreshDeclarativeSetupGuide`：引导行在页面上时只重绘这一行；引导行不在页面上时退回 `refreshSettings()` → `update()` 整页重绘：`plugin/src/settings/tab.ts:2043-2050`。
- 首份报告完成后引导行不再出现（`showSetupGuide` 为 false）：`plugin/src/settings/definitions.ts:187-192`，`plugin/src/onboarding.ts:17-21`。
- 真实复现：用“已完成设置”的数据启动，在主题名称里输入一个字符后，原输入框已从页面移除，焦点落在设置窗口容器上。

本地 `main` 上未推送的 `5486790` 只处理了引导行存在时的情况，远程的 `6b96b57` 同样如此。

**建议**：引导行不在页面上时，只有引导需要重新出现（`shouldShowSetupGuide()` 变为 true）才整页刷新，否则什么都不做。补测试：设置已完成时编辑主题不触发整页刷新。

### F3 — “Generate first report” 在周末或 arXiv 当天未公告时必然失败

**Priority: P1 · CONFIRMED（代码路径）**

- 首份报告固定使用配置时区的“今天”：`plugin/src/settings/tab.ts:2260-2269`。
- 当天没有公告列表时返回 `failed_transient`（`packages/core/src/sources/arxiv-source-adapter.ts:100-111` 附近），列表为空时返回 `pending`（`packages/core/src/pipeline/pipeline.ts:187-191` 附近）；提示只有一句 `describeResult` 文本。
- 默认时区 `Asia/Shanghai`，北京时间上午 arXiv 公告之前运行也会失败。
- 同样的“固定今天”还出现在命令 “Run today”（`plugin/src/commands.ts:76-82`）和 Dashboard 的 “Run today”（`plugin/src/dashboard/view.ts:2214-2222`）；Dashboard 日历对“今天”跳过了已公告日期检查（`plugin/src/dashboard/view.ts:1268`）。

**建议**：首份报告改为最近一个已公告的日期（`plugin.recentDates` 已提供）；取不到时退回今天并说明原因和何时重试。命令和 Dashboard 的 “Run today” 语义是“今天”，保持不变，但失败提示应说明“当天尚未公告”。

### F4 — 新主题标签重复；引导前三步全部完成却没有生成按钮

**Priority: P1 · CONFIRMED（代码路径）**

- `addTopic` 用 `topic-${topics.length + 1}` 作标签：先删掉第一个主题再新增，就会与现存的 `topic-2` 重名：`plugin/src/settings/tab.ts:909-918`。
- 按名称自动生成标签对新主题永远不生效：`wasAuto = topic.tag === slugify(topic.name)` 在新主题上是 `"topic-1" === ""`，结果为 false：`plugin/src/settings/tab.ts:2507-2514`。
- 引导第 3 步只检查字段非空：`plugin/src/onboarding.ts:33-40`；能否运行用的是 `validateFilterConfig`，它还检查重复标签、重复分类、输出目录：`packages/core/src/settings/validation.ts:75-139`。
- 结果：前三步都显示 Complete，第 4 步写着 “Complete the earlier configuration steps…”，没有按钮，真实原因藏在折叠的 “Configuration details” 里——而那里显示的是调度器校验结果，不是 `validateFilterConfig` 的原因（`plugin/src/settings/tab.ts:2179`）。

**建议**：新主题的标签留空并随名称自动生成，冲突时加后缀；三步都完成但仍不能运行时，在第 4 步直接写出阻塞原因。

### F5 — 新版设置页没有 Quick start 主题模板

**Priority: P2 · 需用户决定**

模板入口只在旧版 `display()` 里：`plugin/src/settings/tab.ts:1419-1435`。新版 Research topics 列表只有 “Add topic”：`plugin/src/settings/definitions.ts:252-265`。1.13+ 的新用户只能手写主题。

### F6 — 引导文案里的分组名与新版标题不一致

**Priority: P3 · CONFIRMED**

文案写 “under AI model” / “under arXiv”：`plugin/src/settings/tab.ts:2144,2152`；新版标题是 “LLM” / “arXiv categories”：`plugin/src/settings/definitions.ts:205,240`。

### F7 — “Generate first report” 运行期间按钮不禁用、没有进度

**Priority: P3 · CONFIRMED（代码路径）**

`plugin/src/settings/tab.ts:2164-2177`、`2184-2216`、`2260-2269`。重复点击只会得到 “already running / lock held” 类提示。引导会因为其它编辑被重绘，忙碌状态需要记在 tab 对象上，不能只改按钮。

### F8 — 1.13+ 引导卡片标题重复、被挤在控件栏里

**Priority: P2 · CONFIRMED（真实 Obsidian 1.13.7 截图）**

引导行定义了名称 “Getting started”（`plugin/src/settings/definitions.ts:187-192`），`renderSetupGuideRow` 把卡片塞进同一行（`plugin/src/settings/declarative-rows.ts:202-207`），卡片自己也有 “Getting started” 标题。截图中左侧是行名，右侧是只占控件栏宽度的卡片。邮件引导行用 `name: ""` 加专用 class 解决了同样的问题（`definitions.ts:434-440`，`declarative-rows.ts:340-360`）。

### F9 — 只剩一个分类时，删除按钮点了没反应

**Priority: P3 · CONFIRMED（真实 Obsidian 1.13.7）**

列表对每一项都设置了 `onDelete`（`plugin/src/settings/definitions.ts:250`），真实页面上唯一的分类行也有 “Delete” 控件；`deleteCategory` 遇到只剩一个分类时静默返回（`plugin/src/settings/tab.ts:901-906`）。另外 `onDelete` 还会启用 Delete/Backspace 快捷键。

### F10 — 引导显示“完成”，但每日自动运行默认是关的

**Priority: P2 · 需用户决定**

`schedule.enabled` 默认 false（`packages/core/src/settings/defaults.ts:27`），`getSetupStatus` 不检查它（`plugin/src/onboarding.ts:23-59`）。首份报告完成后，旧版显示 “Setup complete”，新版直接隐藏引导，用户容易以为之后每天会自动生成。

### F11 — 1.13+ 的文本控件每按一个键就提交一次（输出目录会被逐键切换）

**Priority: P1 · CONFIRMED（真实 Obsidian 1.13.7）**

“Daily reports folder”、“Paper notes folder”、“From email”、“From name” 使用声明式 `text` 控件（`plugin/src/settings/definitions.ts:287-313,480-497`）。探针在输入框里连续输入 3 个字符（间隔 120 ms），`setControlValue` 被调用 3 次，依次提交 `arxiv-daily/daily-`、`…-a`、`…-ab`，没有防抖。

对输出目录，每次提交都会走一遍完整的输出存储切换（`plugin/src/settings/change-service.ts:159-176`）；只要有运行或操作在进行，每个按键都会失败、弹错误提示并整页重绘（`plugin/src/settings/tab.ts:252-269`），输入被打断。旧版路径用 `change` 事件、失焦时才提交（`plugin/src/settings/tab.ts:1514-1544`），没有这个问题。

**建议**：这几个字段改用自绘行，在 `change`（失焦或回车）时提交，沿用旧版的草稿校验。

### F12 — 启用本地 PDF 解析 sidecar 后，永远改不了它的地址

**Priority: P2 · CONFIRMED（代码路径）**

- 两个地址（capability、parse）必须同源，校验在任何 `pdfParserSidecar.*` 修改时执行：`packages/core/src/settings/validation.ts:190-209`，`plugin/src/settings/change-service.ts:255-258`。
- 每个输入框只提交自己的那个键：`plugin/src/settings/declarative-rows.ts:717-743`，`plugin/src/settings/tab.ts:1724-1744`。改 capability 的端口会因与 parse 不同源被拒绝，反之亦然。
- 地址行只在启用后显示（`plugin/src/settings/definitions.ts:413-426`），而未启用时才不校验——所以界面上没有任何顺序能把端口从 5001 改成别的。

**建议**：两个地址合并成一次提交（同一行里两个输入，或提交时带上另一个字段的当前草稿）。

### F13 — 旧版（<1.13）设置页若干文本框逐键保存并触发副作用

**Priority: P2 · CONFIRMED（代码路径；真实 Obsidian 的 `TextComponent.onChange` 在每次 input 时触发）**

- 自定义分类输入框：每输入一个字符就把它存成分类并整页重绘，输入框随即消失，第一个字符就成了分类名：`plugin/src/settings/tab.ts:1387-1397`。
- Embedding base URL / model：每个按键都走 `saveEmbeddingEndpointField`；远程模式且已授权时，第一个按键就让授权失效并弹出披露确认框：`plugin/src/settings/tab.ts:1654-1698`。
- Sidecar 开关与地址：没有 catch，校验失败时产生未处理的 Promise rejection，开关停在错误状态：`plugin/src/settings/tab.ts:1715-1744`。

另外，新版分类行只有下拉框，没有自定义输入（`plugin/src/settings/declarative-rows.ts:209-231`，注释写着 “free-text override” 但没有实现），1.13+ 用户无法添加列表之外的分类。

### F14 — 模型只能从服务商返回的列表里选

**Priority: P2 · CONFIRMED（代码路径；含原 C3）**

两条路径的模型控件都是下拉框，只含当前值和 “Get models” 返回的列表：`plugin/src/settings/declarative-rows.ts:144-200`，`plugin/src/settings/tab.ts:1235-1284`。服务商不提供 `/models` 或请求失败时，无法填写模型名；换了 base URL 之后模型仍是默认的 `deepseek-v4-pro`，引导第 1 步照样显示完成，第一次运行才失败。

“Get models” 在当前模型不在列表里时，静默把模型改成列表第一项，也不刷新引导：`plugin/src/settings/tab.ts:2635-2667`。

### F15 — 开启 Enable 并选择 “Run today” 后，设置页要等整次运行结束才更新

**Priority: P3 · CONFIRMED（代码路径）**

`setScheduleEnabled` 在同一个队列里先保存开关、再 `await this.scheduler.tickToday()`（`plugin/main.ts:2289-2314`）；开关行在 `finally` 里才刷新（`plugin/src/settings/declarative-rows.ts:282-299`）。整次运行（可能几分钟）期间行名仍是 “Enable · Paused”，此时再关掉开关，关闭请求会排在运行之后。

### F16 — 选择一个已存在的分类，另一行会静默消失

**Priority: P3 · CONFIRMED（代码路径）**

`setArxivCategories` 去重（`plugin/src/settings/tab.ts:823-828`），把第 2 行改成与第 1 行相同的分类后，第 2 行直接消失，没有提示。

## Plugin runtime findings

### F17 — 旧版 data.json 里一条不完整的运行记录会让插件永远加载失败

**Priority: P3 · CONFIRMED（真实 Obsidian 1.13.7，构造数据）**

启动时若 `run-state.json` 为空而 data.json 里有旧的 `runState`，会原样 `replaceAll` 进去（`plugin/main.ts:418-423`），不做校验；写入走严格读取（`packages/core/src/services/state-store.ts:172-178`、`loadAuthoritativeRunState`），遇到缺 `lastAttempt` 的条目抛出 `invalid run-state entry`，`onload` 失败。探针复现：第一次启动把坏条目写进 `run-state.json`，之后每次启动都报 “Plugin failure”，只有手动删文件才能恢复。只影响极旧版本迁移上来的数据，但后果是插件完全不可用。

### F18 — 命令 “Run today” 和引导的首份报告完成后，已打开的 Dashboard 不刷新

**Priority: P3 · CONFIRMED（代码路径；原 C7）**

`plugin/src/commands.ts:76-82`、`plugin/src/settings/tab.ts:2260-2269` 都没有调用 `refreshOpenDashboardViews`，而 “Summarize by paper ID” 调用了（`commands.ts:278`）。Dashboard 不监听 vault 变化。

## 已核实的疑点（handoff C1–C7）

| 编号 | 结论 |
| --- | --- |
| C1 直接改内存设置再保存、无回滚 | **确认**。`saveSettings` 走 `persistCurrent`，此时内存已被改过，失败时无法回滚（`plugin/src/settings/change-service.ts:85-104`）。涉及主题卡片（`tab.ts:2507-2596`）、`addTopic` / `deleteTopic` / `applyTopicTemplate` / `setArxivCategories`（`tab.ts:823-980`）、首次选择 embedding 模式（`tab.ts:635-641`）、embedding dimension（`declarative-rows.ts:657-676`、`tab.ts:1699-1712`）。主题卡片的 `oninput` / `onchange` 没有 catch，保存失败是未处理 rejection。并入 F2/F4 的修复顺带处理，单列 P3。 |
| C2 关闭设置时未提交的输入丢失 | **未复现**。1.13.7 中 `app.setting.close()` 会对聚焦的输入框触发 `blur` / `focusout`，API key 被保存。其它关闭方式未测。 |
| C3 Get models 静默换模型 | **确认**，并入 F14。 |
| C4 旧版 Reasoning effort 强制 thinkingMode=true 不更新开关 | **确认**，P3。`tab.ts:1320-1341` 只回写下拉框；新版遇到自定义值显示成 Medium（`declarative-rows.ts:115-118`）。 |
| C5 “Open today's daily report” 自拼路径 | **不是缺陷**。输出目录在加载和保存时都已规范化（`plugin/src/settings/load.ts:74-100`、`change-service.ts:215-231`），与 `MarkdownWriter.dailyPath` 结果一致。 |
| C6 声明式文本控件的提交时机 | **确认有问题**，见 F11。 |
| C7 首份报告后 Dashboard 不刷新 | **确认**，见 F18。 |

## Core, CLI and relay findings

批次 A 来自早期后台审查（agent 又分出三个子 agent，看 LLM 客户端、arXiv 抓取与日期、Markdown 输出；这超出了“只用一个 agent”的约定，已叫停，其结论由主会话逐条复核）。以下补齐已接受的批次 B 修复与主会话独立完成的批次 C；本次接续没有使用子代理。

### F19 — 推理 / thinking 参数包在 `extra_body` 里发出，从未到达服务商

**Priority: P1 · CONFIRMED（代码路径）· 需用户决定**

`packages/core/src/llm/client.ts:255-262` 把 `thinking` 参数放进 `params.extra_body`，`requestChat` 直接 `JSON.stringify` 发送（`client.ts:352-370`），全仓库没有任何地方展开 `extra_body`。`extra_body` 是 OpenAI Python/Node SDK 的约定（由 SDK 展开到顶层），这里没用 SDK，所以请求体里是字面量的 `"extra_body"` 字段：Anthropic 预设的 extended thinking 从未开启；DeepSeek 的 `thinking` 开关同样无效；对严格校验参数的服务商，未知字段还可能被拒。

修改会改变发给每个服务商的请求，本环境无法用真实 API 验证，**未修**，留给用户决定（建议：按服务商把 `thinking` 放到顶层，并用真实 key 各验一次）。

### F20 — 论文笔记 frontmatter 的 `primary_topic` / `tags` 未加引号

**Priority: P2 · CONFIRMED · 已修（df8d1c0）**

`packages/core/src/pipeline/markdown-writer.ts` 的 `paperFrontmatter` 对 `title`、`authors` 加引号转义，但 `primary_topic` 和每个标签原样输出；主题标签是设置里的自由文本，`AI: Robotics`、`ml,dl`、`#hash` 会让 YAML 解析失败或被拆开。修复后普通 slug 输出不变，其它值加引号。

### F21 — 刷新论文笔记 frontmatter 会丢掉用户自己加的属性和标签

**Priority: P2 · CONFIRMED · 已修（2abb4c6）**

`refreshPaperNoteFrontmatter`（`markdown-writer.ts:171-184`）整块替换 frontmatter；对已验证的详情笔记重新执行 “Summarize by paper ID” 会走到这里（`packages/core/src/services/manual-fetch.ts:304-347`）。修复后只重算插件管理的字段，其它键原样保留，用户额外的标签接在插件标签后面。

### F22 — 已滚出 `/recent` 的日期被当作临时失败反复重试

**Priority: P2 · CONFIRMED · 已修（3413d35）**

`packages/core/src/sources/arxiv-source-adapter.ts:100-106` 对“不在 `/recent`”一律返回 `failed_transient`；`/recent` 只含最近几个公告日，早已没有其它回退来源，比最旧一天还早的日期永远拿不到，却会被调度器重试最多 10 次。修复后这类日期直接 `failed_permanent`；比最新一天还新的日期（尚未公告）仍是临时失败。两个钉住旧行为的 pipeline 测试同步改为新预期。

### F23 — 流式“空闲超时”实际上不起作用

**Priority: P3 · CONFIRMED · 未修**

`postChatStream`（`client.ts:340-350`）先等整个响应体到手再解析 SSE，两个 `HttpClient` 实现都一次性返回完整文本，所以 120 秒的空闲超时永远量不到网络空闲，卡住的服务商只受 300 秒总超时约束，加上 3 次重试最长约 15 分钟才失败。要修需要让 `HttpClient` 支持增量读取，改动面大，本轮不做。

### F24 — 按固定长度截断论文文本可能切开代理对

**Priority: P3 · PLAUSIBLE · 未修**

`packages/core/src/pipeline/detail-selector.ts:148`、`personalized-paper-filter.ts:535` 用 `.slice()` 截断，边界处的 emoji 等字符可能变成孤立代理项。不崩溃，仅影响边界一个字符。

### F25 — 崩溃中断的运行被永久放弃

**Priority: P1 · CONFIRMED · 已修（8b4b2a6）**

`packages/core/src/services/state-store.ts` 的 `recoverStaleRunning` 现在将未达到重试上限的过期 running 记录恢复为 `failed_transient`；达到 10 次上限才永久失败，并说明重试耗尽。提交记录包含两条先失败的回归测试；2026-09-28 全量 core 测试复核通过。

### F26 — 模型用单层代码围栏包裹 JSON 时过滤失败

**Priority: P1 · CONFIRMED · 已修（25889d7、b6a06ce）**

`packages/core/src/pipeline/paper-filter.ts` 只解开包裹整个回答的一层围栏，再执行原有严格 JSON 契约校验。外围说明、多段围栏、缺失闭合和内部非法 JSON 仍失败。b6a06ce 是同一修复的严格索引类型检查补充；本次全量 core 测试与工作区类型检查通过。

### F27 — 不支持自动邮件的系统只记录日志，测试邮件却成功

**Priority: P1 · CONFIRMED · 可见性已修（c55e53f）**

无 exclusive-create 能力时，插件邮件设置、测试发送结果及被拒绝的自动发送提示都会说明限制；CLI 的 email status/test 与运行日志也说明自动发送不可用。投递安全机制保持不变。macOS/Windows 实际自动投递能力按 2026-09-25 用户决定留给后续 initiative。

### 批次 B 的其他处置

- 过滤阶段的 paper index 写入失败现按可重试失败处理（7aa19aa；P5 chunk 3b），本次 core 全量复核通过。
- `RunLock`（`packages/core/src/services/run-lock.ts`）只在进程内互斥，`PaperIndexStore`（`packages/core/src/services/paper-index.ts`）的内存快照不能协调多个进程并发写同一 Vault。跨进程锁与索引一致性按既有 P5 计划记录、延期；没有宣称解决。
- 非空旧 `topics` 数组迁移的疑点仍为 PLAUSIBLE，未获得可接受的失败复现，不作为已修问题。

### F28 — CLI 初始化配置文件向其他本机用户开放密钥读取权限

**Priority: P1 / High · CONFIRMED（真实文件系统）· 已修（f6ef773）**

`apps/cli/src/init.ts` 的默认写入原来沿用 umask；实测新建配置为 0664，已有 0644 配置在 Overwrite 和 Keep existing 后均保持 0644。配置含 LLM API key、邮件 API key 或 hosted token，在可遍历的配置目录中会被其他本机用户读取。

修复复用 `NodeStorageAdapter.writeTextAtomic`，先以 0600 写入同目录临时文件再替换目标，新建配置目录使用 0700。三条真实文件测试先失败后通过，并验证内容仍能加载、Keep existing 保留旧设置且没有临时文件残留。普通配置读取不改权限；旧安装须再次运行 init 才应用此写入规则。未验证 Windows ACL。

### F29 — CLI 安装路径含空格时生成的 cron 命令无法执行

**Priority: P2 · CONFIRMED（本地只读探针）· 本轮延期**

`apps/cli/src/schedule-cmd.ts` 的 `buildCronLines` 直接插入 binaryPath。传入 `/tmp/my tools/arxiv-daily` 得到未加引号的 `/tmp/my tools/arxiv-daily run --today`，shell 会将它拆开。默认普通路径不受影响；后续应同时处理 shell 引号与 cron 的 `%` 语法。

### F30 — 倒置的重复运行时间窗会“成功安装”零条 cron

**Priority: P2 · CONFIRMED（内存 crontab 探针）· 本轮延期**

`scheduleFireSlots`（`apps/cli/src/config.ts`）对 on=18:00、until=09:00、intervalHours=1 返回空数组；`scheduleInstall` 返回 0 并打印 `installed 0 cron line(s)`。它仍会移除旧托管任务，因此手工 TOML 时间窗配置错误可使定时运行停止。未修改真实 crontab。

### F31 — 部分异步子命令绕过 CLI 统一错误处理

**Priority: P2 · CONFIRMED（注入失败的命令探针）· 本轮延期**

`apps/cli/src/main.ts` 的 data 等分支在 try 中直接 return Promise，未 await。探针对 data export 注入拒绝后，`runCli` 向调用者 reject，stderr 为空，而非进入 catch 返回 1；可执行入口最终走原始 `console.error(err)`，绕开该入口的统一输出脱敏。此探针仅使用虚构字符串，不证明真实密钥曾泄漏。后续应统一 await / 错误边界，并检查 email 分支的 signal handler 清理时机。

### F32 — 邮件中继对 JSON null 返回 500

**Priority: P3 · CONFIRMED（Worker handler 本地探针）· 本轮延期**

`services/email-relay/src/index.ts` 的 verifyStart 将已解析 JSON 直接当对象访问；POST `/v1/verify/start` 的 body 为 `null` 时返回 500 `relay request failed`，而非 400 输入错误。没有触发邮件或泄漏底层异常；后续可统一入口 JSON 对象形状校验。

### 批次 C 范围与验证（2026-09-28）

- 审查 CLI 的配置与初始化、命令分派、运行时组装、cron、更新、ZIP 导入导出；Node HTTP 的超时/取消、普通和独占存储路径、受限文献目录；relay 的验证发起/完成、认证、收件人绑定、账本/配额、provider 结果分类和 cutover 的 ready/issuance/authorize/action 主路径。cutover 并非逐分支形式化验证。
- 本地探针位于 `/tmp/arxiv-helm-review-probe.ts`，使用虚构配置、内存 crontab 和本地 Worker handler；没有真实 API 请求、邮件、cron 安装或线上 cutover 操作。
- `NODE_OPTIONS=--max-old-space-size=8192 npm test` 在新增 F28 前通过：core 2052 passed / 2 skipped、node-runtime 45、CLI 75、plugin 751。F28 后相关完整回归：CLI 78、node-runtime 45；core/plugin 无后续改动，未重复运行。
- 最终 root lint（0 errors / 21 existing warnings）、typecheck、build、check:boundaries、check:obsidian-submission 全部通过。relay 独立 typecheck 与 141 tests 通过；先前缺少 Cloudflare 类型，通过按现有锁文件安装依赖解决，未修改依赖声明或锁文件。
- F28 是本批唯一新增 High；F29–F32 按 P5 chunk 7 的范围记录延期。未重跑真实 Obsidian 验收；沿用 P1–P4/P6 已记录的真实会话证据。

### 已核对、不构成缺陷

- `/recent` 请求 `show=2000` 不分页：2026-09-24 实测 cs.LG 最近 5 个公告日共 1198 条，远低于 2000，暂不构成问题。
- LLM URL 拼接、鉴权头与密钥脱敏、重试与取消清理、过滤响应校验、每日摘要解析与渲染转义、prompt 注入防护、Markdown 写入的覆盖保护：子 agent 读代码未发现问题。

## 需人工验证

- F8 修复后的卡片排版（窄窗、宽窗、亮/暗主题）。
- F1 修复后在真实 Obsidian 中点击三个引导按钮，确认滚动到对应分组并聚焦。
- C2：用窗口关闭按钮、Esc、切换到其它设置标签页三种方式关闭时，未失焦的输入是否保存。

## Fix status（分支 `fix/onboarding-review`，未推送）

| 问题 | 状态 | 提交 |
| --- | --- | --- |
| F1 引导按钮没反应 | 已修，真实 Obsidian 复核 | 4cb2742 |
| F2 设置完成后编辑主题失焦 | 已修，真实 Obsidian 复核 | b5ce048 |
| F3 首份报告用今天 | 已修（改用最近已公告日） | 33a40d4 |
| F4 标签重复、第 4 步无原因 | 已修 | c1efaf2 |
| F5 1.13+ 无主题模板 | 用户决定不做 | — |
| F6 文案与标题不符 | 已修 | 1c39c8a |
| F7 首份报告无进度 | 已修 | 30612f0 |
| F8 引导卡片排版 | 已修，真实 Obsidian 截图复核 | 1c39c8a |
| F9 最后一个分类的删除按钮 | 已修，真实 Obsidian 复核 | 1c39c8a |
| F10 每日运行默认关闭 | 按用户决定加第 5 步 | 0d1091e |
| F11 1.13+ 文本控件逐键提交 | 已修，真实 Obsidian 复核 | 7d5fabf |
| F12 sidecar 地址改不了 | 已修 | d4a9662 |
| F13 旧版逐键保存 | 已修 | 6112c86 |
| F14 模型只能从列表选（含 C3） | 按用户决定改为可手填 | c20b9f6 |
| F15 Enable + Run today 卡住设置页 | 已修 | fa13654 |
| F16 重复分类静默丢行 | 已修 | 7f27e4c |
| F17 旧 runState 导致插件加载失败 | 已修，真实 Obsidian 复核 | 9ded48a |
| F18 运行后 Dashboard 不刷新（C7） | 已修 | d93f268 |
| C1 主题 / 分类保存失败无回滚 | 已修 | eea6192 |
| C4 旧版 Thinking 开关不同步 | 已修 | 21eb465 |
| F19 thinking 参数未到达服务商 | 待用户决定 | — |
| F20 frontmatter 未加引号 | 已修 | df8d1c0 |
| F21 刷新 frontmatter 丢用户属性 | 已修 | 2abb4c6 |
| F22 滚出 `/recent` 的日期反复重试 | 已修 | 3413d35 |
| F23、F24 | 记录，未修 | — |
| F25 崩溃恢复不再调度 | 已修 | 8b4b2a6 |
| F26 单层代码围栏 JSON 被拒绝 | 已修 | 25889d7、b6a06ce |
| F27 不支持自动邮件却没有提示 | 可见性已修，跨平台能力延期 | c55e53f |
| F28 CLI 初始化配置权限 | 已修 | f6ef773 |
| F29–F32 | 批次 C 确认的中低优先级问题，按计划记录延期 | — |

## Recommended fix order

1. F1、F2（用户直接感知的“没反应”和失焦）
2. F4（引导卡住）
3. F3（首份报告失败）
4. F9、F7、F6、F8（引导细节）
5. F11、F12、F13、F14（设置页其它行为缺陷）
6. core / CLI / relay 中确认的高优先级问题
7. 其余 P3

F5、F10 的用户决定已执行；本轮目标已收尾。F19、F23、F24、批次 B 的跨进程/迁移疑点与 F29–F32 的本轮豁免和后续边界见 `docs/helm/2026-09-24-onboarding-review-fixes/journal.md`。

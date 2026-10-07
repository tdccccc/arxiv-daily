# P6 — real-daily-and-desktop-acceptance

goal_ref: ../goal.md
created: 2026-09-07T04:07:50+08:00
updated: 2026-09-11T00:19:55+08:00
revision: 10

## Outcome

以测试库现有研究方向生成一篇真实日报，保留方向来源，并由用户查看新复审/设置交互后关闭本Helm。

## Assumptions

- 使用现有已授权LLM端点和现有6主题10方向（8条library来源），真实arXiv输入；不修改用户主题。
- 日报先生成到/tmp/arxiv-helm-daily/vault；邮件关闭，仅调用实际pipeline。默认每日20篇保留。
- goal约束仍要求用户在真实Obsidian看过新复审与设置页；自动化DOM和截图不能冒充用户确认。

## Approach

使用实际CLI composition root buildCliRuntime组装与插件共享的core，记录运行结果、方向标记和来源。真实模型P12已一次通过全覆盖零新增；P11三个合成对照已用真实端点验证。最后构建可部署插件，并在测试vault备份旧构建后安装，供用户完成桌面查看。

## Chunks

### Chunk 1 — 真实日报

- change kind: verification only
- strategy: proportionate real-runtime verification
- command: node /tmp/arxiv-helm-daily/build.mjs；node /tmp/arxiv-helm-daily/run.cjs run 2026-09-04
- evidence: completed、有实际论文条目、可解析topic方向标记，至少一条命中现有library来源方向；数量不超过20。
- failure handling: 保存真实错误与阶段，不把0篇或失败当通过；沙箱网络错误按工具规则请求必要执行权限。
- [x] observed real report accepted — 真实pipeline completed，10篇、10个有效来源标记，全部命中library来源方向（5条不同方向），五项结构化摘要字段各10份。

### Chunk 2 — 可复核构建与桌面验收

- change kind: delivery verification
- strategy: build/checksum + user acceptance
- checks: 最终diff、npm run test:workspaces/typecheck/lint/check:boundaries和plugin build的已观察结果；备份及cmp。
- desktop cases: 全覆盖零新增；已有主题追加与改归属；只接受A后可接受B；改名/删除与重新打开；接受保存时编辑，显示与持久化一致；cues只读；日报来源与遗漏说明。
- [x] test-vault build installed and checked — main.js/styles.css与真实日报已安装到/home/tiandc/Desktop/plugin_test；旧构建已备份，逐文件SHA-256一致。
- [ ] user desktop acceptance recorded

### Chunk 3 — 日报阅读体验反馈

- change kind: 折叠输出及方向回读兼容为行为修改；段落留白、计数措辞与现有日报重排为展示修正。
- strategy: 折叠与标记回读使用 Red→Green；留白与文案用现有回归及真实渲染检查。
- expected Red: 新版默认折叠区、方向逐条显示与折叠内标记读取的断言失败；旧格式仍有 Green 基线。
- implementation: 使用 Obsidian 原生折叠 callout 收起方向、来源及机器标记；方向不重复主题；保留新旧标记格式的身份与位置校验。五项摘要加段落间距，“详细收录”明确为“附独立论文总结”。
- Green and regressions: core 的 assembler、marker、parser、rescue、summarizer/pipeline 回归，四包 typecheck、boundaries、lint、plugin build；原日报与重排日报的10篇身份、方向、来源和五项摘要逐一一致。
- delivery: 先在工作区生成可复核日报，核对后备份并替换用户指定测试库日报及插件构建；无须重跑真实模型。真实 Obsidian 检查折叠、展开和正文可读性。
- boundary: 保留现有主题顺序、空主题与限额遗漏区别；摘要压缩、单位/对象名和公式转写疑点需原文核验，本块不猜测改写。
- [x] folded metadata and parser compatibility verified — 13 Red→86 Green；rescue 固定行传输 3 Red→Green；旧引用分隔 4 Red→Green，最终相关三文件137项通过。独立复核无阻塞；真实 Obsidian 阅读/实时预览、点击展开、四种旧来源组合通过。
- [x] reformatted real report and build delivered — 10篇身份/方向/来源和50个摘要字段逐一一致；3088项全workspace、四包typecheck、boundaries、build与diff检查通过，lint仍0 error/20既有warning。测试库日报及main.js已备份安装并校验，证据见 `.artifacts/daily-readability/`。

### Chunk 4 — 统一未来日报的阅读规则

- scope: 用户明确只改善今后生成的日报，CLI与插件一致；不修改、迁移或重新生成存量日报。上一块的通用折叠逻辑继续保留。
- change kind / strategy: 分组、渐进展示、解析兼容、提取器修复与旧摘要缓存失效使用Red→Green；提示词的简洁度与单位忠实性用既有长摘要作历史基线和真实模型小样验证，不以固定返回的mock证明语义质量。
- expected Red: 空主题仍穿插正文、四项辅助摘要未折叠、新折叠字段无法全部回读、旧prompt v1缓存仍能复用；真实MathML显示/TeX重复的最小fixture在共享提取器中失败。
- implementation: 有论文的主题保持原相对顺序，空主题汇总至末尾并区分未匹配/限额遗漏；核心结果默认可见，其余四项折叠；中英摘要提示词要求紧凑表达并保留原始单位与对象标识，prompt contract升级而结果schema不变；HTML提取按结构保留单一数学表示。
- verification: 正常/emergency/rescue两种语言、新旧解析、完整五字段与来源回读；CLI/插件实际pipeline的临时产物；对应checkpoint与MathML回归；全workspace、typecheck、lint、boundaries、CLI/plugin build；隔离Obsidian预览新生成样例；真实模型小样；既有日报目录SHA-256保持一致。
- ownership: 根代理修改共享日报分组/展示/解析、提示词与缓存版本；独立代理仅改MathML提取及测试、跨host测试。goal与journal由根代理维护。
- delivery: 构建CLI和插件并安装更新后的测试插件，只更新程序构建；新样例仅写临时目录或忽略的验收产物目录。
- [x] shared future-report behavior and host regressions verified — 共享分组/折叠/解析/缓存9 Red、rescue7 Red后相关279项Green；CLI中文和插件英文各经过真实pipeline的Red→Green，两文件46项通过，完整五字段索引与既有日报字节保留。
- [x] source extraction and concise grounded summaries verified — MathML与HTML备用路径均有行为Red→Green，相关96项通过；4篇真实缓存页的重复片段消除且原缓存不变。3次真实模型小样通过，中文完整样例核心结果147字符、英文45词，单位样例保留0.75 mJy/beam。
- [x] both products built, plugin installed, existing reports unchanged — 全workspace3109 passed/2既有skipped，四包typecheck、lint(0 error/20既有warning)、boundaries与CLI/plugin build通过；CLI构建可运行。中英×正常/emergency/rescue共6个隔离Obsidian样例、展开及实时预览通过，独立审查无阻塞。只安装main.js，26个既有日报文件（含备份）SHA-256保持一致，证据在`.artifacts/future-daily-readability/`。

### Chunk 5 — 主题设置与文献库复审的信息减负

- evidence: 用户指出 Topics from your library 应归到 Research topics 下，并反馈整条流程信息过多、难以理解。新版 declarative 设置确实把入口放在主题标题之前；复审主屏并列生成/刷新/概览/接受，概览重复方向全文，编辑区混合方向、归属与证据维护。
- scope / classification: P6 的 L1 体验验收修正；保留现有主题、方向、授权、选择及接受合同。9 月 8 日 Helm 的可靠性与可追溯结果继续有效，不重写其历史，也不改其他 active initiative。
- visual thesis: 沿用 Obsidian 原生界面，以简短标题、留白与单一主要操作呈现当前任务。
- content plan: Settings 给出主题入口与当前设置步骤；复审先选择/加入方向，再按需查看证据；完成后显示结果与下一步；概览每条方向仅显示一次。
- interaction thesis: 原生 disclosure 展开次要操作与证据；设置引导定位当前步骤；异步操作保留展开状态、草稿和焦点，遵循减少动画偏好。
- change kind / strategy: 布局、文案和重复段落删除使用既有 Green 基线与真实桌面检查，不为可逆展示改动制造测试；下一步引导、选择计数、接受后的完成状态等行为使用最小 DOM Red→Green，并回归授权、在途/失败保存、部分接受和预览。
- baseline: 改动前 settings-definitions、settings-tab、settings-declarative-tab、personal-library-interest-profile-modal、proposal-acceptance-ui 共 272 项通过。
- Green / regression: 上述相关插件套件、全插件测试、四包 typecheck、boundaries、lint、插件 build 和 diff 检查；隔离 Obsidian 中验证新版/旧版 Settings 入口位置、首次生成、编辑/证据、部分与全部接受、宽窄窗口。自动化桌面结果不替代用户最终体验确认。
- [x] Settings 入口归属与当前步骤引导验收 — 新版入口位于 Research topics 列表后；引导只突出首个未完成步骤，新版分区跳转可用，去除重复标题。两项交互 Red→Green，真实新旧版设置页均通过。
- [x] 复审主要操作、证据渐进展示与完成反馈验收 — 生成/刷新进入 More options，默认主操作为加入研究主题并显示所选方向数；完成后出现 Done。证据、归属和代表维护按需展开；概览去除重复段落；打开论文保留展开与键盘位置，并尊重用户中途移动焦点。计数、完成、去重、展开和焦点均有 Red→Green；最终全插件 806 项通过。
- [x] 真实桌面检查、构建及可复核证据交付 — Obsidian 1.13.7 的独立设置窗口与复审流程 18 项、1.11.5 旧版设置 5 项通过，均无 renderer error。四包 typecheck、boundaries、build、diff 检查通过；lint 0 error / 20 既有 warning。测试库构建已备份安装并核对 SHA-256，data.json 字节未变；记录和截图在 `.artifacts/topic-workflow-simplicity/`。预览/生成的桌面响应受控，未重跑真实模型或 core/CLI 全套；最终用户体验确认仍在 Chunk 2 保持未勾选。

### Chunk 6 — 复审默认页给出明确的下一步

- evidence: 用户实际使用后仍反馈 Review suggestions 内容多、文字乱、不知道该做什么。只读检查当前测试库：2026-09-07 的提案无新增方向，165 篇旧格式覆盖记录缺少可核对的依据；默认页需要更新分析，却把生成操作折叠了，没有主要按钮。Chunk 5 的可靠性验证仍有效，但默认页的信息组织未通过用户体验确认。
- scope / classification: P6 的 L1 反馈修正；按实际结果组织任务，不改变生成、匹配、授权或接受的业务合同。旧结果保留其真实性，不把未核实覆盖说成当前已覆盖。
- visual thesis / content: 默认页只呈现当前结果、一句说明和一个主要动作；覆盖诊断、库概览与维护操作进入下方详情。新建议按展开主题、勾选方向、加入主题的顺序阅读；细节页可明确返回复审。
- strategy: 先以旧覆盖/方向变更场景复现缺少主要动作，再以 DOM Red→Green 验证更新、取消授权、错误重试、无新增完成及返回复审；保持原草稿、部分接受与证据回归。字体、留白和内容顺序通过已有 Green 基线及真实桌面检查验证，不为纯展示改动新增结构测试。
- baseline: 改动前复审与接受两文件 87 项通过；用户实际提案已仅取主题/分析/目录快照到临时目录，不含配置凭据，不改用户数据。
- verification: 针对性复审、全插件、typecheck、boundaries、lint、build；隔离 Obsidian 1.13.7 使用实际旧提案和新建议场景检查宽窄窗口、明确下一步与更新后的结果；本轮不自动调用用户真实模型。
- [x] 按结果提供更新、重试或完成动作，并验证授权与失败路径 — 旧覆盖/方向变更/授权取消/读取重试 4 Red→Green；读取失败后仍能主动生成的恢复路径另 1 Red→Green。无待审核方向时更新不再要求确认替换空提案，仍保护未审核内容与草稿。
- [x] 简化默认页和导航，保留草稿、部分接受和可追溯详情 — 默认显示结果、短说明和一个主动作；分析详情与维护操作位于下方 More options，库概览有 Back to review；新建议先读选项再加入，长方向取消整段粗体，改名移入折叠项。原稿、接受、证据及替换确认回归通过，最终全插件 812 项 Green。
- [x] 完成真实桌面验收并交付测试构建 — 以当前旧提案、196 条实际索引标题及之前真实生成的 4 主题/7 方向在隔离 Obsidian 1.13.7 检查；新版 9 项通过、控制台无错误，并保存宽窄窗口前后截图。四包 typecheck、boundaries、build、diff check 通过，lint 0 error / 20 既有 warning。已备份安装测试插件，设置及原提案 SHA-256 未变。证据位于 `.artifacts/review-next-step/`；桌面生成响应受控，未重新请求真实模型，用户整体体验确认仍待补充。

## Phase verification

- 最终成功标准逐条核对；P12的blocked只在用户桌面确认后解除。
- 若仍缺用户确认，明确记录这一项，goal保持active，不伪称整个Helm已done。

## Abort / reshape triggers

- 真实模型持续违反既有合同：回到对应后续修正阶段，不放宽来源真实性。
- 用户观察到目标级体验不成立：记录证据后按Helm steer，不仅给测试打补丁。

# P6 — real-daily-and-desktop-acceptance

goal_ref: ../goal.md
created: 2026-09-07T04:07:50+08:00
updated: 2026-09-08T01:34:05+08:00
revision: 6

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

## Phase verification

- 最终成功标准逐条核对；P12的blocked只在用户桌面确认后解除。
- 若仍缺用户确认，明确记录这一项，goal保持active，不伪称整个Helm已done。

## Abort / reshape triggers

- 真实模型持续违反既有合同：回到对应后续修正阶段，不放宽来源真实性。
- 用户观察到目标级体验不成立：记录证据后按Helm steer，不仅给测试打补丁。

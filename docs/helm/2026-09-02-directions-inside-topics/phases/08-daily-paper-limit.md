# P8 — daily-paper-limit

goal_ref: ../goal.md
created: 2026-09-06T01:38:28+08:00
updated: 2026-09-06T09:12:03+08:00
revision: 2

## Outcome

每日新报告在筛选后按相关性保留最多 N 篇，默认 N=20；截断后的论文才抓取正文、参与详情选取和生成摘要。插件和 CLI 可配置同一正整数上限。

## Assumptions

- 用户已确定默认 20 篇，所有主题合计，不是每主题 20 篇。
- 上限属于共享 output 设置；旧配置缺失该字段时恢复默认值，非法交互/CLI 配置应明确拒绝。
- 筛选结果携带有限的 0–100 relevanceScore；skip 的分数为 0，非 skip 必须给分数。它衡量命中方向的相关性，不替代详情价值评分，也不新增淘汰阈值。
- 排除 ignored 后再按分数降序截断，同分按规范论文 ID 排序；所有筛选命中仍可进索引，但未入选者不关联这篇日报。
- 筛选缓存保存完整评分结果；仅改变上限无需重新筛选。prompt/result 契约分别 2→3，旧缓存失效已向用户说明。
- 已存在的完整日报仍是持久提交，不自动按新上限改写；生成范围未扩展到旧报告重写。

## Approach

先给共享设置与宿主入口补测试，再扩充筛选模型契约与缓存夹具，最后在单一管线截断点验证真实下游集合。每个独立 chunk 使用相应行为红测；不修改其他 active initiative，不提交、不推送、不开 PR。

## Chunks

### Chunk 1 — 共享上限、插件设置与 CLI 配置

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 旧配置没有默认 20、新设置未持久化/CLI 未解析或不校验非法值。
- Green check: 正整数规范化、插件加载/保存/两套设置界面、CLI TOML 解析与 init 输出均支持 output.maxDailyPapers / max_daily_papers；缺失字段取 20。
- regression checks: 设置相关 core/plugin/CLI 测试、typecheck。
- [x] implementation and tests accepted

### Chunk 2 — 筛选评分与缓存合同

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 合法 relevanceScore 四键结果被现有三键 decoder 拒绝；缺失、非有限、越界评分不能被接受；2 版缓存不可用于新合同。
- Green check: paper-filter、daily-filter-checkpoint-store；分数从模型或缓存一路传到 FilteredPaper，命中方向与既有拒绝/取消语义保持。
- regression checks: 管线与宿主缓存适配夹具、提示词快照、四包类型检查。
- [x] implementation and tests accepted

### Chunk 3 — 全局截断与消费端验证

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 指定上限 1 时两篇命中仍进入正文/详情/摘要；默认 20 时超过 20 篇仍全部处理。
- Green check: 跨主题 top-N、同分稳定排序、ignored 不占配额；日志记录省略数；正文抓取/详情候选/摘要与最终日报均只包含保留集合；缓存重跑更改上限可复用评分。
- regression checks: core / plugin / CLI 全量、typecheck、lint、boundaries、build；至少一次变异证明绕过截断会红。
- [x] implementation and tests accepted

## Phase verification

- 默认值 20 在两种宿主及旧配置中一致，非法上限不能使管线变成无上限。
- 评分是完整筛选结果的一部分，上限不是筛选缓存身份的一部分。
- 构建 main.js 与 styles.css 后安装到既有测试 vault，备份之前的构建；P7 真模型输出与桌面体验仍需用户确认。

## Abort / reshape triggers

- 若截断使未入选论文被当作已写日报，停止并修复索引提交边界。
- 若模型评分与方向命中混成详情质量/新颖性判据，收回到仅相关性排序。
- 若需要改写已提交日报或变更用户的已存主题，超出本阶段，先报告边界。

## Observed checks

- 共享正整数设置：19 项目标 Red 后全部 Green；插件加载/事务保存、两套 UI 与 CLI 配置/init 的目标红测均先观察再实现。CLI 临时放行 0 值会触发配置回归失败，恢复后通过。
- 筛选：新四键响应先被旧 decoder 拒绝；评分与范围/缺失/skip 约束的 11 项测试转绿，paper-filter 整文件 53 项通过，提示词快照已更新。缓存 5 项目标 Red→Green；缓存及宿主适配回归 133 项通过，保留小数评分且明确拒绝两个独立 v2 合同与缺分数记录。
- 管线：5 项目标红测分别观察到超上限、无全局排序、ignored 后名额错误或重试仍无上限；实现后全部 Green。真实索引只给最终入选论文挂日报路径；更改上限可复用完整评分。临时绕过 slice 时跨主题用例再次失败，恢复后通过。
- 手动抓取论文不经过筛选，DailyPaperWithContent 允许无评分，FilteredPaper 仍强制带评分；没有给手动抓取伪造分数。manual-fetch 与管线错误回归合计 57 项通过。旧应急报告测试的同分顺序按新的规范 ID 顺序更新，摘要/索引不变量保持。
- 最终全量：core **2043 passed / 2 skipped**、plugin **700 passed**、CLI **85 passed**；四包 typecheck、boundaries、lint（0 errors / 20 既有 warnings）和插件 build 通过；diff 检查通过。
- main.js / styles.css 已复制到既有测试 vault，cmp 校验一致；旧版本备份 `/tmp/arxiv-daily-before-topics-and-cap.l3uAoa/`。本轮没有调用真实 LLM 或执行用户的 Obsidian 操作；P7 桌面验收仍待用户。所有改动未暂存、未提交、未推送。

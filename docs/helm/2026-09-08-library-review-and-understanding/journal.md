# Journal

## 2026-09-08 — 新立项

- user intent: 新开 Helm 落实上轮评审问题并增加测试。
- evidence: 当前文件仍保留上轮复现的三个行为问题；既有 60,000 字符统一输入预算仍存在。工作区有其他会话正在进行的日报摘要修改。
- decision: 独立 initiative；先做四项已确认问题，再推进高优先级入口、预览和概览。项目范围/自动增量已发异步范围询问，不阻塞基础修复。
- verification strategy: 行为修正严格 Red→Green；容量改进同时保留覆盖正确性与规模基线；真实界面与语义结果单独记录。
- boundary: 不改其他 Helm 状态，不提交既有或本轮改动。
- next: P1 草稿 DOM 回归与跨来源证据回归。

## 2026-09-08 — P1 软件验收，进入 P2

- changes: 复审草稿跨选择/刷新保留，接受先保存，失败停止接受且可重试；提案/库身份在异步保存前后校验；本地代表证据从严格提案证据解析。
- evidence: 最初 3 DOM Red + 1 core Red；跨库场景另观察 1 Red。最终 plugin 78、core 65 Green；四包 typecheck 及后续 plugin typecheck 通过。
- scope: 异步范围询问无答复，按已告知推荐范围继续；未提交。
- next: P2 覆盖依据生成、严格存储及当前设置核对。

## 2026-09-08 — P2 软件验收，进入 P3

- changes: 提案保留 coverageEvidence 的方向身份/文字与完整论文分区；旧结果显示未核实，变更方向显示需复核；主题改名保留覆盖并显示当前名称。
- evidence: 2 core Red + 4 UI Red；core 78、plugin 82 Green，四包 typecheck 和 diff check 通过。
- next: 以 500/1000 篇完整证据容量回归推动有界分批与可追溯综合。

## 2026-09-08 — P3 软件验收，进入 P4

- changes: 完整摘要分批传输，中间研究线索综合并保留本地全部成员；方向成员上限与1000篇输入规模统一；生成契约记录新策略和提示词版本。
- evidence: 500/1000 两个容量Red→Green；172 core、10 plugin生成Green，core typecheck和boundaries通过。修订四条固定旧预算策略的测试；取消测试最初误用AbortError，改为既有RunCancelledError，未作为行为Red。
- limitation: 语义质量与真实桌面留在P7，不能由脚本模型响应证明。
- next: P4 从研究主题页直接进入文献库生成/复审。

## 2026-09-08 — P4 软件验收，进入 P5

- changes: 主题设置提供文献库入口，按现状选库/索引/复审；授权及取消沿用现有路径。
- evidence: 2 Red→174设置测试Green，plugin typecheck通过。
- next: 当前方向草稿的只读匹配预览与arXiv分类核对。

## 2026-09-08 — P5 软件验收，进入 P6

- changes: 当前草稿可预览最多20篇库样本，复用日报筛选合同并显示命中方向、分类未知/不覆盖；预览不持久化任何结果，方向修改后预览过期。
- evidence: core 3行为Red→Green、DOM/controller Red→Green、分类变更守卫Red→Green；plugin36项和四包typecheck通过。
- next: 持续可浏览的库概览与按需证据打开。

## 2026-09-08 — P6 软件验收，进入 P7

- changes: 新Library overview页签，分析日期与范围、主题方向概览、覆盖及未覆盖明细；证据标题可经既有scoped PDF opener打开；切换页签保留草稿。
- evidence: 2 DOM Red和controller Red后plugin88 Green，四包typecheck通过。构建通过，lint0 errors/20既有warning。全工作区测试进行中。
- next: /tmp隔离Obsidian宽窄桌面检查；冻结真实库首次/已有方向模型实验。

## 2026-09-08 — 独立审查修正与最终验收

- independent review: 发现刷新读盘失败清草稿、旧预览分类未失效、文字去重接受误报改动、代表选项包含已覆盖论文。逐项核对并加回归；审查者复核无新增阻断。
- corrections: 暂时无法加载提案时保留草稿身份；刷新/重试清预览，分类变化标过期；概览按稳定id或接受使用的规范化文字核对；排除已覆盖代表选项。真实截图另暴露预览重绘折叠编辑区，观察Red后保留展开状态。
- tests: 全workspace3148通过/2既有skip；审查修正后全plugin798通过；最终相关core113通过、混合来源review27通过、详情展开复审81通过及最后32通过。四包typecheck、后续plugin typecheck、boundaries、build和diff check通过；lint0错误/20既有警告。
- real model: 冻结真实库189篇，首次8调用形成4主题7方向，165成组论文完整归属；已有方向8调用，163覆盖、2篇形成一个新增方向；两者均24未覆盖。完整摘要读取避免旧统一截短，代价为额外模型调用。
- real desktop: /tmp独立vault中13项通过，7截图，控制台0错误。覆盖1440/560宽度、概览、草稿、真实保存接受、实际PDF打开、预览展示和设置入口。初期脚本语法/目录夹具/弹窗过渡错误已定位修正后重验，不作为产品通过证据。
- acceptance: 上述证据逐项满足新Helm8条成功标准；保留单库语义与受控桌面预览的验证限制。旧Helm用户桌面确认项不变。
- artifacts: `.artifacts/library-review-and-understanding/` 包含说明、日志、模型结果、截图与临时验收脚本。新Helm关闭，所有改动未提交，用户库未改写。

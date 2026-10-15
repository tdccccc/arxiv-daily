# Journal

## 2026-10-15 — note（P1 完成）

- evidence: 用户 2026-10-15 对四个问题均选 A（novelty 基准=按需检索最相似库论文；覆盖全部日报论文；增量=本地提醒+只扫新增；不做拆分合并）。子代理写出 ADR 0015 与旧 ADR / CONTEXT 注记，`git diff --check` 通过，交叉引用一致。
- change: 验收时发现 ADR 0015 §3 原稿以「当前提案的证据」为提醒基准：只扫新增的重扫替换提案后，老论文会被重新算作新增。owner 改为「已被任一次扫描计入的论文集合」，新提案继承该集合；§4 同步写明替换含未审候选时沿用既有确认。CONTEXT 的 Direction suggestion 同步。
- disposition: ADR 0004 §3 仍有「代表论文」的过时表述，未改（超出 P1 范围，P3 清理文档时一并处理）。旧提醒阈值 3 已由前期调研在 `plugin/main.ts:281`（孤儿常量 `INCREMENTAL_BUFFER_TRIGGER`）确认。
- next: P2 规划（子代理只读出计划，写入 `phases/02-*.md` 后开始执行）。

## 2026-10-15 — note（P2 授权口径）

- evidence: 用户选 A：「尽量减少用户的使用感知和授权次数」。
- change: 旧的文献库处理授权继续有效，不改指纹、不要求重新授权；Chunk 5/6 只更新插件弹窗与 CLI disclosure 文案，写明日报会把检索到的库论文标题与摘要发给同一模型端点。
- disposition: 不新增开关、不做授权迁移。
- next: Chunk 1–2 由子代理执行中。

## 2026-10-15 — note（P2 软件部分验收）

- evidence: Chunk 1–6 均由子代理按 Red→Green 交付，owner 复核：typecheck 0、boundaries OK、根测试与 test:workspaces 全绿（CLI 458、插件 1059、core 分片全过）、插件 build OK、lint 0 error / 16 warning（与基线相同）、diff --check 干净。
- change: 无计划变更。CLI 授权文案暂放 CLI 本地 i18n（子代理不得改 core），可在 P3 移入共享翻译目录。
- disposition: Phase verification 中「真实库运行、重跑命中检查点、真实 Obsidian 检查」尚未做；需要调用用户真实模型，待用户决定。P2 保持 active。
- next: 询问用户真实验证方式；P3 计划由只读子代理并行起草。

## 2026-10-15 — L1 adjust（P2 真实运行暴露的问题）

- evidence: 界面验收（/tmp/novelty-eval/ui-report.md）：固定场景 9/9、acceptance 工具测试 427 pass、真实模型探索 8 任务全过，两条疑点均为工具限制；Dashboard 对比论文只显示 key 不显示标题。真实质量评估（/tmp/novelty-eval/quality-report.md）被阻塞：(1) 词法检索把标题+整段摘要作查询，真实论文 81–191 个不同词，超过 BM25 的 64 词上限，20/20 检索失败——假端口单测未覆盖；(2) 端口只收 catalog 中的 arXiv 论文，用户测试库 baseline_library_207 识别出 0 篇 arXiv，test_library 357 篇中仅 146 篇，file: 兜底文档占前 6 命中的 57%；(3) 系统性失败逐篇打 20 条堆栈；摘要缺失被记成 not comparable input；(4) 「重跑命中检查点」不成立：日报提交后检查点即被清理，检查点复用只能由单测和中断续跑证明。截词后的检索人工判断约 31% 相关 / 35% 弱相关 / 33% 不相关（仅看标题）。
- change: P2 追加 Chunk 7–10：查询受词数上限约束并以真实 BM25 测试；非 arXiv 库论文（file:）以知识库记录中的标题摘要作为对比基准，日报与 Dashboard 都显示对比论文标题；系统性失败只警告一次、跳过原因如实；以检索实测比较 lexical 与本地 hybrid 后定本地模式。Phase verification 的「重跑不再调模型」改为单测+中断续跑证据。
- disposition: 已完成的 Chunk 1–6 保留；新增验收工具改动（scripts/acceptance 下）保留。测试库已装入当前构建（备份 /tmp/arxiv-novelty-before-install-20261015-111520/），修复后需重新安装。
- next: 子代理实现 Chunk 7+9；另一子代理只读规划 Chunk 8 并实测检索模式（不调模型）。

## 2026-10-15 — note（Chunk 7/9 验收，Chunk 10 实测定案）

- evidence: Chunk 7/9 子代理 Red→Green，owner 复跑 4 文件 50 项 Green、typecheck 通过。Chunk 10 实测（research.md §2）：20 篇×前 6 命中，lexical 相关 22 / 弱 35 / 无关 63，本地 hybrid 25/38/57，差异在单人判读噪声内；接受 file: 文档后可用基准由每篇约 2.5 篇升到 6 篇，有至少一篇相关基准的论文由 8/20 升到 11/20；约一半日报论文在任何模式下都无相关库论文，质量依赖模型拒答。
- change: 本地模式保持 lexical（无质量增益证据，且 hybrid 把嵌入模型加载带进宿主，正是 ADR 0007 的风险）；Chunk 10 验收。Chunk 8 方案定为 8a+8b（research.md §1）；旧插件读新标记当天不显示新颖性，功能未发布，owner 接受。新增 Chunk 11：词法检索单次约 3.8 秒、约 70% 在纯 JS sha256。
- next: Chunk 8 与 Chunk 11 由两个子代理并行（文件不重叠）。

## 2026-10-15 — note（Chunk 11 验收）

- evidence: 子代理报告 /tmp/helm-p2-c11-report.md：词法检索单次中位 3784→965 ms，首查 4049→1213 ms，20 篇合计 76→约 20 秒；结果与改前逐字节一致（cmp 三次），完整性校验一项未删。根因为纯 JS sha256 吞吐低（46 MiB/s）及桶值重复计算；sha256 重写到 142 MiB/s（与 node:crypto 全长度比对），桶值缓存与外提。owner 复跑 5 个文件 111 项 Green。
- change: 无。内嵌块校验的第二遍保留（去掉约省 0.2 s/次，但需改既有防损坏测试，不在本阶段决定）。
- next: 等 Chunk 8 交付后全量回归、重新构建安装测试库、真实模型质量评估。

## 2026-10-15 — note（Chunk 8 验收）

- evidence: 子代理报告 /tmp/helm-p2-c8-report.md：8a（file: 文档作基准）与 8b（标记 v2 携带标题、Dashboard/工作台显示标题）各自 Red→Green。owner 复核：typecheck 0、boundaries OK、根测试与 test:workspaces 全绿（CLI 461、插件 1064）、lint 0 error / 16 warning、全量 build OK、diff --check 干净。提示合同版本 2→3；标记上界 10000→21000 码元，v1/v2 最坏情况有测试钉住。Dashboard 分隔符由逗号改分号（标题可含逗号）。
- change: 无计划变更。
- next: 快照构建到 /tmp/novelty-eval/build2；并行：真实 Obsidian 固定验收（novelty 场景）+ 重新安装测试库并用真实模型做质量评估。

## 2026-10-15 — L1 adjust（真实质量评估 2）

- evidence: /tmp/novelty-eval/quality-report2.md：用户测试库 baseline_library_207 全靠 file: 文档（203 篇可检索）；2026-10-14 日报 20 篇，12 篇写出说明、8 篇拒答；8 篇拒答经复查全部正确；12 条中 有用 5 / 勉强 5 / 误导 2（与无关或仅同领域论文比较），说明均忠于摘要、无全文臆断。界面验收 2：固定场景 9/9、工具测试 427/427、探索任务通过，Dashboard 显示标题。阶段耗时 121.7 秒，模型仅约 17 秒。说明为英文而日报为中文。
- change: 追加 Chunk 12（提示收紧+说明语言）、13（阶段耗时）、14（逐篇结果日志+确定性采样）。
- disposition: 不做：文件标题噪声过滤（约 13% 标题为「1 Introduction」等抽取噪声，属于标题抽取质量，另立后续）；arXiv 扫描 429 重试耗时（非本功能）。
- next: Chunk 12 与 13 并行（文件不重叠），14 随后；之后再做一次只针对新颖性的真实模型复评。

## 2026-10-15 — note（Chunk 13 结论）

- evidence: /tmp/helm-p2-c13-report.md：经真实端口的可重复测量（/tmp/novelty-eval/perf2/run.sh）每次检索约 0.54 秒，结果逐字节一致；122 秒主要是 20 次串行流式模型调用（约 5.4 秒/次）。
- change: Chunk 13 以「无需改动」验收。子代理提议的 3 路并发不采纳：每日 LLM 调用串行是既有约定（sequential-daily-summaries），且有限流风险；新颖性约给日报增加 2 分钟，记录为已知代价。
- next: 等 Chunk 12+14 交付，然后真实模型只复评新颖性。

## 2026-10-15 — note（Chunk 12/14 验收）

- evidence: /tmp/helm-p2-c1214-report.md：提示收紧（同一具体问题/对象/场景才算可比，综述与大型目录不算，仅相邻则拒答）、说明语言随日报摘要语言（沿用 normalizeSummaryLanguage，语言进入检查点指纹）、提示合同 3→4；逐篇 info 结果行。owner 复跑 2 文件 123 项 Green、typecheck 0、全量 build OK。新颖性调用本已 temperature 0；思考模式下客户端忽略温度，属全局行为，不在此改。
- next: build3 快照；真实模型只复评新颖性（约 20 次调用，复用 run2 的 20 篇与 perf2 端口测量环境）。

## 2026-10-15 — L1 adjust（提示收得过紧）

- evidence: /tmp/novelty-eval/quality-report3.md：同 20 篇、同检索结果，仅换提示：写出 12→0，误导 2→0，有用 5→0；先前 8 条正确拒答保持；16496（同一距离对偶问题、不同数据）明确误拒；排除目录类导致 eRASS1 交叉匹配类比较丢失。阶段耗时 49.6 秒。
- change: 追加 Chunk 15：在调参集上迭代至多 3 个提示变体，并用同日未入选论文 20 篇作留出集验证，预算 ≤100 次调用。
- next: 子代理执行 Chunk 15。

## 2026-10-15 — note（P2 完成，P3 开始）

- evidence: Chunk 15 提示校准（/tmp/novelty-eval/quality-report4.md）：调参集 20 篇写出 5 条全有用、0 误导、15 条拒答全对，重复运行一致；留出集 A 20 篇全拒答且正确；附加留出集 B 10 篇写出 3 条（2 有用、1 勉强、0 误导）；中文说明核对 8 条均为中文；共 70 次调用。owner 复核：typecheck 0、boundaries OK、根测试与 test:workspaces 全绿、lint 0 error / 16 warning、全量 build OK、diff --check 干净。最终构建已装入测试库（备份 /tmp/arxiv-novelty-before-install3-20261015-162846/，data.json 哈希不变）。
- change: P2 done；goal 成功标准 2、3 勾选；成功标准 4 措辞改为「未被任何扫描计入」（与 ADR 0015 §3 一致）；P3 active，阶段文件取自先前只读规划。
- disposition: 已知代价：新颖性约给日报增加 1 分钟（P2 最终提示下 20 篇约 50 秒）；勉强类说明（模拟/数据发布论文被引用于不同问题）偶有出现。用户尚未在真实 Obsidian 中看过测试库效果。
- next: P3 Chunk 1–2（纯 core）交子代理。

## 2026-09-02 — 立 goal：承接 ADR 0012 / 0013 / 0014

- evidence: 前一个 goal（`2026-08-31-library-directions-drive-daily`）在 P1–P3、P6 落地后被废弃——不是没做完，是 ADR 0012 取消了「已确认方向」这个独立概念，它的 Intent 失去对象。三条新 ADR 定下了替代结构：方向住进主题（0012）、索引只覆盖标题与摘要（0013）、库提议如何归入主题（0014）。
- change: 新建本 goal，六个阶段按依赖排：先立数据结构与迁移，再降索引成本（让后续反复重建变便宜），然后筛选、首次提议、增量归入，最后端到端。
- disposition: **本 goal 与上一个最大的不同是迁移风险的方向变了**。上一个 goal 的对象是文献库子系统——从未发布，索引和提案文档可以随便重建；本 goal 的对象是 `topics`，**已发布的设置键**，还出现在 CLI 配置与用户文档里。所以「可回滚的迁移」进了成功标准而不是约束脚注。
- **三处刻意不在本 goal 内解决**：personal novelty 的新对比基准（ADR 0012 留为开放）、主题名是否作为硬门参与筛选（用户明确说筛选后面再谈）、ADR 0008 的全文授权深度是否退休（ADR 0013 标记，需回头单独看）。把它们写进 non-goals 而不是留白，是为了避免在实现中顺手替用户定掉。
- **两个阈值写成了硬约束**：粗聚类比例与提议新主题的相似度下限。ADR 0014 已声明它们没有可辩护的默认值；本 goal 要求默认值必须有实测依据，否则只能当调参旋钮交付。
- boundary: 只新建本 goal 与本条日志。上一个 goal 的代码不回滚，ADR 0009 与 CONTEXT.md 的新词汇不动。三个并行 active helm 未碰——但记下一处悬置：`2026-08-13-discovery-loop-and-library-insight` 的 P4「检索规模加固」在 ADR 0013 把索引砍到摘要级之后动机基本消失，是否收束由用户决定。
- next: 用户确认 goal 后置 active，写 P1（主题带方向列表与可回滚迁移）的详细计划。P1 不受 LLM 端点阻塞。

## 2026-09-02 — goal 置 active；「可回滚」定为影子字段常驻，P1 计划落盘

- evidence: 写 P1 之前先追问「可回滚」到底承诺什么，查出成功标准与代码实情对不上。ADR 0012 line 44 自己承认可逆性只在「主题恰好只有一条手写方向」时成立——**用户添加第二条方向的那一刻，回滚就悄悄失效了**，而没人定义过那之后它还指什么。代码这边更硬：`settings/migration.ts:34` 把 `topics` 原样透传加 cast，没有任何逐主题规范化的落点；`settings/validation.ts:163` 是无保护的 `topic.description.trim()`，字段一旦消失就是 TypeError 而非降级。
- change: 用户在三个选项（一次性降级脚本 / 影子字段常驻 / schemaVersion 硬拦）中选定**影子字段常驻**。`description` 留在 `data.json` 里，不变式为 `description === directions[0]?.text ?? ""`；`directions` 是新权威。goal 置 active（revision 2），P1 索引行置 active，计划落在 `phases/01-topics-hold-directions.md`。
- disposition: **这个选择把 P1 变成了纯 schema/迁移/界面阶段，管线一行不改**。筛选契约继续读 `description`，于是「`topicLines` 与迁移前逐字相同」成为 P1 最强的回归判据，同时 `detail-selector` / `diagnostics` / `onboarding` / `validation` 四个消费点都不必动。代价是文件里有一份冗余、且写入路径必须收敛到单一函数——两个真相来源各自同步就是这条设计的失败模式，已写进 abort trigger。
- **一处顺带排除的疑虑**：主工作区那个「behind origin/main 220」是本地 `main` 引用陈旧，与本分支无关；`feat/library-directions-drive-daily` 相对 `origin/main` 是 27 ahead / 0 behind，分叉点就是 `origin/main` 的头 `976c12b`。基线干净，不需要 rebase。
- **三处刻意留在 P1 之外**：`origin` 字段的取值域与是否展示（只留字段，不据它分支行为，遵 ADR 0012 §4 的「行为不可区分」）；方向如何进入筛选应答词表与日报如何标出命中的那一条（P3，P1 只保证每条方向有稳定 `id`）；personal novelty 基准（goal 非目标）。
- boundary: 只动 goal 状态、本条日志、新增 P1 阶段文件。一行实现代码未写，未提交、未推送。
- next: P1 Chunk 1 取红——`packages/core/tests/migration.test.ts` 断言老 topic 迁移后带 `directions` 且 `directions[0].text` 等于老 `description`。红在字段不存在。

## 2026-09-02 — P1 Chunk 1+2 done：方向落地，规范化收敛到一处

- evidence: 先取红两次。第一次红在 `directions` 不存在（5 条断言全红，`expected undefined to deeply equal []`）；实现后第二次红是**预期内的撞坏**——既有的 `returns the same topics when already in new shape` 断言 `topics` 原样透传，而它喂的其实是老形状。该断言的前提被 ADR 0012 取代，改断言不改源码。
- change: `Topic` 增 `directions: Direction[]`，`description` 降为派生的回滚影子（类型注释里写明「不得直接写入」）。`Direction` 为 `{ id, text, origin }`，`origin` 取 `manual | migrated | library`，**无任何分支读它**（ADR 0012 §4）。
- **Chunk 2 被 Chunk 1 拉着提前做了**。给 `Topic` 加必填字段后 typecheck 立刻暴露四个构造点：设置迁移、CLI 的 TOML 读取（`apps/cli/src/config.ts:302`）、插件的新建主题与套模板。这正是阶段文件 abort trigger 写的「影子同步需要写在一个以上的地方」——所以没有在两处写下与 `description` 矛盾的 `directions: []`，而是当场把规范化收敛成 `settings/topics.ts` 的 `normalizeTopic`，四处全改走它。CLI 的 `detail` 默认值（`true`，与共享默认相反）显式留在调用点，CLI 71/71 未动。
- **一处真实的设计冲突，判据因此改了**：`{ description: "stale", directions: [] }` 既可能是老文件、也可能是用户在新界面刚删掉最后一条方向。按「空列表就从 description 复活」处理会让**最后一条方向永远删不掉**——而设置页正是这样调用规范化的。改为**按键是否存在判定**：`directions` 键在（哪怕空数组）即为权威，键不在才迁移 `description`。这条是取红逼出来的，不是设计时想到的。
- disposition: 代价是「降级—编辑—再升级」会丢那次编辑。判为可接受：它是「directions 是权威」的固有代价，且失败**可见**（主题变空，配置检查会报 description is empty），不是静默算错。已记进 open questions；若日后不可接受，出路是给影子加写入指纹，不在 P1 内做。
- validation: core 全分片绿、plugin 705/705、CLI 71/71、typecheck 四包、lint 0 error（20 warning 为既有基线）、check:boundaries OK。
- boundary: 只动设置层与四个构造点。筛选契约、日报格式、检索排序未动——`paper-filter-contract.ts` 仍读 `description`，这是 P1 的设计前提。ADR 0005/0007/0008 未动，授权面未变。三个并行 active helm 未碰。
- **本 chunk 没有证明的事**：界面。设置页仍是那个单一 textarea，方向列表要到 Chunk 3 才有。用户看不到任何变化。
- next: Chunk 3——设置页把 textarea 换成方向列表。这是 P1 里唯一有界面的 chunk，按上一轮教训，测试绿不构成交付证据，必须由用户在真实 Obsidian 里看过。


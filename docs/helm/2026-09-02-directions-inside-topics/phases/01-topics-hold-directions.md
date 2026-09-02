# P1 — topics-hold-directions

goal_ref: ../goal.md
created: 2026-09-02T13:07:28+08:00
updated: 2026-09-02T13:07:28+08:00
revision: 1

## Outcome

设置里的一个主题带一列方向，每条一行文字；已有主题的 `description` 迁移为第一条方向。`description` 作为**影子字段常驻**——旧版插件与旧 CLI 读同一个 `data.json` 行为逐字不变，因此任何时刻降级都无损。管线（筛选、日报、详情选取）在本阶段一行不改。

## Assumptions

- **影子字段是回滚的载体，不是遗留物**（2026-09-02 用户决定）。不变式：`description === directions[0]?.text ?? ""`，由 core 的单一规范化函数维持。ADR 0012 line 44 只承诺「主题恰好只有一条手写方向时可逆」；影子字段把这个窗口从「用户添加第二条方向之前」拉长为「永远」，代价是文件里有一份冗余。
- **P1 不碰筛选契约**。`packages/core/src/pipeline/paper-filter-contract.ts:107` 继续读 `description`，产出的 `topicLines` 必须与迁移前**逐字相同**。方向真正参与筛选是 P3。这一条同时是 P1 最强的回归判据。
- 同理不动的消费点：`packages/core/src/pipeline/detail-selector.ts:155`、`packages/core/src/services/diagnostics.ts:132`、`plugin/src/onboarding.ts:41`、`packages/core/src/settings/validation.ts:163`。它们都读 `description`，影子字段保证其语义不变。**若某个消费点被发现必须看到全部方向，那是 P3 的活，不在 P1 顺手改。**
- **`packages/core/src/settings/migration.ts:34` 今天把 `topics` 原样透传加 cast**，不存在逐主题规范化的落点。P1 必须先建这个落点——它同时是老数据迁移与影子同步的唯一入口。
- 方向的 `origin` 只作来源留痕，**P1 不据它分支任何行为**（ADR 0012 §4 要求接受后的库方向与手写方向在行为上不可区分）。

## Approach

先在 core 定 `Direction` 与 `Topic.directions`，把 `migrateArxivSettings` 从透传改成逐主题规范化，让老数据在读入时就长出 `directions` 并维持影子。再把设置页那个单一 textarea 换成方向列表。最后用一条「降级」测试把「可回滚」这个验收标准钉成可执行判据。

## Chunks

### Chunk 1 — `Direction` 类型与迁移长出 directions

- change kind: behavior change（schema + 迁移）
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `packages/core/tests/migration.test.ts` 新增——喂入只有 `description` 的老 topic，产出的 topic 带 `directions`，`directions[0].text` 等于老 `description`，`directions[0].id` 非空，且 `description` 逐字不变。红在 `directions` 字段不存在。
- Green check: `npm run test --workspace @arxiv-daily/core -- migration`
- regression checks: `validation.test.ts`、`real-corpus-migration.test.ts` 仍绿；`npm run typecheck`
- exception: 无
- **撞坏一条既有测试，是预期内的**：`returns the same topics when already in new shape` 断言 `topics` 原样透传，而它喂的其实是**老**形状（有 `description`、无 `directions`）。该断言的前提被 ADR 0012 取代，已改为「身份字段不变 + directions 由 description 派生」。改的是断言，不是放宽源码。
- [x] implementation and tests accepted

### Chunk 2 — 影子同步收敛到单一函数

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 断言三件事——(a) 改 `directions[0].text` 后规范化产出的 `description` 跟着变；(b) 删掉唯一方向后 `description` 变为 `""`；(c) 喂入 `description` 与 `directions[0].text` 不一致的脏数据时**以 `directions` 为准**。红在没有这个函数、或 `description` 不跟随。
- Green check: `npm run test --workspace @arxiv-daily/core -- settings-topics`
- regression checks: core 全量、CLI 71/71
- **本 chunk 被 Chunk 1 拉着提前做了**。给 `Topic` 加必填字段后 typecheck 暴露出四个构造点：`migration.ts`、`apps/cli/src/config.ts:302`（CLI 自己读 TOML 造 Topic）、`plugin/src/settings/tab.ts` 的新建主题与套模板。这正是本阶段 abort trigger 写的「影子同步需要写在一个以上的地方」，所以没有在两处写下与 `description` 矛盾的 `directions: []`，而是当场把规范化收敛到 `packages/core/src/settings/topics.ts` 的 `normalizeTopic`，四处全部改走它。CLI 的 `detail` 默认值（`true`，与共享默认相反）显式保留在调用点。
- **取红时发现一处真实的设计冲突，判据因此改了**：`{ description: "stale", directions: [] }` 既可能是老文件，也可能是用户在新界面里**刚删掉最后一条方向**。原实现按「列表为空就从 description 复活」处理，会让最后一条方向永远删不掉。改为**按键是否存在判定**：`directions` 键在（哪怕是空数组）即为权威，键不在才从 `description` 迁移。代价记在下面的 open questions。
- exception: 无
- [x] implementation and tests accepted

### Chunk 3 — 设置页编辑一列方向而不是一个 textarea

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `plugin/src/settings/tab.ts:2534–2554` 的单一 textarea 换成方向列表。新增测试——渲染带两条方向的主题得到两个单行输入与增/删控件；改第一条后落盘的 `description` 同步。红在只渲染出一个 textarea。
- Green check: `npm run test --workspace obsidian-arxiv-daily -- settings-declarative-tab`
- regression checks: 既有设置页断言全绿（含 0.4.3 刚修的 masked-reveal 密钥输入）；plugin 全量 710/710
- **第一次取红是假的，值得记**：测试原本走 `tab.refreshSettings()`，但它在 1.13+ 走声明式路径，而该路径的 `update()` 被同文件其它测试打了桩——结果渲染出 **0 张主题卡片**。于是「不再有 description textarea」那条**因为什么都没渲染而绿了**。改用公开的 `tab.renderTopicRow(new Setting(...), 0)` 真渲染后才得到诚实的红（`expected <textarea> to be null`）。与 journal 里那条宽泛 `toThrow()` 同类：**红必须红在被测行为上**。
- 影子的派生规则由 core 的 `deriveTopicDescription` 提供，设置页调用它而不是自己写一遍——否则界面就成了影子的第二个作者。
- 撞坏两条既有测试，都是契约变更的直接后果：一条夹具里的 topic 没有 `directions`（改夹具走 `normalizeTopic`）；一条源码文本断言指着 `descId`（改为 `dirId`），并把「输入时不被重渲染夺走焦点」这条保护从 description 挪到方向输入框上。
- 样式：`styles.css` 中 `topic-description` 的三处引用改为方向列表，并补上行内布局与增删按钮样式。
- exception: 无
- [ ] **实现与测试已绿，但未交付**——P1 唯一有界面的 chunk，按上一轮教训必须由用户在真实 Obsidian 里打开看过才算数。

### Chunk 4 — 模板与「新建主题」产出方向

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `packages/core/src/settings/topic-templates.ts` 三套模板与 `plugin/src/settings/tab.ts:916` 的新建路径。断言——套用 `astro-ml` 模板后每个主题恰有一条方向且 text 等于原描述；新建空主题得到零条方向且 `description === ""`。红在模板产出的主题没有 `directions`。
- Green check: `npm run test --workspace @arxiv-daily/core -- settings` 与 plugin 设置页测试
- regression checks: core + plugin 全量
- exception: 无
- [ ] implementation and tests accepted

### Chunk 5 — 「可回滚」被钉成可执行判据

- change kind: behavior change（验收覆盖）
- strategy: 先取红，再取绿
- Red / baseline signal: 新增降级测试——取新版写出的 settings，**只经由旧版会读的字段**（`name` / `tag` / `description` / `detail`）走 `validateFilterConfig` 与 `buildPaperFilterRequest`：不抛错、不报 `description is empty`、且 `topicLines` 与迁移前逐字相同。红在迁移实现之前，也红在影子字段被写坏时（例如误用 join）。
- Green check: `npm run test --workspace @arxiv-daily/core -- validation paper-filter-contract`
- regression checks: `npm run test --workspace arxiv-daily`（CLI 71/71——CLI 读同一份配置，是「插件与 CLI 行为一致」这条约束的硬判据）
- exception: 无
- [ ] implementation and tests accepted

## Phase verification

- core 全量、plugin 全量、CLI 71/71、`npm run typecheck`（四包）、`npm run lint` 0 error、`npm run check:boundaries`。
- **不预设桌面验收**：P1 改的是设置页里已有验收覆盖的区域，但方向列表是新控件。是否需要新增桌面场景在 Chunk 3 完成后判断——不重复上一轮「自己写下的验收条件自己跳过」的错误，若判定需要就必须由用户在真实 Obsidian 里看过。
- 端到端日报仍被 LLM 端点不可达阻塞，不属本阶段。

## Abort / reshape triggers

- 若影子同步需要写在一个以上的地方，停下先收敛写入路径——两个真相来源各自同步，正是这条设计的失败模式。
- 若某个 `description` 消费点在 P1 内就必须看到全部方向，停下把它移交 P3，不要在本阶段扩大契约变更。
- 若 `topicLines` 无法与迁移前逐字相同，说明影子字段的定义不对（例如误用 join 而非取第一条），回到 Chunk 2，**不要改测试**。

## Open questions（P1 不顺手定掉）

- **「降级—编辑—再升级」会丢那次编辑**。`directions` 键一旦存在即为权威，所以用户降级到旧版后改动 `description`、再升回新版，那次改动被派生值覆盖。这是「directions 是权威」的固有代价，且失败是**可见的**（主题变空，`validateFilterConfig` 会报 description is empty），不是静默算错。若日后判定不可接受，唯一的出路是给影子字段加一个写入指纹——不在 P1 内做。

- `origin` 的取值域，以及它将来是否在界面上显示——本阶段只留字段，不据它分支行为。
- 方向如何进入筛选的应答词表、日报如何标出命中的那一条——P3。P1 只保证每条方向有稳定 `id`。
- personal novelty 的对比基准——goal 的非目标，ADR 0012 留开放。

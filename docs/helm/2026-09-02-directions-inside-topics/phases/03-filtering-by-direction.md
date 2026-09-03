# P3 — filtering-by-direction

goal_ref: ../goal.md
created: 2026-09-03T23:21:47+08:00
updated: 2026-09-03T23:21:47+08:00
revision: 1

## Outcome

日报的筛选按**方向**判定而不是按主题的一句话描述判定，且每一篇入选论文都说得出「是主题 X 下的哪几条方向选中了它」——可见地写在日报里，也机器可读地留在标记里。筛选契约（提示词与结果）因此变更一次，缓存的分类结果显式作废。

## Assumptions

- **命中粒度：一个主题 + 该主题下的若干条方向**（2026-09-03 用户决定）。归组仍由单一 `category` tag 决定，日报分节结构一行不改（ADR 0010 §3：方向不变成报告分节）。一篇论文可以命中同主题下的多条方向，**不跨主题**。
- **主题名仍只作标签，不参与判定**（2026-09-03 用户决定，收掉 ADR 0012 line 45 与 goal Non-goals 留下的那个未决项）。硬门需要一个「像不像本主题」的判据，属于 goal Constraints 明令不得拍脑袋定的阈值类决策；它是随时可加的增量，不绑在 P3。
- **画像文档驱动的第二个分类器不在本阶段退休**（2026-09-03 用户决定）。`personalized-paper-filter.ts` 与 `personal-library` 类别原样保留，留到 P4/P5 随画像文档一起处理。**推论：本阶段必须让两种来源在日报里可区分**——主题方向的命中不能借用 `PaperDiscoveryProvenance.directions`，那个字段的语义是「库方向 + 其代表论文证据」，而 ADR 0012 §4 说被接受的方向不留证据。
- **`description` 影子字段继续存在且继续被别的消费点读**。P1 把它钉成 `directions[0].text` 的影子，`detail-selector.ts:155`、`validation.ts:163`、`onboarding.ts`、`diagnostics.ts` 都还读它。**本阶段只改筛选那一个消费点**；详情选取是否也该看到全部方向，是独立的一次判断，不在 P3 顺手做。
- **P1 钉死的回滚契约会被本阶段正面推翻一条**。`packages/core/tests/settings-rollback.test.ts:68` 断言「迁移后筛选提示词逐字不变」——那是 P1 为「不碰筛选」立的护栏，不是 P3 要守的承诺。goal Constraints 已经认下这次代价：筛选契约变更让已缓存分类结果整批失效。**要守住的是数据侧的可回滚**（旧构建读同一个 `data.json` 仍能工作），不是提示词的逐字不变。
- 单测里的 LLM 是桩。**桩能通过不等于真模型能通过**——严格解码一旦收紧到「必须回报合法方向标识」，真模型不遵守就是整批失败。因此本阶段的 Phase verification 里有一条真实端点的小批量实测，不能只靠单测收工。

## Approach

先把筛选契约改成方向驱动并让结果带回命中的方向标识（core 单点），同时显式抬两个契约版本号让旧缓存作废；再重新判定 P1 那条回滚断言，把「可回滚」重新表达在数据侧；然后让命中方向沿 `FilteredPaper → summarizer → assembler → 渲染` 流到日报，可见行与机器标记各出一份；最后补退化行为与 CLI 一致性，并用真实端点验一次严格解码在真模型上站得住。

## Chunks

### Chunk 1 — 筛选提示词按方向出题，结果带回命中的方向

- change kind: behavior change（提示词契约 + 结果契约）
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `packages/core/tests/paper-filter.test.ts` 新增——(a) `buildPaperFilterRequest` 产出的 system 消息里，每个主题下逐条列出它的方向文本与各自的标识，`topic.description` 不再是判据来源；(b) `decodePaperFilterRecords` 接受 `{id, category, directions:[...]}`，**方向标识不属于所选主题时判违约**；(c) 重复方向标识判违约。红在今天的 `- ${t.tag}: ${t.description}` 只有一行、且 record 多一个键就被 `hasExactKeys` 拒掉。
- Green check: `npm run test --workspace @arxiv-daily/core -- paper-filter`
- regression checks: `npm run test --workspace @arxiv-daily/core`；`npm run typecheck`
- 方向标识用什么形态由本 chunk 定：必须能**唯一映射回一条方向**，且未知标识是违约而不是被静默丢弃——与既有 `knownIds` / `validTags` 的严格解码同形。
- exception: 无
- [ ] implementation and tests accepted

### Chunk 2 — 两个契约版本号显式抬，旧缓存作废是被断言的

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `DAILY_FILTER_PROMPT_CONTRACT_VERSION` 与 `DAILY_FILTER_RESULT_CONTRACT_VERSION` 各 1→2。新增测试——按旧版本号存下的 checkpoint **不得**被复用。红在版本号未抬时旧 checkpoint 照样命中。
- Green check: `npm run test --workspace @arxiv-daily/core -- daily-filter-checkpoint-store`
- regression checks: core 全量
- **变异检验（本 chunk 必做）**：只退回其中一个版本号，对应那条测试必须仍然红。P2 栽过一次——两个版本号写在同一条测试里，退回一个照样绿，那条测试实际只钉住「至少有一个字段变了」。每个版本号各一条测试。
- exception: 无
- [ ] implementation and tests accepted

### Chunk 3 — 回滚契约重新判定：守数据，不守提示词

- change kind: behavior change（测试契约重新表达）
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `settings-rollback.test.ts:68`「迁移后筛选提示词逐字不变」在 Chunk 1 之后必然红——**这是预期内的红，且它证明 Chunk 1 真的改到了筛选**。重写为两条：(a) 旧构建只读那四个字段时，`validateFilterConfig` 仍过、`migrateArxivSettings` 往返仍稳定（数据侧可回滚，逐字不变）；(b) 新旧提示词**必须不同**，且差异由抬起的契约版本号显式宣告。其余四条断言原样保留。
- Green check: `npm run test --workspace @arxiv-daily/core -- settings-rollback`
- regression checks: `real-corpus-migration.test.ts`、`migration.test.ts`、`validation.test.ts` 全绿
- **不允许的做法**：把那条断言删掉了事，或放宽成「大致相同」。P1 的验收标准是「迁移可回滚」，本 chunk 要让它继续以可执行形式存在，只是换到它真正该待的那一侧。
- exception: 无
- [ ] implementation and tests accepted

### Chunk 4 — 命中方向沿管线流到日报组装

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `FilteredPaper` 带上命中的主题方向（tag + 方向标识 + 方向文本），经 `summarizer.ts:107` 与 `daily-summary-assembler.ts` 透传到组装层，`preflightDailySummaryAssembly` 对其做与既有 provenance 同级的校验（不合法即抛，不静默丢）。红在组装层收不到这个字段。
- Green check: `npm run test --workspace @arxiv-daily/core -- daily-summary-assembler`
- regression checks: core 全量；`npm run check:boundaries`
- **新字段与 `discoveryProvenance` 并列而不是塞进去**：后者的 `directions[].representatives` 至少一条是硬性约束，而主题方向按 ADR 0012 §4 不留证据，塞进去只能靠放宽那条约束，等于让两种语义共用一个校验器。
- exception: 无
- [ ] implementation and tests accepted

### Chunk 5 — 日报写出命中的方向：可见一行 + 机器标记

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 渲染带方向命中的论文块时，(a) 可见行说出主题与命中的每一条方向文本（中英双语，走既有 `escapeDiscoveryProvenancePlainText` 的转义，方向文本是用户输入，不可信）；(b) 机器标记可被解析回等价结构；(c) **既有报告里的 v1 标记仍能被解析**。红在渲染不产出方向、或旧标记解析回归。
- Green check: `npm run test --workspace @arxiv-daily/core -- daily-summary-rendering`；`-- discovery-provenance-marker`
- regression checks: `daily-summary-parser` 与 `paper-index` 相关测试全绿；core 全量
- 标记形态由本 chunk 定：扩 `discovery-provenance` 到 v2 并保留 v1 解析，或另起一个前缀。判据是**磁盘上已有的日报不能因为升级而解析失败**——`parseDailyReportDiscoveryProvenance` 会把它判成 invalid 并影响派生索引。
- exception: 无
- [ ] implementation and tests accepted

### Chunk 6 — 退化行为：没有方向的主题、模型一条都不命中

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: (a) 一个主题的方向被删空时（影子 `description` 为 `""`），筛选与配置检查的行为被钉死——今天 `validation.ts:163` 报 “description is empty”，在新模型下这句话该说的是「这个主题一条方向也没有」；(b) 模型给出 `category` 却回报空方向列表时的行为被显式钉死（保留该论文并只标主题，还是判违约），不留给巧合。
- Green check: `npm run test --workspace @arxiv-daily/core -- validation`；`-- paper-filter`
- regression checks: core 全量；plugin 全量
- exception: 无
- [ ] implementation and tests accepted

### Chunk 7 — CLI 侧行为一致

- change kind: behavior change（跟随）
- strategy: strict Red-Green-Refactor
- Red / baseline signal: CLI 从 TOML 造 `Topic`（`apps/cli/src/config.ts:302` 走 core 的 `normalizeTopic`），筛选走同一条 core 路径因此自动跟随。新增一条 CLI 测试钉住：从 TOML 配置出发产出的筛选请求里带着方向。红在 CLI 侧仍只喂 description。
- Green check: `npm run test --workspace arxiv-daily`
- regression checks: 四包全量；`npm run typecheck`；`npm run lint`
- goal Constraints 要求插件与 CLI 两侧行为一致；ADR 0003 的两产品边界不因此变动。
- exception: 无
- [ ] implementation and tests accepted

## Phase verification

- 四包全量测试、`npm run typecheck`、`npm run lint`（基线 0 error / 20 warning）、`npm run check:boundaries` 全过。
- **真实端点小批量实测（不可省）**：用真实 LLM 端点对一小批真实 arXiv 论文跑一次筛选，确认真模型能稳定回报合法的方向标识、严格解码不会整批失败。单测里的 LLM 是桩，桩通过不构成这条证据。若真模型不遵守，是 Chunk 1 标识形态的问题，就地调整而不是放宽解码。
- 每个行为 chunk 至少一次变异检验：拿掉被测那一行，对应断言必须红在**它自己**身上（本 goal 前两个阶段共三次栽在「绿是别的原因造成的」）。
- 日报渲染改动**不需要**桌面验收：goal Constraints 点名的是复审页与 Dashboard，本阶段不碰它们。若实现中发现必须改 Dashboard 的 provenance 展示，那条约束立刻生效，需要用户在真实 Obsidian 里看过。

## Abort / reshape triggers

- 若真模型无法稳定回报方向标识、必须放宽严格解码才能跑通 → 停下。先在标识形态上做 L1；若「方向级归因交给分类器」这条路本身不成立，就是 L2，改用别的归因方式（例如筛选仍按主题、方向归因另出一步）。
- 若标记升级牵动 Dashboard、`history-sync`、`paper-index` 的改动超出「解析兼容」这一层 → 停下拆阶段，不要在 P3 里顺手改派生索引。
- 若发现回滚在**数据侧**真的被破坏（旧构建读同一个 `data.json` 会坏），而不只是提示词变了 → L3，因为那动摇的是 goal 的第一条验收标准。
- 若为了让方向参与判定，不得不同时改动详情选取或主题名的角色 → 停下确认，这两项都被显式排除在本阶段之外。

## Open questions（P3 不顺手定掉）

- 详情选取（`detail-selector.ts:155`）是否也该看到全部方向而不是影子 `description`。
- Dashboard 是否展示主题方向的命中来源（本阶段只保证机器标记里有，不保证界面上有）。
- 画像文档驱动的第二分类器与 `personal-library` 类别的退休时机（P4/P5）。

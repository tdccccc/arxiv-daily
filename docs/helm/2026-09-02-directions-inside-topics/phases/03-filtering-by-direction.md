# P3 — filtering-by-direction

goal_ref: ../goal.md
created: 2026-09-03T23:21:47+08:00
updated: 2026-09-04T12:55:00+08:00
revision: 4

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
- 方向标识定为 **`<tag>#<n>`**（n 是方向在其主题内的序号）。方向的存储 `id` 是 UUID，让模型逐字回抄又长又易错；`<tag>#<n>` 短，且**自带所属主题**，所以「论文归在 A 主题却报了 B 主题的方向」是可检出的违约而不是静默接受。按最后一个 `#` 切分，tag 自身含 `#` 也不歧义（`nlp|llm` 那条既有测试同时覆盖了含 `|` 的 tag）。
- `identity` 一并带上 `directions: [{ref, tag, id, text}]` 而不是只带可用标识：settings 可能在 LLM 调用期间被改（既有测试「persists the immutable exact request snapshot when settings mutate during the LLM call」钉着这件事），所以 Chunk 5 要用的方向文本必须来自**冻结的请求**，不能回头读实时设置。一次成形，避免 Chunk 4 再改一次契约、再抬一次版本号。
- **L1：与 Chunk 2、Chunk 3 合并为一次提交。** 三者是同一次契约变更的三面——改提示词必然推翻 Chunk 3 那条「提示词逐字不变」的断言，分开提交会留下一次自知的红。各自的变异检验仍分开做。
- exception: 无
- [x] implementation and tests accepted

### Chunk 2 — 两个契约版本号显式抬，旧缓存作废是被断言的

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `DAILY_FILTER_PROMPT_CONTRACT_VERSION` 与 `DAILY_FILTER_RESULT_CONTRACT_VERSION` 各 1→2。新增测试——按旧版本号存下的 checkpoint **不得**被复用。红在版本号未抬时旧 checkpoint 照样命中。
- Green check: `npm run test --workspace @arxiv-daily/core -- daily-filter-checkpoint-store`
- regression checks: core 全量
- **变异检验当场抓到一条自己写的假绿，值得记。** 第一版把陈旧版本号写成 `CURRENT - 1`，两条测试各自独立、看起来满足「每个版本号各一条」。但退回任一版本号时，`CURRENT - 1` 跟着一起降，陈旧值与当前值**永远不相等**，测试照样绿——它实际只钉住「版本号不等于版本号减一」这句废话。改成旧契约真正发布过的**字面量 1**后，退回 prompt 版本只有 prompt 那条红、退回 result 版本只有 result 那条红。
- **与 P2 那次的同形之处**：两次都是「测试通过了，但通过的原因不是被测行为」。P2 是两个断言挤在一条测试里，这次是断言的参照系跟着被测对象一起动。判据要钉在**不随实现变动的锚点**上。
- exception: 无
- [x] implementation and tests accepted

### Chunk 3 — 回滚契约重新判定：守数据，不守提示词

- change kind: behavior change（测试契约重新表达）
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `settings-rollback.test.ts:68`「迁移后筛选提示词逐字不变」在 Chunk 1 之后必然红——**这是预期内的红，且它证明 Chunk 1 真的改到了筛选**。重写为两条：(a) 旧构建只读那四个字段时，`validateFilterConfig` 仍过、`migrateArxivSettings` 往返仍稳定（数据侧可回滚，逐字不变）；(b) 新旧提示词**必须不同**，且差异由抬起的契约版本号显式宣告。其余四条断言原样保留。
- Green check: `npm run test --workspace @arxiv-daily/core -- settings-rollback`
- regression checks: `real-corpus-migration.test.ts`、`migration.test.ts`、`validation.test.ts` 全绿
- **不允许的做法**：把那条断言删掉了事，或放宽成「大致相同」。P1 的验收标准是「迁移可回滚」，本 chunk 要让它继续以可执行形式存在，只是换到它真正该待的那一侧。
- 落地形态：原「提示词逐字不变」的两条换成——(a)「迁移后按方向出题」，断言提示词里是方向行、且 `identity.directions` 逐条对得上迁移产物；(b)「提示词变了这件事由两个版本号显式宣告」，断言老的一行式提示词已不存在、两个版本号都大于 1。数据侧回滚的三条（旧构建配置检查通过、影子等于第一条方向、往返稳定）**原样保留未动**。
- **顺带定死了一件事**：`buildPaperFilterRequest` 不再接受未经 `normalizeTopic` 的设置。原测试直接把老形状喂给它来比对提示词，现在会抛。这是 P1「`normalizeTopic` 是所有主题的唯一入口」的自然结果，没有加 `?? []` 去兜——兜住只会造出「有主题但一条方向都没有」的静默状态，而那正是 Chunk 6 要正面定义的退化行为。
- exception: 无
- [x] implementation and tests accepted

### Chunk 4 — 命中方向沿管线流到日报组装

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `FilteredPaper` 带上命中的主题方向（tag + 方向标识 + 方向文本），经 `summarizer.ts:107` 与 `daily-summary-assembler.ts` 透传到组装层，`preflightDailySummaryAssembly` 对其做与既有 provenance 同级的校验（不合法即抛，不静默丢）。红在组装层收不到这个字段。
- Green check: `npm run test --workspace @arxiv-daily/core -- daily-summary-assembler`
- regression checks: core 全量；`npm run check:boundaries`
- **新字段与 `discoveryProvenance` 并列而不是塞进去**：后者的 `directions[].representatives` 至少一条是硬性约束，而主题方向按 ADR 0012 §4 不留证据，塞进去只能靠放宽那条约束，等于让两种语义共用一个校验器。落地为 `topic-direction-hits.ts` 的 `TopicDirectionHit{tag,id,text}` 与 `normalizeTopicDirectionHits`。
- **命中方向按主题自己的顺序排，不按模型回答的顺序**：用户在设置页看到的就是这个顺序，日报里换一个顺序没有道理。实现上是过滤冻结请求里的方向表，而不是遍历模型给的数组——变异检验把后者写回来时，顺序那条断言立刻红。
- 校验多钉了一条计划里没写的不变量：**一篇论文的所有命中方向必须同属一个主题**。这是「归组由单一 tag 决定」的直接推论，混着两个 tag 说明命中是被错误拼装的。
- exception: 无
- [x] implementation and tests accepted

### Chunk 5 — 日报写出命中的方向：可见一行 + 机器标记

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 渲染带方向命中的论文块时，(a) 可见行说出主题与命中的每一条方向文本（中英双语，走既有 `escapeDiscoveryProvenancePlainText` 的转义，方向文本是用户输入，不可信）；(b) 机器标记可被解析回等价结构；(c) **既有报告里的 v1 标记仍能被解析**。红在渲染不产出方向、或旧标记解析回归。
- Green check: `npm run test --workspace @arxiv-daily/core -- daily-summary-rendering`；`-- discovery-provenance-marker`
- regression checks: `daily-summary-parser` 与 `paper-index` 相关测试全绿；core 全量
- **标记形态定为另起前缀 `arxiv-daily-topic-directions`，而不是把 discovery-provenance 扩到 v2。** 理由有两条，第二条是读代码才发现的：(a) 手动路径的论文在没有文献库时根本没有 `discoveryProvenance`，扩它等于为了搭车而凭空造一个；(b) 既有的两个标记族**各自把自己钉死在从标题数起的固定行号上**（`personal-novelty-marker.ts:139` 依赖 discovery 标记是否存在来算第 3 行），中间插一族会把这套算术全部打乱。新标记族**排在最后**，两个既有解析器的行号算术一行不用改。
- **`renderVisibleTopicDirections` 是本阶段真正交付验收标准的那一行**：`> 命中方向：主题 <tag> — <方向文本>、<方向文本>`。方向文本是用户输入，走既有转义。
- **一处自查漏网**：变异检验发现「标记必须在规范槽位」这条规则**删掉后没有任何测试变红**——原本以为「标记跑到 block 外面」那条覆盖了它，其实那条是被另一个守卫抓住的。补了一条「标记在 block 内但不在规范槽位」的测试后才真红。这条规则不是装饰：它和另两族一样，是防止不可信的摘要正文伪造出一行看起来合法的标记。
- **本 chunk 的实现写在测试之前**（marker 模块整体成形后才补测试），不满足 strict Red-Green。补偿验证是逐条变异检验：可见行、转义、槽位规则、身份校验各删一次，对应断言都红。
- exception: 见上，已记补偿验证
- [x] implementation and tests accepted

### Chunk 6 — 退化行为：没有方向的主题、模型一条都不命中

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: (a) 一个主题的方向被删空时（影子 `description` 为 `""`），筛选与配置检查的行为被钉死——今天 `validation.ts:163` 报 “description is empty”，在新模型下这句话该说的是「这个主题一条方向也没有」；(b) 模型给出 `category` 却回报空方向列表时的行为被显式钉死（保留该论文并只标主题，还是判违约），不留给巧合。
- Green check: `npm run test --workspace @arxiv-daily/core -- validation`；`-- paper-filter`
- regression checks: core 全量；plugin 全量
- **两处行为都定死了**：(a) 配置检查的文案从「description is empty」改成「has no directions」——描述是影子字段，设置页根本不显示它，指着它让用户去修是指错了地方；(b) **没有方向的主题整个不进提示词、也不进合法 tag**，一条方向都没有时直接不调 LLM。理由：留在提示词里等于邀请模型把论文归到它下面、然后因为报不出方向而触发严格解码违约——**一次配置问题葬送整次运行**。两种情况都 warn 出被跳过的主题名，不静默。
- `validateFilterConfig` 这里**刻意做了防御性取值**（`topic.directions ?? []`）：它的职责就是报告坏配置，遇到没规范化过的主题应当报出来而不是抛。这与 Chunk 3 里 `buildPaperFilterRequest` 不做防御是两回事——那里防御会造出「有主题但没方向」的静默状态，正是本 chunk 要消灭的。
- **牵出一条既有测试的前提错误**：`settings-rollback` 里「降级后配置检查仍通过」原本把**未经迁移**的四字段形状直接喂给校验器。生产上任何构建读盘都先迁移，未迁移形状根本不会被校验。改为走一次「降级—再升级」的真实往返，断言仍是「不该有方向/描述相关的抱怨且 ok」。
- exception: 无
- [x] implementation and tests accepted

### Chunk 7 — CLI 侧行为一致

- change kind: behavior change（跟随）
- strategy: strict Red-Green-Refactor
- **计划里这条预期是错的，如实记下**：原写「红在 CLI 侧仍只喂 description」。实际 CLI 早在 Chunk 1 就跟着变了——它走的就是同一个 `buildPaperFilterRequest`，没有独立的筛选代码可以掉队。**所以这里取不到诚实的红**，两条测试是确认跟随成立的表征测试，不是 Red-Green。
- Green check: `npm run test --workspace arxiv-daily`（73/73）
- regression checks: 四包全量；`npm run typecheck`；`npm run lint`
- 钉住的两件事：legacy 的 `description = "..."` TOML 主题迁移成一条可判定的方向；TOML 里显式写的 `directions` 列表按顺序进入提示词。goal Constraints 要求插件与 CLI 两侧行为一致；ADR 0003 的两产品边界不因此变动。
- exception: 无法取红，见上
- [x] implementation and tests accepted

## Phase verification

- 四包全量测试、`npm run typecheck`、`npm run lint`（基线 0 error / 20 warning）、`npm run check:boundaries` 全过。
- **真实端点小批量实测：已做，通过。** 用户配置里的真实端点（`gpt-5.6-sol`，temperature 0）+ 9 篇真实 arXiv 论文（3 篇 photo-z、3 篇星系团、3 篇 hep-th 无关项），跑两轮：
  - 两轮**都**返回严格合法的 `{id, category, directions}`，`<tag>#<n>` 标识全部属于所选主题，`decodePaperFilterRecords` 直接判 ok，无需放宽任何一条解码规则。**标识形态在真模型上站得住。**
  - 无关的 3 篇 hep-th 两轮都判 skip，没有一次乱塞。
  - **一条如实记下的观察：18 次判定里没有出现过一次「一篇命中多条方向」。** 提示词明写了「可以同时命中多条、全部列出」，模型仍然只挑最匹配的一条——包括那篇既是 SZ 选源星表、又做质量标定的 SPT 论文。契约允许 1..n 且单测覆盖了多条的路径，**但多条命中在真模型上没有活证据**。这是召回口径问题，取决于方向文本怎么写，属于 P6 拿真实日报判断的事，不是本阶段的契约缺陷。
  - 另一条观察：模型偏保守，`redMaPPer 子结构`那篇（optical 星表相关）两轮都判 skip。同样是召回口径，留给 P6。
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

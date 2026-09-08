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


## 2026-09-02 — P1 Chunk 3：设置页改成方向列表（未交付，待用户过目）

- evidence: **第一次取红是假的**。测试原本用 `tab.refreshSettings()` 渲染，但它在 1.13+ 走声明式路径，而该路径的 `update()` 被同文件其它测试打了桩——实测渲染出 **0 张主题卡片**。于是「不再有 description textarea」那条断言**因为什么都没渲染而绿了**。加探针打印卡片数为 0 才发现。改用公开的 `renderTopicRow(new Setting(...), 0)` 真渲染后，红变成 `expected <textarea> to be null`，5 条全红且红在被测行为上。
- **这是本轮最该记住的一条**：与之前那次宽泛 `toThrow()` 同类——红对了不等于红在对的地方。一条断言若在功能缺失时也能绿，它守不住任何东西。
- change: 主题卡片里的单个 description textarea 换成方向列表：每条一行 `input`、各带删除按钮，底部一个「Add direction」。输入即改 `directions[i].text`，`description` 由 core 的 `deriveTopicDescription` 重新派生——设置页调用该函数而不是自己写一遍派生规则，否则界面就成了影子的第二个作者。
- disposition: 撞坏两条既有测试，都是契约变更的直接后果，改的都不是源码：一条夹具的 topic 缺 `directions`（改走 `normalizeTopic`）；一条源码文本断言指着 `descId`（改为 `dirId`），同时把「输入时不被重渲染夺走焦点」这条保护从 description 挪到方向输入框上——那条保护本身比它守的具体控件更重要。
- **一处自行做的判断**：编辑中允许方向为空行，不在每次按键时清理。理由是按键即清理会把光标所在的行删掉；空的第一条方向会让 `description` 为空，配置检查照常报 empty——这是**正确的反馈**，不是缺陷。空行在下次读入时由 `normalizeTopic` 清掉。
- 样式：`styles.css` 里 `topic-description` 的三处引用改为方向列表，并补上行内布局与增删按钮样式。未动那条 `@media (max-width: 600px)`——它是设置页既有写法，不在本 chunk 的范围内。
- validation: plugin 710/710、core 全分片、CLI 71/71、typecheck 四包、lint 0 error（20 warning 既有）、check:boundaries OK。
- **本 chunk 没有证明的事，也是它不能算完成的原因**：界面只在 happy-dom 里渲染过。一行一条读起来顺不顺、删除按钮认不认得出、加方向的位置顺不顺手，测试一概证明不了。上一轮的返工正是「测试绿 + 我认为够用」，所以这条**不自己勾**。
- next: 把产物装进 plugin_test 由用户打开确认；确认后再进 Chunk 4（模板与新建主题产出方向）。

## 2026-09-02 — 用户打开后：焦点仍被夺走。查出是 0.4.4 修复的漏网情形

- evidence: 用户在 plugin_test 打开设置页，报「打一个就把焦点移出去了，没法连着打字」，并称「还是老毛病」。**先写探针复现，没复现出来**——那本身是线索：探针里 `update()` 是空操作，而 obsidian 桩的 `requireApiVersion` 返回 true，所以声明式重渲染确实发生了、只是在测试里没有可见后果。
- **根因**：`definitions.ts:187` 只有 `showSetupGuide` 为真时才把「Getting started」行放进定义。设置一旦完成，该行不存在，`declarativeSetupGuideRow` 便从未被赋值；`refreshDeclarativeSetupGuide()` 于是落到兜底的 `this.refreshSettings()` → `update()` → **整页重渲染**，正在输入的控件被替换掉。
- **这不是 directions 引入的，是 0.4.4（`6b96b57`）修复的漏网情形**。那次修复只在引导卡还在屏幕上时走便宜路径，而它的回归测试恰好先手动渲染了引导行——于是只覆盖了引导期。name、tag 与旧的 description 在设置完成后同样会丢焦点，用户说「老毛病」是准确的。
- change: 没有引导行可更新时，只有「引导需要重新出现」才整页重渲染；否则什么都不做。
- **测试的教训**：既有那条焦点测试断言的是 `refreshSettings` 未被调用——判据是对的，**但它把自己放进了引导卡仍在的情形**，于是永远走不到出问题的分支。新测试显式构造「设置已完成」（LLM 三项齐备 + 运行状态里有 completed 记录），并覆盖 name / tag / direction 三个字段。红的原文是 `expected "refreshSettings" to not be called at all, but actually been called 1 times`。
- validation: plugin 712/712。产物已重新构建并装入 plugin_test（main.js md5 `9bde2d7c`）。
- **仍未交付**：Chunk 3 的验收框继续不勾，等用户重开设置页确认焦点与其余观感。
- next: 用户复看；通过后提交 Chunk 3 与本修复，再进 Chunk 4。

## 2026-09-02 — 方向改为可折行：收起两行、聚焦展开

- evidence: 用户问长方向能否折行显示而不是挤在一行。核实后确认**不违反 ADR 0012 §2**——那条约束的是数据模型（一个字符串，不是「名称+描述+线索」的结构化记录），不是渲染成几行。它给出的理由「几十条也一眼扫得完」才是真代价，用户选了「收起最多两行、聚焦展开全文」来平衡。
- change: 方向控件由 `input` 改为 `textarea`。收起时最多两行、超出显示省略号；聚焦展开全文，失焦收回。短方向两种状态完全一致，不跳动。
- **一个硬约束逼出的交互**：筛选 prompt 是 `- tag: description` 按 `\n` 拼的，文本里混进换行会切断那结构。所以回车不插入换行，改为**在下方新建一条方向并聚焦**；粘贴进来的换行一律折成空格。
- **两处实现上的坑，都不是第一直觉**：
  - `textarea` **做不出省略号**（`line-clamp` 不作用于其内部文本）。省略号改为同级的覆盖标记，`pointer-events: none`，仅在收起且确实溢出时出现。
  - 最初用 JS 测 `scrollHeight` 定高，撞上 obsidian 的 lint 规则 `no-static-styles-assignment`（禁止写 `element.style.height`）。改用 CSS grid 的「文本复制到 ::after」自增高技巧后**一行内联样式都不需要**——顺带消掉了原本的隐患：卡片折叠时 `scrollHeight` 为 0，按它定高会让展开后的文本框是 0 高。高度全交给 CSS 之后，这个问题不存在了。
- disposition: 收起状态的类挂在自增高的外层容器上而非 `textarea` 上（高度上限属于容器）。测试相应改查容器。
- validation: plugin 715/715、core 全分片、CLI 71/71、typecheck 四包、lint 0 error（20 warning 既有）、check:boundaries OK。产物已装入 plugin_test（main.js md5 `a6fa448d`）。
- **仍未交付**：Chunk 3 验收框继续不勾，等用户看过折行、省略号、回车新建这三件事的真实观感。
- next: 用户复看。

## 2026-09-02 — 截断提示改为行尾「+N」徽标，收起改回一行

- evidence: 用户看过两行折行版后说省略号「不太起眼，不太好看」。**症结不只是审美**：省略号贴在右端，而截断实际发生在下方（第三行被压住），用横向记号表示纵向截断，指错了方向，看不见是必然的。用户随后定下形态：收起一行，行尾给计数标记。
- change: 收起上限由两行改为一行；`…` 换成行尾的 `+N` 小胶囊（带底色，`pointer-events: none`，点它照样落进文本框展开），悬停标题写全 `2 more lines` / `1 more line`。收起时文本框右内边距让出徽标位置，末尾的字不会钻到徽标底下。
- **单位选「行」不选「词」，是被可测量性决定的，不是偏好**：更想显示 `+6 words`（描述内容而非版式），但要算被藏起来的词数得测量文本在哪个字符被切断（Range 或 canvas 测宽 + 二分），为一个提示标记造这套机器不值当；而「还差几行」用现成的 `scrollHeight / clientHeight / lineHeight` 就能算准。**只显示能算准的量。**
- **不写「行」字**：插件界面通体英文（Name / Tag / Directions / Add direction / Detail report），中间插一个中文单位会突兀；徽标只写 `+2`，把完整措辞放进悬停标题，因为行尾空间紧张，而显眼靠的是样式不是字数。
- **新测试做了变异检验**：它一次就绿，按本轮教训不能就此采信。把 `+${hidden}` 改成 `+${hidden + 1}` 后确认它变红（`expected '+3' to be '+2'`），再还原。happy-dom 没有布局，几何（lineHeight 20px、clientHeight 20、scrollHeight 60）在测试里显式声明。
- validation: plugin 716/716、typecheck 四包、lint 0 error、check:boundaries OK。产物已装入 plugin_test（main.js md5 `29ec7ce8`）。
- **仍未交付**：Chunk 3 验收框继续不勾。
- next: 用户复看徽标的显眼程度与整体观感。

## 2026-09-02 — P1 Chunk 4+5 done：P1 完成

- evidence: Chunk 4 取红在 `topicFromSeed` 不存在；Chunk 5 五条断言**一次全绿**，因此按本轮规矩做了变异检验——把 `deriveTopicDescription` 改成返回空串，五条全红（prompt 不等、`expected [ Array(2) ] to deeply equal []`、影子为空、往返取不到 text），再还原。
- change（Chunk 4）: 模板种子由 `{ description }` 改为 `{ directions: string[] }`，套用经 `topicFromSeed`，方向 `origin` 为 `manual`。**改这个是因为原来会标成 `migrated`，而用户挑模板时根本没有任何东西被迁移——那是句假话。**
- **一处被设计前提逼定的克制**：模板每个主题仍只给一条方向。影子只等于第一条方向，而 P1 的筛选仍读影子；若把逗号分隔的模板描述拆成多条，第二条起不进分类器，等于**悄悄削窄新用户的筛选面**。拆分留到 P3，并已写成测试钉死，防止日后有人「顺手优化」。
- change（Chunk 5）: `packages/core/tests/settings-rollback.test.ts` 把「可回滚」变成可执行判据五条，其中最有价值的是**往返**那条：降级 → 旧版只按四个字段回写 → 再升级，影子与第一条方向都稳定。
- disposition: 撞坏一条既有测试（`topic-templates.test.ts` 断言种子上的 `description`），改断言为方向行非空——种子现在承载方向，`description` 是派生值。
- **桌面验收的判断在阶段文件里明写了，不静默跳过**：判定「应当补，但不阻塞 P1」。用户已在真实 Obsidian 逐轮看过（Chunk 3 的交付条件已满足），但方向列表有依赖几何的行为（截断判定、`+N` 计数、折行不横向溢出），正是单测证明不了、上一轮翻车的那一类。建议单独补一个桌面场景，优先级交用户定。
- validation: core 2050 项全绿、plugin 718/718、CLI 71/71、typecheck 四包、lint 0 error（20 warning 既有）、check:boundaries OK。
- boundary: 管线仍未动，`paper-filter-contract.ts` 继续读 `description`——这是 P1 的设计前提，也由 Chunk 5 逐字钉死。ADR 0005/0007/0008 未动，授权面未变。三个并行 active helm 未碰。
- next: P2（索引只覆盖标题与摘要，重建降到分钟级）。P1 的六条 open questions 保持开放，尤其是 personal novelty 基准与「主题名是否作为硬门」。

## 2026-09-02 — Enter 由「新增方向」改为「确认并失焦」

- evidence: 用户试用后指出 Enter 直接新增方向没有必要——「点击添加足矣」。同时注意到 Enter 与 Shift+Enter 都换不了行。
- **换行这条不能顺势放开，理由是数据契约不是偏好**：方向存的是一行文字，筛选 prompt 把每个主题拼成 `- tag: description` 再按 `\n` 连接，文本里混进换行会切断那个列表结构（ADR 0012 §2 的数据模型亦然）。所以 Enter 只能被吞掉。
- change: Enter 先改为完全静默，用户随即定为**确认**——吞掉按键、令输入框失焦，方向框收回一行。Shift+Enter 同样处理：它一样插不了换行，行为不一致反而更迷惑。新增方向仍只由「Add direction」按钮承担。
- disposition: 顺手收掉一处多余的通用性——「按位置插入方向」只剩追加到末尾一个调用点，简化为 `appendDirection`。点 Add 后光标仍落进新方向框。
- **变异检验**：改完一次全绿，遂把 `blur()` 注释掉确认那条测试变红（焦点仍在输入框）。该测试断言的是「Enter 后焦点离开且方向框收回」这一行为，不是「代码里存在 blur 调用」。
- validation: plugin 718/718、typecheck 四包、lint 0 error、产物已装入 plugin_test（main.js md5 `7b358e46`），用户确认。
- next: P2（索引只覆盖标题与摘要）。另有一项待用户定优先级：给方向列表补一个桌面验收场景，量截断判定、`+N` 计数与折行不横向溢出。

## 2026-09-02 — 补上方向列表的桌面验收

- evidence: 七条断言全过，其中最关键的是**徽标承诺的 `+N` 等于展开后真正多出的行数**（实测 `+2 matches the 2.00 lines gained on opening`）。其余：收起恰为 1.00 行；能装下的方向无徽标；徽标在框内；宽 689px 与窄 447px 面板下都折行不溢出；无 console 错误。
- **这条断言刻意不复算插件的公式**。若照 `scrollHeight / clientHeight / lineHeight` 再算一遍，就只是重述插件自己的算术——布局错了也照样通过。改为「展开字段、量真实增高」，是对同一主张的独立测量。
- **发现框架本身的一个坑，值得单独记**：验收部署的是**已构建的 `main.js`，不是源码**。我第一次做变异检验时改了源码却没重新构建，于是它拿旧产物跑出了一个 PASS——正是 build-deploy README 里说要防的「静默地报告一个分支上任何构建都不产生的布局」。现有守卫比对 manifest 版本号，而两次构建版本相同，拦不住这种陈旧。重新构建后变异才真正变红。**建议让 `test:desktop` 先构建**（一行脚本改动），但那属于桌面验收 helm，未擅自改。
- disposition: **关于「不修改并行 active helm」这条约束的判断**：新增了 `topic-directions.mjs` 并在 `acceptance.mjs` 接了两行。判为可以——这是**使用**框架而非改动框架，且 `test/obsidian-desktop-harness` 已是本分支祖先（91 ahead / 0 behind），不存在未合并的并行改动会冲突。
- **安全**：夹具会覆盖 `data.json`，而测试库是用户的真实设置。核实了框架在装夹具**之前**捕获状态并在结束时还原，另外自行备份一份；跑完比对确认逐字节还原。
- validation: `OBSIDIAN_TEST_VAULT=/home/tiandc/Desktop/plugin_test npm run test:desktop` 整体 PASSED（含既有的库设置页 18 条）；lint 0 error；release-tools 317/317。截图四张写入 `.acceptance-out/`。
- next: P2（索引只覆盖标题与摘要，重建降到分钟级）。

## 2026-09-02 — P2 阶段计划：索引只覆盖标题与摘要

- evidence: 读代码定住了三件之前只在 ADR 层面说过的事。(1) **摘要已经在手边**——`PersonalLibraryPaperRecord` 带 `title` + `abstract`，`evidenceDepth` 就是 `"metadata-and-abstract"`，而 `catalog` 本来就是 `indexPersonalLibraryFullText` 的入参；所以 P2 是改索引的**输入来源**，不是新建元数据链路。(2) **`textHash` 不参与复用判定**（`index-orchestration.ts:213–218` 的 `exactReady` 只看 `observationFingerprints` + `modelId` + `derivation`，而 `identificationFingerprint` 只覆盖扩展名与识别策略版本）——改吃 catalog 文本之后这会变成一个真实的陈旧洞：摘要变了而文件没动，索引静默留旧向量。(3) 复用判定**已经**看 `derivation`，所以抬 `CHUNK_DERIVATION_VERSIONS` 就是 ADR 0013 §3「重建而非迁移」的现成执行点。
- **ADR 0013 只数了 embedding 调用，但重建耗时的大头可能是 PDF 解析**。识别出的论文改从 catalog 取文本后整个不必 `readBinary` + `parseIndexDocument`。这是最大的机会，也正因为它没被 ADR 量过，Chunk 1 把「先量基线」写成了**门而不是仪式**：fallback 占比与「解析 vs embedding」的耗时拆分若不利，abort trigger 当场触发。
- change: 写 `phases/02-index-titles-and-abstracts.md`，五个 chunk（量基线 → catalog 论文只索引标题摘要 → fallback 的有界首页路径 → 强制重建并堵陈旧洞 → 复量并确认聚类没塌）。goal.md 的 P2 行由 `pending` 改 `active`，revision 3 → 4。
- **两处刻意的克制，都写进了 abort trigger**：其一，聚类每篇只剩一到两块之后 `maxChunkCosine` 退化为摘要对摘要，方向变粗是 ADR 0013 认下的代价、且是「要测的量」——**若聚类塌成一个大簇或每篇一簇，那是对 ADR 0013 §1 的反证，按 L2/L3 分类，不许在 P2 里调阈值掩过去**（goal 的 Constraints 明写两个阈值不得拍脑袋定，且实测排在 P4/P5）。其二，`chunkFullText` / `chunkParsedDocument` 保留不删——ADR 0013 §2 把「纳入结论段」列为最可能的下一增量，删分块器等于给回头路凭空加成本。
- **Chunk 2 的取红判据选了「`readBinary` 一次也没被调用」而不是「块数变少」**：块数少可以靠截断伪造，而不读文件才是省下解析时间的那个机制。同理 Chunk 4 把「混合索引」点名为最坏失败模式——一部分论文 113 块、一部分 2 块，`maxChunkCosine` 会系统性偏向全文那部分，**而且不报错**。
- disposition: 桌面验收判为不需要（索引是后台过程，无渲染几何可量），但该判定挂了条件——若改动触及设置页的索引进度显示即作废。上一轮那条教训一并写进了 trigger：**任何桌面变异检验必须先重新构建**。
- validation: 尚未动任何源码，无测试可跑。P1 的全量绿仍是当前基线。
- next: Chunk 1——在测试库上重建一次索引，量总块数 / 体积 / 墙钟耗时 / embedding 调用数，拆出解析与 embedding 各占多少，并数出 catalog 识别数 vs fallback 数。ADR 0013 的 207 篇 → 23,423 块 → 140MB 是对照。

## 2026-09-02 — P2 Chunk 1 done：基线与识别天花板，以及一次 L1 调整

- evidence（全部来自冻结语料上的真实代码运行，不是估算）：
  - **ADR 0013 的基线从磁盘完整复原**，无需重跑全文侧。那份 140MB 索引还在：**207 篇 / 23,423 块 / 768 维 / `remote:nomic-embed-text`**，均值 113.2 块/篇、中位数 79、**最大 1002**，ready 204 / failed 3。与 ADR 记的数字逐项吻合。
  - **识别天花板**：arXiv id 命中 **3 篇（1.4%）**；从 `/Title` 里再抽 id **0 篇**；有 DOI 无 arXiv id **96 篇（45.3%）**；**什么元数据都没有 113 篇（53.3%）**。
  - **那 207 篇当年 100% 落在 fallback 键上，且用的就是今天这版识别**——两代索引的 identification fingerprint 同为 `73049d71…`。所以「换新识别就能好」这个念头当场被排除。
  - **轻量提取吞吐**：212 篇 26 秒，约 123ms/篇 → 千篇约 2 分钟。这是「分钟级」唯一的可行性依据。
- **冻结语料**：`~/Desktop/plugin_test/baseline_library_207`，212 个文件。由 manifest 的 `filePaths` 复原（源根是 `test_library/`，今天已长到 376 篇，所以必须冻结那 212 个才可比）。用**硬链接**而非 `cp`：磁盘 91% 已用，硬链接零增长（实测 68G 空闲不变，link count 2），且原名被删也不丢内容。Chunk 5 用同一份做前后对照。
- **一次决定的反转，值得记全**。按天花板数字，用户先选了「先补识别率再瘦身」（L2，插入新阶段）；我把 arXiv / DOI / 无元数据三条路的占比摆出来后，用户改为**「以 PDF 内容为准，不要纠结 arXiv 号或者文件名」**。这条判断把方案**变简单了**，不是变复杂：
  - 双路径（catalog 优先 / fallback 兜底）压成**一条路径**；
  - revision 1 里那个「摘要变了而文件没变 → 复用旧向量」的陈旧洞**直接不存在**了——文本既然派生自文件内容，现有 `observationFingerprints` 就已经是正确的变更信号，`textHash` 不必进复用判定。
- **分类改判：这是 L1 不是 L2。** P2 的 outcome（每篇一到两块、MB 级、分钟级重建）一字未改，goal 的成功标准也未动；被证伪的只是 approach 里「摘要从 catalog 来」这一条，而替代路径达成同一个 outcome。因此原地改 phase 02（revision 1 → 2），不新增阶段、不动 goal。先前跟用户说的 L2 是针对「先补识别率」那个选项的，该选项已被放弃。
- **顺带发现一个真缺陷，按用户方向不修**：`pdf-text-utils.ts` 的 `LEGACY_ARXIV_ID_IN_TEXT_RE` 少了 `(?:v\d+)?`，而新式正则有——`astro-ph/0003380v2` 这类带版本号的旧式 id 因此提取不出（尾部断言 `(?=$|[^0-9A-Za-z])` 撞上 `v`）。这是新旧两条正则之间的不对称，不是设计取舍。用户判「以 PDF 内容为准」，故本阶段不动，记入 open questions。
- **一个测量方法上的教训**：仓库里没有 node 侧 PDF 解析器（解析在 Electron 的 pdfjs 或 Docling sidecar 里），也没有 tsx，所以 TS 侧的一次性测量走的是「临时 vitest 测试 + 跑完即删」。第一次跑撞上 vitest 默认 5s 超时判红，但 `console.log` 已经打完——**输出有了不等于测试通过**，差点把一次超时当成功。加 `timeout` 重跑后才是诚实的绿。临时文件已删，工作区干净。
- disposition: `chunkFullText` / `chunkParsedDocument` 保留不删（ADR 0013 §2 的结论段增量要用）。DOI/Crossref 拿真摘要这条路实测可达 45.3%，记入 open questions，不在本阶段做。
- validation: 未改任何产品源码，无回归可跑。P1 的全量绿仍是当前基线。
- next: Chunk 2——写「从首页文本定位标题与摘要」的纯函数，兜底为首页前 N 字符且**必须有上界**，用冻结语料里的真实首页（含双栏、含无 Abstract 标题的期刊版式）做夹具。

## 2026-09-02 — P2 Chunk 2 done：从首页提取摘要，84.4% 走真摘要路径

- evidence: 先看真实版式再写代码，四种版式各不相同，任何一种单独拿去推广都会错——A&A 的 `ABSTRACT` 独占一行；Kluwer 的 `Abstract.` 行内起头、正文紧随；**REVTeX 根本不印 "Abstract" 这个词**，正文直接跟在 `(Dated: …)` 后；ApJ 一篇 50+ 作者的论文把摘要挤到了第 2 页。另有扫描件首页只剩一行 bibcode 水印。
- change: `abstract-extraction.ts`，三条路径**逐页**尝试（marker → dated → 有界兜底），逐页是为了让摘要不跨页流进正文。无文本层判 `none` 而**不兜底**：把 bibcode 水印当摘要嵌进去比没有向量更糟。所有路径都受 `MAX_ABSTRACT_CHARS` 约束——这条上界正是「每篇一到两块」赖以成立的东西，少了它一种怪版式就能把全文体积放回来。
- **真实语料实测（212 篇）**：marker 177（83.5%）、dated 2（0.9%）、leading-text 26（12.3%）、none 7（3.3%）。**84.4% 命中真摘要**。摘要长度中位数 1594 字符、p90 2397；**总索引字符 337,786**，对比改造前 23,423 块约 4800 万字符——降到约 0.7%。abort trigger 里「兜底成多数」这条**不触发**。
- **一处实现语义写错了，测试逮到**：最初把「首页文本太短即无文本层」的门槛放在所有路径之前，于是一个明明有 `ABSTRACT` 标记、正文也在的短页被判成 `none`。**找到摘要标记本身就是文本层存在的证据**，长度门槛只该管兜底路径。改的是实现，不是测试。
- **变异检验暴露了一个测试盲点，比变异本身更值钱**：禁掉终止规则后三条主路径变红，但「摘要在第 2 页」那条**照样绿**——它断言的 `not.toMatch(/Bocquet/)` 查的是上一页的内容，根本管不住终止有没有生效。补了 `not.toMatch(/Introduction/)`。P1 记过「红必须红在被测行为上」，这是它的另一面：**绿也必须绿在被测行为上**，而只有变异检验能把这种假绿翻出来。
- **一处刻意不做的调优**：兜底的 26 篇里，`Beck2022`、`DES Collaboration2022` 取到的是期刊卷页页眉而不是标题，加几条正则当场就能「修好」。没有修——我手上只有 8 个样本，按它们调正则就是拿样本当判据，与 goal 的 Constraints 里「两个阈值不得拍脑袋定」是同一类错误；兜底文本的好坏我也确实没有可测判据，只有观感。记入 open questions，等有判据再说。
- **测量方法上的一点交代**：真实语料的路径分布是用 `pdftotext` 抽前两页文本喂进函数量的，而产品里解析走 pdfjs / Docling。**文本会有出入，所以这些百分比是趋势而非承诺**；Chunk 3 接进索引后要用真解析器复核一次。临时测量文件跑完即删，工作区干净。
- validation: 新增 10 条测试全绿；core 全量 **2060 项**（P1 时 2050，新增 10）、typecheck 四包、lint 0 error（20 warning 既有基线）、check:boundaries OK。
- boundary: 新模块尚未被任何地方引用，索引仍走全文——接线是 Chunk 3。筛选管线一行未动。
- next: Chunk 3——把索引的文本来源换成它，并把解析范围收到前两页。取红判据要落在**解析范围**上而不是块数上：块数少可以靠截断伪造，「不解析全文」才是「分钟级」赖以成立的那件事。

## 2026-09-02 — P2 Chunk 3 done：解析加页数上界

- evidence: 接线前发现 `PdfTextExtractor` 的选项里**只有 `signal`，没有页范围**——「只解析前两页」在当前契约下根本表达不出来。这是 revision 3 的单个 Chunk 3 没料到的：它跨到宿主。用户定：加 `maxPages`。
- change: `PdfExtractionOptions` 与 `ParseDocumentOptions` 各加可选 `maxPages`；`pdf-text-extractor.ts:229` 的逐页循环收上界。**契约明写「解析器可以忽略它，调用方不得假定结果已被限长」**——sidecar 那条路径吃不下这个选项，与其假装统一，不如把差异写进契约。
- **取红判据落在 `getPage` 调用次数上，不在返回页数上**，这是本 chunk 唯一值得设计的一条：解析全篇再 `slice(0,2)` 能让「返回两页」的断言变绿，却一点解析时间都不省——而省解析正是这个改动存在的全部理由。断言写成「第 3 页从未被打开」，假实现就过不去。
- disposition: 不传 `maxPages` 时行为逐字不变，既有 extractor 测试全绿即是这条的判据。
- validation: plugin **721/721**（P1 时 718，新增 3）、core 2060、CLI 71/71、typecheck 四包、lint 0 error（20 warning 既有）、check:boundaries OK。
- boundary: 索引仍在走全文分块——`maxPages` 只是把能力加上，真正用它是 Chunk 4。ADR 0005/0007/0008 未动。
- next: Chunk 4——索引改走 extractor + 摘要提取，产出一到两块；结构化 parser 在索引路径上退场（用户已定）；标题提取改为对所有论文生效，因为 catalog 有标题的只有 1.4%。

## 2026-09-02 — Chunk 4 补一条：extractor 缺 provenance，换引擎不会触发重建

- evidence: 用户问「PDF 内容提取是否支持后期更换引擎」，查证时发现一处不对称。`DocumentParser` 声明 `provenance {id, version}` 与 `capabilities`，provenance 进 chunk 指纹、也进 `sameDerivation` 的复用判定——所以换解析器或升版本，索引会整批重建，这是设计好的。**但 `PdfTextExtractor` 接口根本没有 provenance**：走 extractor 分支时 derivation 被硬编码成 `LEGACY_PARSER_PROVENANCE`（`index-orchestration.ts:666`）。换掉 extractor 实现，索引察觉不到，也不重建。
- **这直接打到 Chunk 4 的设计上**：用户定的「索引走 extractor 不走 parser」把索引挪到了这条没有来源标记的路上，等于把「换引擎自动重建」这层保护丢掉。**这是我在计划里点名的「混合索引」最坏失败模式的第二个入口**——之前担心的是一部分论文 113 块、一部分 2 块；现在是一部分文本来自这个引擎、一部分来自那个引擎。两者同样静默、同样不报错、同样让 `maxChunkCosine` 的相似度失去可比性。
- change: Chunk 4 加一条判据——给 extractor 补 provenance，与 parser 那层对称；取红为「换一个 provenance 不同的 extractor 实现后，既有 ready 记录不得被判为 `reused`」。phase revision 5 → 6。
- **值得记的是发现方式**：这个洞不是测试或回归翻出来的，是回答一个「以后能不能换引擎」的问题时查证出来的。**当时如果照着计划直接写 Chunk 4，它会静默地跟着落地**——因为现有测试全绿，typecheck 也过，没有任何东西会响。
- validation: 尚未改源码，无回归可跑。Chunk 3 的全绿仍是当前基线。
- next: Chunk 4，含新增的这条 provenance 判据。

## 2026-09-03 — 三条 parser 断言的判定：都可删，且都验过接管方

- evidence: Chunk 4 第二部分动手前的必要一步。三条断言（`fulltext-index-orchestration.test.ts:307/344/403`）在索引不再走 parser 之后全部失去对象，但「失去对象」不等于「可以删」——**判据是每条守的失败模式在别处有没有 home**，不是它还编不编得过。
- **判定结果：三条都删，无一需要搬走。** 每条都逐一查到了接管方：`:307` 的 headings/locator 断言在 `fulltext-structured-chunking.test.ts:93-107`（locator 那条**逐字是同一个断言**），derivation 记录在 extractor 侧的 `:431`+`:472`；`:403` 与 `:431` 是逐项孪生；`:344` 见下。
- **`:344` 是唯一真需要想的一条，而 handoff 担心的方向对了**：它守着「selector 逐篇选引擎，索引要记**实际选中**那个而非首选那个」。查证后这条保护在 `sidecar-document-parser-client.test.ts:132-167` 已经钉住，第 164 行直接断言回退后报告的是 fallback 的 provenance——**在 selector 自己的层高上验，比在索引里验更贴切**。所以是「早就有更好的家」，不是「搬家」。
- **删它时有一处必须连带删**：`index-orchestration.ts:139` 的 `expectedDerivation` 取 `parserSelector.preferredParser.provenance`，而 `parseIndexDocument:651` 记的是实际选中的 parser——这处 preferred/actual 不对称目前只有 `:344` 在钉。selector 退出后不对称自然消失，但**若只删测试不删这段分支，就留下一段没有测试守的歪逻辑**。
- **判定过程顺带翻出一处计划低估**：计划第 88 条写「Docling 在索引路径上失去对象」，实测 `buildFullTextDocumentParser`（`plugin/main.ts:1603`）只有 `:1695` 一个调用方，就在索引里——**索引路径即全部路径**。失去调用方的是整条 sidecar 装配，含三个用户可见设置行 `pdfParserSidecar.{enabled,capabilitiesUrl,parseUrl}`。**Chunk 4 只让它们失去调用方，一个都不删**：删用户可见设置是收窄，须用户单独决定。记入 open questions。
- validation: 本次只动文档，未改源码，无回归可跑。Chunk 4 第一部分的 core 2062 / plugin 721 / CLI 71 仍是当前基线。
- boundary: 未改任何测试与源码——判定先落纸，改代码是下一步。phase revision 7 → 8。
- next: Chunk 4 第二部分实现。取红判据落在**解析范围**上（第 3 页从未被打开），不在块数上——块数少可以靠截断伪造。

## 2026-09-03 — P2 Chunk 4 done：索引真正接到摘要提取上

- change: `buildPaperDocument` 改走 extractor 取前两页 → 标题 + 摘要 → 拼一段文本 → 既有分块器。`parseIndexDocument` 变 `extractIndexPages`；parser / parserSelector 退出 `IndexPersonalLibraryFullTextInput`，`extractor` 必填；`preferredParser` 那处 preferred/actual 不对称按判定连带删除。标题改为对所有论文生效。
- **取红判据按计划落在解析范围上**：`PagedExtractor` 记录实际打开了几页，40 页的夹具断言「第 3 页从未被打开」。改前红在 `expected 40 to be less than or equal to 2`——红在被测行为本身，不是在某个副产品上。
- **`maxPages` 传下去而不是切结果，且与摘要提取共用同一个导出常量。** 两个边界各写各的话，「摘要在第 2 页」那 0.9% 会静默退化成兜底——这是本阶段反复出现的同一种失败形状：不报错、只是悄悄变差。
- **实现中翻出一个真问题，比预定的改动更值得记**：分块器默认 `minChunkChars: 16` 会把短标题整条滤掉。走全文时这条过滤是丢页眉噪声，完全正确；**走标题+摘要时它把「有标题、无可用摘要」的论文压成零块 ready 记录**——manifest 里在、检索永远匹配不到、没有任何东西报错。实测 `none` 路径占 3.3%，正是扫描件那批。索引路径改传 `minChunkChars: 0`：这段文本是刻意拼出来的，按定义不含噪声。**发现方式**是一条既有测试（`refreshes fallback titles`）断言 `chunks.length > 0` 变红——它本来测的是别的事，却兜住了这个洞。
- **两条既有检索测试也红了，但红得没有价值**：它们把假向量按 `chunkFullText(page)[0].text` 逐字为键，索引文本一改就对不上。**这是测试耦合到了实现细节**，不是回归。改成按论文里的稀有词选向量，并给夹具换成有标题有 `ABSTRACT` 的真实首页——原来那种一行关键词的假页，索引现在什么也提不出来。
- **变异检验三条，位置都对**：去掉 `maxPages` → 只有「只开前两页」那条红，其余 35 条不动，说明这条判据独立、不靠别的断言兜底；恢复 `minChunkChars` 默认 → 短标题那条红；标题退回只给 fallback → 6 条红。
- validation: core **2062**、plugin **720**、CLI **71/71**、typecheck 四包、lint **0 error / 20 warning**（回到基线；中途因删 sidecar 装配多出 4 条 unused import，已清）、check:boundaries OK。
- boundary: **sidecar 代码与三个设置项一个未删**，只是失去调用方——删用户可见设置是收窄，须用户单独决定。`pdfParserSidecar.enabled` 为真时索引记一条 info 说明它不参与索引，避免设置看着生效实则不然。筛选管线一行未动。
- next: Chunk 5——显式抬 `CHUNK_DERIVATION_VERSIONS`。**本次已因 provenance 与 derivation 变化隐含触发重建，但仍要显式抬**：靠副作用达成的重建，下次有人改回去就静默失效。

## 2026-09-03 — P2 Chunk 5 done：显式抬重建版本，并抓到一条自己写的假绿

- change: `chunkerVersion` 2 → 3，`embeddingInputVersion` 1 → 2。两个都抬——块的形成方式和喂给 embedding 的文本各自都变了，只抬一个会让留下的那个字段记着假话。常量上补了注释说明每个数字对应什么改动。
- **本 chunk 真正的收获是抓到一条我自己刚写的假绿。** 最初的测试把两个版本号一起改老，红绿都正常，看着没问题。变异检验时只把 `chunkerVersion` 退回 2（另一个仍是新值），**37 条全绿**——因为剩下那个字段的差异已经足够触发重建了。也就是说那条测试只钉住了「至少有一个字段变了」，`sameDerivation` 里少比一个字段它同样发现不了。
- **改法是把一条拆成两条，每条只让一个字段变老。** 拆完之后四次变异全部红在对应的那条上：退 `chunkerVersion`、退 `embeddingInputVersion`、从 `sameDerivation` 里删掉任一字段。**这比原来的写法多守住了一件事**——不只是「版本抬了没」，还有「这两个字段是不是都还参与复用判定」，而后者原先整个 orchestration 测试里没有人守。
- **和 Chunk 2 那次是同一类错误的第三次出现**：绿也必须绿在被测行为上。前两次分别是断言查错了页、断言落在块数而非解析范围上；这次是一个断言被另一个字段的副作用顶绿了。共同点是**只看红绿不看红在哪，就会把「别的原因导致的绿」当成保护**。
- 测试里的 2 / 1 写死为字面量，不读常量——读常量会让断言跟着常量走。这两个数字是磁盘上现存索引的真实版本。
- validation: core **2064**、plugin 719、CLI 71/71、typecheck 四包、lint 0 error / 20 warning、check:boundaries OK。
- boundary: 只动了常量与测试。用户下次建索引会整库重建，这是 ADR 0013 §3 认下的代价。
- next: Chunk 6——冻结语料复量 + 确认聚类没塌。**这一步有外部阻塞**：需要真实重建索引，而 vault 配的是远程 embedding（`nomic-embed-text`，`http://100.64.2.3:8081/v1`），该端点从 P1 起就没验证过可达。开工前先确认它通不通。

## 2026-09-03 — Chunk 6 做了不需要 embedding 的那一半：真解析器复量，并翻出两处与计划不符

- **端点不再是阻塞**：`http://100.64.2.3:8081/v1/models` 8 毫秒返回 **401**，服务在跑，缺的是 key 而不是连通性。P1 起挂着的那条「未验证可达」可以撤了。
- **真解析器复核完成**（`pdfjs-dist` + `ObsidianPdfTextExtractor` 注入，走的是和生产逐行相同的路径；临时脚本跑完即删）。与 Chunk 2 的 `pdftotext` 代理量对照，212 篇：

  | | pdftotext（Chunk 2） | pdf.js（真解析器） |
  |---|---|---|
  | marker | 177（83.5%） | **181（85.4%）** |
  | dated | 2（0.9%） | 2（0.9%） |
  | leading-text | 26（12.3%） | **22（10.4%）** |
  | none | 7（3.3%） | 7（3.3%） |
  | 摘要长度中位数 | 1594 | **1471** |
  | 总索引字符 | 337,786 | **335,166** |

  **趋势与数值都对上了**，真摘要命中率 84.4% → **86.3%**，比代理量还略好。abort trigger 的「兜底成多数」离得很远。解析吞吐 **29.5ms/篇**（只解析前两页），千篇约 30 秒——「分钟级」在解析这一侧成立。
- **发现一：「每篇恰有一到两块」这条写法与实测不符。** 块数分布是 0 块 1 篇、1 块 164 篇、3 块 3 篇、**4 块 44 篇**（21.7% 超过两块），最大 4。原因是算术上的必然：`MAX_ABSTRACT_CHARS = 2400`，而分块目标 `DEFAULT_TARGET_TOKENS = 512` 换算成 2048 字符，**标题+摘要一旦逼近上界就必然被切开**。
  - **不建议改分块目标**。512 token 是对着 e5 系模型的上下文窗口定的，抬上去等于把截断推给模型，那是更坏的失败。
  - **该改的是计划里的措辞**：真正要守的是「体积回到 MB 级、不因怪版式退回全文量」，这条**达成得非常充分**——每篇均值 **1.65 块**，对比 Chunk 1 基线的 113.2 块，降到 1/68；总块数 23,423 → **349**。最大值 4 由 `MAX_ABSTRACT_CHARS` 硬顶死，不存在版式导致的反弹。
- **发现二（真缺陷）：完全提不出文本的论文会存成「ready 但零块」。** 冻结语料里 `hello-algo_1.3.0_zh_python.pdf`（一本混进来的中文算法书，前两页是封面）标题与摘要都为空，索引文本为空串，产出 0 块——**manifest 里状态 ready，检索永远匹配不到，且不报错**。这与 Chunk 4 修掉的「短标题被噪声过滤吃掉」是同一个失败形状的残余：那次 `minChunkChars: 0` 只救了「有标题但很短」，救不了「什么都没有」。**留待决定**：应当记为 `failed`（用户能在运行摘要里看见，下次重试，代价只有两页解析），还是维持现状。本 chunk 是测量性质，未动源码。
- validation: 未改源码，无回归可跑。Chunk 5 的 core 2064 / plugin 719 / CLI 71 仍是当前基线。
- boundary: 临时测量脚本已删，工作区只剩文档改动。
- **Chunk 6 未完成**，剩下的三项都要真实 embedding：索引体积实测（按 349 块 × 768 维估算约 1.4MB，对比基线 140MB，但这是估算不是实测）、含 embedding 的重建墙钟耗时、以及**聚类是否仍能产出方向**。全部卡在那个端点的 API key 上。

## 2026-09-03 — Chunk 6 done：全部实测完成，并翻出旧索引本来就是塌的

- **先纠正上一条记错的前提。** 「卡在 API key」是我把端点搞混了：`http://100.64.2.3:8081/v1` 是 **LLM**（`settings.llm.baseUrl`，跑摘要用的 deepseek），而 **embedding 配的是本机 Ollama** `http://127.0.0.1:11434/v1`，模型 `nomic-embed-text` 768 维，key 是 6 字符的占位串（Ollama 不校验）。Ollama 在跑、模型装着、能算出 768 维向量——**Chunk 6 从来就没有被阻塞**。教训：查阻塞时要读配置里那一项的真实字段，不要凭「远程 embedding」四个字去猜端点。
- **全流程实测**（真 pdf.js + 真 Ollama embedding + 真 clusterer，`modelId` 为 `remote:nomic-embed-text:768`，与 Chunk 1 基线逐字相同，可比）：
  - **212 篇 / 349 块 / 每篇 1.65 块**；向量 **1.07MB** + 文本 435KB ≈ **1.5MB**。
  - **重建 14.8 秒**（解析 6.2s + embedding 8.7s），**70ms/篇** → 千篇约 **70 秒**。「分钟级」成立。
- **本 chunk 最重要的发现：旧的全文索引本来就是塌的，而且塌的正是判据点名的那一种。** 把磁盘上幸存的那份 140MB 索引用同一个 clusterer、同一个模型、并加上生产的 `MAX_CLUSTERING_CHUNKS_PER_PAPER = 80` 上限跑一遍：

  | | 旧（全文，142MB） | 新（标题+摘要，1.5MB） |
  |---|---|---|
  | 每篇块数 | 66.8（封顶后；存储实为 114.8） | **1.65** |
  | 簇数 | **2** | **21** |
  | 最大簇占比 | **95.6%（195/204）** | **10.4%（22/212）** |
  | 离群 | 7 | 130 |
  | 聚类耗时 | **62 秒** | **63 毫秒** |

  判据原文是「不得退化为『一个大簇』或『每篇一簇』」。**旧索引就是「一个大簇」**：96% 的库挤在一个簇里，方向根本分不出来。原因与 clusterer 自己注释里警告的「weak best-passage coincidence」一致——每篇上百块时，任意两篇天文论文之间总能找到某一对段落相似（方法、仪器、常见措辞），于是人人相连。每篇只剩一到两块之后，信号收敛成「这篇讲什么」，真主题才分得开。
  - 所以本阶段**不是「确认聚类没塌」，而是聚类从塌的状态恢复了**。这比计划预期的强，也顺带解释了 goal 里「方向太粗」的体感从哪来。
  - 新的 130 篇离群（61%）是「没有近邻」的论文，不是退化——退化成「每篇一簇」会是 0 簇 / 212 离群。方向是否变粗仍需人读一遍新旧方向对比，**判断权在用户**（计划原文如此）。
- validation: 未改源码；两个临时脚本跑完即删。Chunk 5 的 core 2064 / plugin 719 / CLI 71 仍是基线。
- boundary: 只读磁盘上的旧索引，未改写；vault 的 `libraryConnection` 现指向 `small_library`，未动。
- next: P2 的六个 chunk 全部完成。**留给用户的两项**：(1) 空文本论文存成「ready 但零块」要不要改判 `failed`；(2) 读一遍新旧方向，判断方向粒度是否可接受。

## 2026-09-03 — 收掉留给用户的两项

- **用户定：空文本改判 `failed`。** `buildPaperDocument` 在索引文本为空时抛错，落进既有 catch → `recordFailed`，运行摘要里可见、下次自动重试、代价只有两页解析。错误文案写明「首两页里既没有标题也没有摘要」。变异检验：拿掉这个守卫，对应那条测试立刻红。core **2065**、plugin 719、CLI 71、typecheck、lint 0 error / 20 warning、boundaries 全过。
- **用户定：方向粒度用「列出新簇的论文标题」来判，不跑 LLM 合成。** 理由是**这个库里根本没有旧的合成方向可比**——vault 的两个主题各只有一条 P1 迁移来的手写句子，方向合成从未在此库上跑过。所以「新旧方向对比」这个说法本身不成立，能比的是聚类分组是否是说得出名字的主题。产物写到 `~/Desktop/plugin_test/clusters-after-adr0013.md`（21 个簇 + 130 篇离群，带论文标题），临时脚本已删。
- **读下来的结果值得记**：簇 2（12 篇）**全是神经网络测光红移**，簇 3（7 篇）**全是星系团星表（Wen 系列）**——正好对上用户已有的两个主题 `Photo-z` 和 `Galaxy Cluster`，说明聚类抓到的是真主题而不是巧合。簇 1（22 篇）是唯一松散的一个：巡天/宇宙学/SZ/X 射线混在一起，还混进了一本量子计算教材、一篇射电望远镜和一篇神经网络校准——**尾部那批「不属于这个库」的文件都落在这里**。其余 18 个簇都是 2–3 篇的小簇。
- **粒度判断仍在用户手上**，但可以说的是：能对上既有主题的两个簇是干净的，粗的只有最大那个，而它粗的原因看起来是语料里混了非天文文件，不是聚类本身失灵。

## 2026-09-03 — 用户指出重复论文，暴露我那份清单的一个缺陷

- **先更正自己的数字。** 上面那份「21 个簇」的清单**把论文标识写成了文件名**，绕过了产品真正的去重——产品对识别不出的论文用内容寻址键 `file:sha256:<PDF 字节>`，**字节相同的文件本来就会合并成一篇**。冻结语料 212 个文件按字节去重正好是 **207 篇**，也正是 Chunk 1 基线里那个 207。按产品方式去重后重跑：**簇数 21 → 18**，最大簇 22 → 21，第二簇 12 → 15。原清单里的簇 13（Sadeh2016）之类纯粹是我造出来的假簇。
  - **教训与本阶段前几次同形**：测量脚本为了绕开 store 而自造标识，就等于悄悄改掉了被测系统的一个关键行为。绕过 store 是对的（Chunk 6 不测 store），但**标识不能跟着一起绕过**。
- **用户指出的重复是真问题，只是层次不同。** 字节去重解决的是「同一个文件复制了两份」；**用户指的那类是「同一篇论文的两份不同文件」，字节不同，内容哈希抓不到**。按提取出的标题归一化后比对，冻结语料里有 **7 组**：
  - `Raichoor2023` / `Raichoor2023 1`（5,329,696 vs 5,341,280 字节）
  - `Vikhlinin2009_1` / `Vikhlinin2009_2`
  - `Wen2009` / `Wen_2009`、`Wen2011` / `wen_galaxy_2011`、`Wen2015` / `wen_calibration_2015`、`Wen2024` / `Wen_2024_ApJS_272_39`
  - `Zhou2023` / `Zhou2023_1`
- **影响比看上去大**：7 组里有 **4 组是 Wen 系列，全都落在用户认可的那个「星系团星表」簇里**。也就是说那个 7 篇的簇实际只有 4–5 篇不同的论文，**它的凝聚度被重复论文虚高了**。近重复还会凭空造出 2 篇的小簇，直接污染方向列表。
- **未决（新增）：近重复是否要在聚类前合并，以及用什么判据。** 归一化标题精确相等是可行且**不含阈值**的判据（goal 的 Constraints 明确警告阈值不得拍脑袋定，所以向量相似度 > 0.98 这类做法要慎重）；但标题相等也会误合并同名不同篇的情况（勘误、同名章节），且标题提取失败的论文参与不了。**另一条路是不自动合并、只把疑似重复报给用户**，由用户清理文献库——本阶段的一贯原则是宁可可见也不要静默。留给用户定。
- 产物：`~/Desktop/plugin_test/clusters-deduped.md`（去重后的 18 个簇 + 7 组近重复明细）。临时脚本已删。

## 2026-09-03 — 按用户决定：标题相等即合并，并把合并结果报出来

- **用户定：归一化标题精确相等就合并，同时报出合并了哪些。** 合并做在 `buildClusteringInput`（聚类输入这一层），**不动索引里的论文身份**——两份文件都是用户真实拥有的，索引照常各存各的，只是聚类时算一篇。返回值从数组改成 `{ papers, mergedDuplicates }`，proposer 新增可选回调 `onDuplicatesMerged`，非空才调。
- **一处需要判断的地方：标题多短就不足以标识一篇论文。** 定为归一化后 **30 字符**下限。理由写进注释：字节去重之后还能撞上的只有「同论文不同文件」，这时标题相等是强证据；但 `Erratum`、`Introduction` 这种通用标题相等不是证据，合并它们会把无关论文静默并成一个聚类成员。**这是保守下限不是调出来的参数**——没有拿任何语料去拟合它，位置远高于通用残句、远低于真实论文标题。
- **在真实语料上验过合并结果**：207 篇里合并 **7 组、减少 7 篇**，与人工核出的完全一致，**没有多合并任何一组**。另有 14 篇标题过短、2 篇无标题，按规则不参与合并——保守侧留白。
- **变异检验两条都红**：把下限降到 0 → 「Erratum 不该合并」那条红；去掉合并动作 → 「相等标题要合并并上报」那条红。
- validation: core **2068**（+3）、plugin 719、CLI 71、typecheck、lint 0 error / 20 warning、boundaries 全过。
- **另一件已澄清、无需改动的事**：用户提出「离群太多，暂时不作为方向」。查证 `personal-library-direction-proposer.ts` —— **离群本来就不进入方向合成**，proposer 只遍历 `clustering.clusters`，每簇一次模型调用；`outliers` 只被增量放置当待定池用（`recluster.ts`），等后续论文进来再看能不能成簇。现状即用户想要的行为，未改一行。

## 2026-09-03 — P2 收尾，P3 开工前定下三条

- **P2 标记 done。** 验收标准「索引只覆盖标题与摘要，千篇量级分钟级重建」勾上：冻结语料 207 篇产出 349 块、索引 1.5MB（基线 142MB）、重建 14.8 秒，千篇外推约 70 秒。聚类从旧全文索引的 2 簇 / 最大簇 95.6% 回到 18 簇 / 最大簇 10.4%——本阶段是把塌掉的聚类救回来，不只是没弄坏它。
- **用户定：测试库里混着的无关 PDF 落进离群，是可接受的现实，不清库、不为此调聚类参数。** 理由是真实用户的文献库同样会混入杂七杂八的 PDF，产品必须在这种语料上成立。**这条同时改变了上一份阶段记录的判断**——那里把「最大簇偏粗」归因于语料脏、建议清库；现在这不再是待办，粗簇是产品要接受的输入形态。离群本来就不进入方向合成（`personal-library-direction-proposer.ts` 只遍历 `clustering.clusters`），因此无需改动一行。
- **P3 开工前定下三条**，都会长出不同的计划，所以先问再写：
  - **命中粒度：一个主题 + 该主题下命中的若干条方向。** 归组仍由单一 tag 决定，日报分节结构不动（ADR 0010 §3）；一篇论文可命中同主题下多条方向，不跨主题。理由是「一个主题里跑着几十条线」正是 ADR 0012 想表达的形态，强行二选一会把它压回去。
  - **主题名仍只作标签，不参与判定。** 收掉 ADR 0012 line 45 与 goal Non-goals 留的未决项。硬门需要一个「像不像本主题」的判据，属于 Constraints 明令不得拍脑袋定的阈值类决策，且是随时可加的增量。
  - **画像文档驱动的第二个分类器不在 P3 退休**，留到 P4/P5。**推论写进了 P3 的 Assumptions**：主题方向的命中不能借用 `PaperDiscoveryProvenance.directions`——那个字段要求每条方向至少一位代表论文作证据，而 ADR 0012 §4 说被接受的方向不留证据。两种来源必须在日报里可区分。
- **P3 计划已写**（`phases/03-filtering-by-direction.md`，七个 chunk）。其中两条是本 goal 前两个阶段的教训直接变成的判据：Chunk 2 要求两个契约版本号**各有一条测试**（P2 栽过一次，两个写在同一条里，退回一个照样绿）；Phase verification 里钉了一条**真实端点小批量实测**，因为单测里的 LLM 是桩，桩通过不构成「真模型会回报合法方向标识」的证据。
- **Chunk 3 是本阶段最需要谨慎的一处**：`settings-rollback.test.ts:68` 断言「迁移后筛选提示词逐字不变」，那是 P1 为「不碰筛选」立的护栏，P3 会正面推翻它。要守的是**数据侧可回滚**（旧构建读同一个 `data.json` 仍工作），不是提示词逐字不变；不允许删掉那条断言了事或放宽成「大致相同」。

## 2026-09-04 — P3 done：筛选按方向工作，日报说得出命中了哪几条

- **七个 chunk 全部验收**，两次提交：`42c9e71`（契约变更：提示词按方向出题、结果带回命中方向、两个版本号 1→2、回滚契约改守数据侧）、`38b8456`（命中方向流到日报、可见行与标记、退化行为、CLI 一致）。core 2101、plugin 719、CLI 73、typecheck、lint 0 error / 20 warning、boundaries 全过。**未推送。**
- **方向标识形态定为 `<tag>#<n>`**。方向的存储 id 是 UUID，让模型逐字回抄又长又易错；`<tag>#<n>` 短且**自带所属主题**，于是「论文归在 A 主题却报了 B 主题的方向」变成可检出的违约。含 `|` 与含 `#` 的 tag 都不歧义（按最后一个 `#` 切分）。
- **新标记族另起前缀而不是扩 discovery-provenance 到 v2**，且**排在既有两族之后**。读代码才发现的硬约束：`personal-novelty-marker.ts` 把自己的标记钉在「有没有 discovery 标记」算出来的固定行号上，中间插一族会打乱这套算术。排在最后，两个既有解析器一行不用改，磁盘上的旧日报解析行为完全不变。
- **本阶段抓到两条自己写的假绿**，都是变异检验抓的，形态与 P2 那次同源：
  - 版本号那两条测试原本把陈旧值写成 `CURRENT - 1`，退回版本号时陈旧值跟着降，**永远不相等，测试永远绿**。改成旧契约真正发布过的字面量 `1` 才真红。**判据要钉在不随实现变动的锚点上。**
  - 标记「必须在规范槽位」那条规则删掉后**没有任何测试变红**——原以为「标记跑到 block 外面」覆盖了它，其实那条是被另一个守卫抓住的。补了「在 block 内但不在槽位」的测试后才真红。
- **计划里有一条预期是错的，如实记下**：Chunk 7 原写「红在 CLI 侧仍只喂 description」。实际 CLI 走的就是同一个 `buildPaperFilterRequest`，Chunk 1 之后它已经跟着变了，**根本取不到红**。那两条 CLI 测试是确认跟随成立的表征测试，不是 Red-Green。
- **Chunk 5 的实现写在了测试之前**（marker 模块整体成形后才补测试），不满足 strict Red-Green。补偿验证是逐条变异检验，四条规则各删一次都红。记在阶段计划里。
- **退化行为定死了两条**：没有方向的主题**整个不进提示词、也不进合法 tag**（留着等于邀请模型把论文归进去、再因为报不出方向而触发违约，**一次配置问题葬送整次运行**），一条方向都没有时直接不调 LLM；两种情况都 warn 出被跳过的主题名。配置检查文案从「description is empty」改成「has no directions」——影子字段在设置页里根本不显示，指着它让用户去修是指错了地方。
- **真实端点实测通过，但带一条要留给 P6 的观察。** 真端点 + 9 篇真实论文跑两轮，标识全部合法、严格解码直接通过、3 篇 hep-th 两轮都判 skip。**但 18 次判定里一次多条命中都没出现**——提示词明写了「可以同时命中多条、全部列出」，模型仍只挑最匹配的一条，包括那篇既是 SZ 选源星表又做质量标定的 SPT 论文；`redMaPPer 子结构`那篇也两轮都判 skip。契约允许 1..n 且单测覆盖了，**但多条命中在真模型上没有活证据**。这是召回口径问题，取决于方向文本怎么写，P6 拿真实日报判。

## 2026-09-04 — P4 开工前：判定画像文档那套的退休范围

- evidence: 这批代码**从未发布过**。`docs/releases/` 里 0.3.0–0.4.6 共 13 份发布说明零次提到 personal library；README 只有一段 "Personal library access (desktop preview)"，且那段写着「The current preview does not change daily filtering, reports, paper notes, or email delivery」——**这句话现在是假的**：`buildPipeline()` 每次日报都调 `buildPersonalizedDailyDiscoverySnapshot()`（`plugin/main.ts:3083`），没有任何 feature flag。退休第二分类器不是撤回承诺，反而是把行为拉回文档说的样子。
- evidence: **CLI 一行都不涉及**。`apps/cli` 对 personalized / interest-profile / libraryProfile 的引用数为 0，整条路径是插件独有的。goal Constraints 里「插件与 CLI 行为一致」这条在此不构成约束。
- **退休范围的大部分不是选择，是被 P4 的验收标准逼出来的。** 接受动作一旦写进 `settings.topics`，`interest-profile.json` 就失去唯一的写入者，挂在它下游的整条链同时失去主语：资格判定 → 日报快照 → 第二分类器 → provenance → novelty，以及 incremental 的 attach/split/merge。问题不是「要不要一起退休」，而是「改道之后它们还有没有输入」——没有。
- **存活**：`personal-library-direction-proposer.ts`（P4 要在它上面加粗/细两级）、`direction-proposal.json` 与 store 的候选那一半（ADR 0012 line 41 明确要求候选仍需持久家）、catalog / clustering / fulltext 索引、`incremental/` 的打分半边（placement / recluster / diff-suggestions / suggestions-store 约 1400 行，算法不变，P5 只换写入目标）。
- **用户定：失去主语的那批物理删除，不留着不接线。** 范围是 profile schema、`personal-library-interest-profile-review.ts`、profile store 那一半、`evaluatePersonalLibraryInterestEligibility`、`buildPersonalizedDailyDiscoverySnapshot`、`personalized-paper-filter.ts`、`PERSONALIZED_LIBRARY_ONLY_CATEGORY` 的日报分节与 `discoveryProvenance` 的**生产端**，约 3000 行源码 + 约 4000 行测试。理由：从未发布、CLI 无引用、无迁移负担，git 历史随时可取回；留着不接线会多出一大片永不执行却仍在跑的测试。**P2 对 sidecar 的处理不构成先例**——那次不删是因为有用户可见设置，这里没有。
- **停产但解析必须保留**：`personal-library` 分节与 discovery-provenance 标记族不再产出，但磁盘上的旧日报仍要能被 `paper-index` 与 `dashboard/history-sync` 正确读回。这是删除动作的第一条 abort trigger——**红要红在「旧日报解析行为逐字不变」上**。
- **用户定：personal novelty 休眠并记账，不在 P4 内删也不改基准。** `personalized-novelty.ts`（1388 行 + 1200 行测试）在输入断供后该 stage 永不触发。删掉它等于替用户决定了 Non-goals 第 4 条明令不在本 goal 内决定的事（ADR 0012 line 43 明说 novelty 未必要走，基准可以改成检索索引里最相似的论文）。**这是本 goal 结束时必须交还给用户的一笔显式欠债。**
- **被迫接受的一条暗期**：incremental 建议界面在 P4 之后到 P5 之前是暗的——它的建议挂靠对象是 profile 里的已确认方向，profile 空了就没有可挂靠的东西。`incremental/apply.ts`（450 行）的三个 `apply*Suggestion` 都返回 `PersonalLibraryInterestProfile`，写入目标要换成 topics，属 P5（ADR 0014 §2/§3）。P4 计划要写明这段暗期是预期而非缺陷。
- **P4 要新造的一小块机器：接受时生成唯一 tag。** `normalizeTopic`（`packages/core/src/settings/topics.ts:68`）是所有主题的唯一入口，但它**不管 tag 唯一性、也不从 name 派生 tag**；重复 tag 是在 `validation.ts:158` 被判为错误的。按 ADR 0014 §1 直接写入会产出一份校验不过的设置。这就是 ADR 0014 Consequences 那句「凡是守设置写入的机制都要覆盖它」的具体落点。
- **复审页必须重写并桌面验收。** `plugin/src/library/interest-profile-modal.ts`（970 行）现在的主体是「逐条确认方向」，P4 要变成「接受一整套主题（含各自的方向）」。goal Constraints 点名复审页从未进过桌面验收，**测试绿不构成交付证据**。
- validation: 未改源码，只读与文档。P3 的 core 2101 / plugin 719 / CLI 73 仍是基线。
- boundary: 只追加本条 journal，未动 goal.md 的阶段状态（P4 仍为 pending，待计划落盘再说）。删除了 `docs/handoffs/2026-09-04-p4-library-proposes-topics.md`（其全部未决项在 P2/P3 计划与本 journal 里均有留存），目录随之为空一并移除。
- next: 起草 P4 阶段计划 `phases/04-*.md`。先读 ADR 0014 §1 与 Consequences 最后两条；粗/细两级聚类不需要新机器，同一批向量跑两遍 `clusterVectors`，靠 `relativeStopRatio` 区分（`clusterer.ts:95`，proposer 现用 0.65，见 `personal-library-direction-proposer.ts:712`）；**粗聚类比例必须标为「待实测的调参旋钮」而非默认值**。

## 2026-09-04 — P4 计划落盘；写计划时翻出提议器的两处缺陷

- **P4 计划已写**：`phases/04-first-scan-proposes-topics.md`，七个 chunk。顺序是提议侧（两级 schema + 一行方向 + 生成契约修真 → 粗/细两级聚类 → 主题建议名）→ **停下来实测粗聚类比例交用户判** → 写入侧（唯一 tag 派生）→ 复审页重写（含桌面验收）→ 物理删除旧路径。
- **缺陷一：聚类提议器的生成契约在说谎。** `createPersonalLibraryClusteredDirectionGenerationContract`（`personal-library-direction-proposer.ts:733`）记 `synthesisPrompt: "none"`、`strategy: "...-no-synthesis"`，模块头注释也写着 "skip the cross-cluster synthesis stage"，**但代码实际跑了综合阶段**（同文件 :946）。契约的自述职责是让参数漂移可从提案里检出，而整整一个 LLM 阶段连同其提示词版本对该指纹不可见——改综合提示词不会让任何旧提案失效。形状是当初真跳过、后来按 ADR 0009 §2 加回来，契约与注释没跟上。P4 Chunk 1 修。
- **缺陷二：提议器在冻结语料上会当场抛错，而这从未被观察到。** `renderClusteredExtractionMessage`（:1047）对超过 20 篇的簇硬抛 `evidence-too-large`，而 P2 实测去重后**最大簇 21 篇**。P2 的测量是临时脚本直接调 `clusterPaperVectors`，提议器本身从未在这个库上跑过，所以这条撞线一直没露面。粗聚类只会让簇更大，P4 第一次真跑必然撞上。**另有一处自相矛盾**：schema 允许 512 成员的簇（`PERSONAL_LIBRARY_MAX_CLUSTER_MEMBERS`），提取见到 21 就抛。
- **用户定：方向的一行文本由提取提示词直接产出**，不做「name + description + cues 拼成一行」的确定性折叠。理由是拼出来的一行会直接变成用户设置页里的正式文本，且提议预览与最终文本对不上。cues 仍产出但只供复审页帮判断，不进 settings。**推论**：`origin: "library"` 在 P4 第一次有生产者——P1 把它加进枚举时没有任何写入方（至今 `grep 'origin: "library"'` 零命中），所以这条路是 P4 接通的，不是既有行为的延续。
- **用户定：抬提取的篇数上限，让代码单位上限当真守卫。** 真正的约束是消息体积（`MAX_BATCH_CODE_UNITS = 60_000`），20 篇只是它的代理，21 篇的标题+摘要远不到 60k。**不引入新阈值**——超体积仍然抛。备选的「簇内分批再综合」被否掉，因为它把「两批说的是不是同一条方向」的判断又交回给模型，而细聚类本来就是为了避开这个判断。
- **一行的长度不新造常数**：沿用既有的 `PERSONAL_LIBRARY_MAX_DESCRIPTION_LENGTH`（1000）作 DTO 硬上限防膨胀，「一行」由提示词表达。goal Constraints 已把两个阈值列为必须实测，再添一个拍脑袋的数是同一个错误的第三次。
- **计划里显式写进去的一条判据**：粗聚类比例的实测判据是「提议出的主题是否说得出名字」，不是「离群率低不低」（2026-09-03 用户定，混进来的无关 PDF 落进离群是可接受的现实）。判断权在用户；判不出来就作为调参旋钮交付并在 Open questions 里标明。
- validation: 未改源码，只读与文档。P3 的 core 2101 / plugin 719 / CLI 73 仍是基线。
- boundary: 只新增 P4 计划与本条 journal。goal.md 未动——P4 状态仍为 pending，按本 goal 的惯例等实现完成才标 done。
- next: 用户确认计划后开 Chunk 1（提案文档两级 + 提取产出一行 + 生成契约修真，三者与 Chunk 2/3 合并为一次提交，各自变异检验分开做）。

## 2026-09-04 — P4 计划改序（revision 2）；Chunk 1 删除做到一半，**未提交**

- **计划顺序有硬伤，已修正。** revision 1 把「物理删除」排在最后（Chunk 7），但 Chunk 1 要把提案文档改成两级，而它的消费方（复审逻辑、复审页、store）正是要删的东西——先改后删等于为将死的代码做一次移植。删除移到 Chunk 1，其余顺延。
- **又一处无调用方的死代码**：未聚类提议器 `proposePersonalLibraryDirections` 与整个 grouping 阶段（约 400 行 + 一个提示词 + 551 行测试）只被自己的测试引用，插件走的是 `proposeClusteredPersonalLibraryDirections`（`plugin/main.ts:959`）。它与聚类提议器共用 Chunk 2 要改的提取契约，按同一条原则一并删，已写进计划。
- **一处读代码才发现、会改变删除边界的事实**：`parseDiscoveryProvenanceMarker` 用 `renderDiscoveryProvenanceMarker` 做规范形式校验（round-trip，`discovery-provenance-marker.ts:137`）。**渲染函数是解析路径的一部分，不是写入方**——删掉它等于删掉解析器的完整性校验。所以「删生产端」的正确边界是「管线不再调用它」，标记模块整体保留。已把这层意思写进该模块的文件头注释。
- **另一处**：`personal-library` 分节的标题只被写、从不被解析（`parseDailyReportDiscoveryProvenance` 按 `###` 论文块与标记行定位，不看 `##` 分节标题）。所以删掉分节产出对旧日报解析零影响。
- **已完成（core 侧全部 typecheck 通过）**：provenance 的类型与五个边界常数从被删的 `personalized-paper-filter` 搬进标记模块并改名 `DISCOVERY_PROVENANCE_MAX_*`；`filterPapers` 的 union 分支删除，只剩单一分类器；pipeline 的 personalized 依赖与 novelty stage 摘除（novelty 模块本身保留）；assembler / rescue 的 `personal-library` 分节删除；filter checkpoint store 的 personalized 读写删除（**`removeAll` 的清理保留**，否则磁盘上已有的 `.personalized.json` 永远没人收）；`personalized-paper-filter.ts` 与其 580 行测试删除。core src 净减约 750 行。
- **未完成，工作区留着未提交**：插件侧还剩 21 个 typecheck 错误，全部来自同一条线——`markPersonalizedDailyDiscoveryUnavailable` / `restorePersonalizedDailyDiscoveryAvailability` / 沿方法签名穿下去的 `discoveryRevision`，约 20 处调用点。**这套机制存在的唯一目的就是给已删掉的日报快照做可用性门控**，所以整条要拆，但它缠在文献库操作的生命周期守卫里，需要逐处读过再动，不能机械替换。此外画像文档模块本体（profile schema 那一半、review、store 那一半、`incremental/apply.ts`）、复审页的减法、以及测试清理都还没做。
- validation: **树当前编译不过**（插件 21 个错误），因此没有提交。core 2101 / plugin 719 / CLI 73 的基线在动手前已复核，与 P3 一致。
- next: 接着拆 `discoveryRevision` 那条线到编译通过，跑四包回归确认「旧日报解析逐字不变」，再作为 Chunk 1 的第一次提交。

## 2026-09-04 — P4 Chunk 1 完成，Chunk 2 起了个头（三次提交）

- **计划改序（revision 2）**：删除从 Chunk 7 提到 Chunk 1。原顺序有硬伤——Chunk 2 把提案文档改成两级会立刻打断它的消费方，而那些正是要删的东西，先改后删等于为将死的代码做一次移植。
- **`e7498dc` 日报侧退休**：删掉第二分类器与 union 分支、日报快照装配、`personal-library` 分节、provenance 生产、novelty stage 接线。插件那套「日报要不要走文献库」的可用性门控降级成只增不减的 `libraryMutationRevision`——它真正还有读者的地方只有「授权对话框期间文献库变没变」这一条守卫。core 2101 → 2051。
  - **读代码才改掉的删除边界**：`parseDiscoveryProvenanceMarker` 用 `renderDiscoveryProvenanceMarker` 做 round-trip 规范形式校验，渲染函数是**解析路径的一部分**，不是写入方；`personal-library` 分节标题只被写、从不被解析（解析按 `###` 论文块定位）。
  - **变异检验抓到一个真空档**：删掉那个 round-trip 校验，**整个 core 套件没有一条测试变红**。先补 `discovery-provenance-marker.test.ts`（伪造一个内容合法但序列化不规范的标记，必须被拒），再变异，才只红在它自己那条上。这个文件同时是「旧日报解析逐字不变」的独立回归——管线已经产不出标记，没法再靠 round-trip 来测。
- **`8afff77` 画像文档退休**：schema、store、已确认方向的全部操作、资格判定、`incremental/apply.ts` 删除。**拆分而非整删**：提案编辑拆成 `personal-library-proposal-review.ts`，提案 store 拆成 `personal-library-proposal-store.ts`，`incremental/` 的打分三件套改吃新的 `PlaceableDirection`（按「打分实际读什么」写出来，不从已不存在的 store 借类型，P5 只改适配层）。core 2051 → 1933，plugin 719 → 637。
  - **顺带退休一条产品规则**：ADR 0010 §1 的「有主题 或 文献库有可用方向，任一足够开跑」。它服务的分类器没了，现在没有主题就不能跑。这个改动是以两条 validation 测试变红的形式暴露的，换成了一条新判据（拒绝信息不再提文献库）。
  - **incremental 建议界面此后到 P5 之前是暗的**，计划里写明是预期。
- **`079abf1` Chunk 2 起手**：删掉无调用方的未聚类提议器与整个 grouping 阶段（`selectPersonalLibraryDirectionPapers` 插件还在用，保留，测试移到 `personal-library-paper-selection.test.ts`）；**修掉生成契约说谎**——它记 `synthesisPrompt: "none"` 而代码实际跑综合。同样先确认这条没有守卫（改回 `"none"` 无人变红），补测试后再变异，只红在那一条。**与标记那次是同一种空档形状，本 goal 第二次撞上。**
- validation: core 1914 / 2 skipped、plugin 637、CLI 73、typecheck 干净、lint 0 error / 20 warning（基线）、boundaries 通过。三次提交合计净减约 12800 行。**未推送。**
- **一笔要记的账**：插件那两个大测试文件（modal 950 行、lifecycle 530 行）是按「块里提到已删 API 就整块删」处理的，不是逐条判断哪些断言仍有价值，可能误删了仍然成立的覆盖。Chunk 7 重写复审页并做桌面验收时要重新长出来。
- next: Chunk 2 的主体——提案文档变两级（schema 3→4）、提取提示词直接产出一行方向、cues 只供复审页不进 settings；然后 Chunk 3（粗/细两级聚类、篇数上限让位给体积上限）与 Chunk 4（每个粗簇一次命名调用），三者按 L1 合并为一次提交。

## 2026-09-05 — P4 Chunk 2–4、6–7 代码完成；**P4 未完**，卡在两处需要用户的事

- **`d709f85` 提议侧改成两级**（Chunk 2–4 按 L1 合并）：提案 schema 3→4，扁平候选列表变成「主题（带建议名）→ 它的一列一行方向」；方向从 `name`+`description` 变成一行 `text`，由提取提示词直接写出，cues 只供复审页不进 settings。聚类跑两遍——粗（→主题）+ 细（→方向），**综合只在一个主题内做，不跨主题**（方向属于且仅属于一个主题）。新增命名阶段：每个粗簇一次调用，失败退化为可辨认的占位名。
  - **变异检验抓到一个设计问题**：细聚类默认会在粗簇内部重新做语料居中，而居中减掉的正是「让这批论文成为一个簇」的共同信号，剩下近乎正交的残差把粗簇打散（测试里 6 篇输入只覆盖 5 篇）。细聚类因此默认关掉居中、走原始余弦；拆掉这条，两个成员测试立刻红。
  - **两个上限的矛盾修掉**：提取对 >20 篇的簇硬抛错，而 schema 允许 512 成员——冻结语料就有 21 篇的簇。篇数上限改为与 schema 对齐，真正的守卫仍是消息体积。
  - **粗比例用 `PERSONAL_LIBRARY_UNMEASURED_COARSE_STOP_RATIO = 0.35` 占位**，常数名与注释都写明这**不是实测默认值**。
- **`0155a8f` 接受即写设置**（Chunk 6 + Chunk 7 的代码部分）：`acceptProposedTopics` 从建议名派生 slug tag，并对既有设置与同批兄弟都保证唯一、且确定性；全是非 tag 字符的名字（中日文主题名）退化为序号 tag 而不是猜测转写。所有产物仍走 `normalizeTopic`，影子字段的唯一写者不变；方向带 `origin: "library"`——**P1 加进枚举后的第一个生产者**。复审页补上逐主题勾选、建议名可当场改（改完才派生 tag）、一次接受写入设置，并显式说出「有哪些论文没被任何方向覆盖」与「incremental 建议在重建前不可用」。
  - 变异检验：拿掉 tag 唯一化 → 冲突与确定性两条红；让接受忽略勾选 → 选择那条红。
- **P4 不能由我一个人跑完，两处结构上要用户**：
  - **Chunk 5 粗比例实测**：计划原文「判断权在用户」。而且做不成——磁盘上冻结语料那份索引是**旧的全文索引**（207 篇 / 23423 块 / 113 块每篇 / 140MB），不是 ADR 0013 之后的标题+摘要那一份（应为 349 块 / 1.65 每篇 / 1.5MB）。实测要先在 Obsidian 之外重搭 pdf.js + Ollama embedding 的临时脚手架重建索引。已确认 Ollama 在跑且 `nomic-embed-text` 可用；LLM 端点 `100.64.2.3:8081` 返回 401，需要 key。
  - **Chunk 7 桌面验收**：goal Constraints 点名复审页从未进过桌面验收，**测试绿不构成交付证据**，必须由用户在真实 Obsidian 里看过。
- validation: core 1918 / 2 skipped、plugin 640、CLI 73、typecheck 干净、lint 0 error / 20 warning（基线）、boundaries 通过。**未推送。**
- next: 用户跑桌面验收；粗比例实测需要先决定是否搭那个重建脚手架，或者改为在真实 Obsidian 里扫一次库再导出聚类结果。**goal.md 的 P4 仍为 pending**，这两件事都落地才能标 done。

## 2026-09-05 — L3 steer：LLM 组织少量主题，新增每日 20 篇上限

- evidence: 已消费的交接与当前工作区吻合：HEAD `655a867`，14 个修改文件、6 个非交接未跟踪文件。旧版已在真实 Obsidian 跑通，但 10 个主题、46 条方向被用户判断过细。189 篇输入来自 196 篇索引论文（7 组近重复合并）；平均连接的相似度 max 0.786、p99 0.424、p95 0.287、中位数 0.027。单链接没有可用中间粒度，调整细聚类分位数也只能把方向数从 34 变到 46。
- change: 按用户已确认的方案，由 P7 替代 P4 的粗细两级生成路径，改为紧聚类后全局组织成 2–4 个主题、每主题 1–2 条方向，不写用户配置范例。主题按覆盖量排序，默认选前两个并折叠。用户本轮确认每日总数默认 **20 篇**；P8 在筛选后、摘要前按相关性截断，详情选取只看保留集合。已向用户说明旧筛选缓存会失效。P1–P3 的已验收结论不受影响，P5/P6 保持 pending。
- disposition: 保留未提交的平均连接聚类、索引摘要（标题版本 9）、非 arXiv 证据四处修复、生成时授权、真实错误日志与进度修复。替换逐簇提取/综合/命名、两级参数及其实现绑定测试；保留取消、证据边界、严格解码、接受设置和旧日报解析的契约测试。ADR 0009 的生成路径退休，ADR 0014 §1 与词汇表同步修订。四个临时测量/探针文件在读过后清理。
- verification: 本轮改动前 `clustered-direction-proposer.test.ts` 19 项、`personal-library-proposal-contract.test.ts` 5 项通过。交接记载的 core 1928 / plugin 655 / CLI 73、lint 20 warnings 为历史结果，后续重跑。接手 owner 为 `codex-main-session`；按明确约束不提交、不推送、不开 PR。
- next: P7 Chunk 1 先补组织契约的行为红测，同时独立实施复审页 Chunk 2；构建后由用户查看真实结果。P8 随后实施，不提前改动其他 active initiative。personal novelty 基准、P5 相似度下限、Dashboard 来源展示与全文授权深度仍未决定。

## 2026-09-06 — P7 代码收尾，P8 开始

- evidence: 一次组织调用替代逐簇提取/综合/命名；proposer 23 项与 decoder 61 项通过，真实成员并集/跨组引用变异能触发对应断言。复审 modal 28 项、plugin 全量 670 项通过，取消方向后的真实设置持久化也已覆盖。core 全量、CLI 73、typecheck、lint 0 errors / 20 warnings、boundaries 曾在集成中通过，最终全量随 P8 重跑。
- change: 仅修改生成 fingerprint 不会使旧提案在加载时失效，补 schema 5 与合法 v4 的重生成路径；38 项针对性回归通过。薄证据从代表展示数量改按去重完整成员数判定，4 红→9 绿。原测量脚本 2 份和无断言探针 2 份已清理，旧 3 个阶段提示词退休。
- disposition: P7 的代码与回归保留且未提交，P7 仍等真实模型方向高度与 Obsidian 桌面验收，不能标 done；切到独立的 P8（默认每日 20 篇）。P5/P6 不受本次 phase 转换影响。
- next: P8 在排除 ignored 后、抓正文前按 0–100 relevanceScore 全局排序截断；共享 output 设置正整数上限，缓存保留完整评分。构建与 vault 安装在两项改造都就绪后一起完成。

## 2026-09-06 — P8 完成，构建已安装，P7 待真实桌面验收

- evidence: 每日上限默认 20，插件两套 UI 与 CLI `[output].max_daily_papers` 一致；非法配置被拒绝，旧配置缺失时恢复 20。筛选使用 0–100 relevanceScore，prompt/result 合同分别升到 3；缓存保留所有评分，上限变更可复用。忽略论文先排除，随后全局按分数/规范 ID 截断，正文、详情、摘要、日报索引引用使用同一保留集合。
- verification: 设置、评分、缓存与截断均有观察到的 Red→Green；针对 CLI 0 值、跨主题截断、代表范围、完整成员、复审排序与真实接受写入的变异均能触发目标断言，已恢复。最终 core 2043 passed / 2 skipped，plugin 700 passed，CLI 85 passed；四包 typecheck、boundaries、lint 0 errors / 20 既有 warnings、插件 build 和 diff 检查通过。
- change: 集成复核补上一处摘要预算误拒绝：短/长摘要混排时，空预算下的截断标记可能比正文更大；用独立 abstractTruncated 与码点安全前缀恢复单调预算，302 篇回归先红后绿。P7 的旧 v4 提案已真实失效且可重新生成，薄证据看完整去重成员，方向勾选会影响最终设置。
- delivery: main.js 与 styles.css 已安装到 `/home/tiandc/Desktop/plugin_test/.obsidian/plugins/arxiv-daily/`，cmp 与工作区构建一致。旧文件备份于 `/tmp/arxiv-daily-before-topics-and-cap.l3uAoa/`，临时目录不作长期恢复保证。HEAD 仍为 `655a86754240488b12a0827dbc89a7a2a4d171ae`；全部源码、测试与文档改动保持未提交。
- next: 用户重载 Obsidian 插件并重新生成提案，查看主题/方向高度、默认前两个主题、折叠和方向选择是否可用。P7 保持 blocked（仅待真实模型/桌面验收），P8 done，P5/P6 仍 pending；未实施增量归入或真实日报验收，也未决定 personal novelty 新基准。

## 2026-09-06 — Accept 已写入但设置页未刷新

- evidence: 用户报告 Accept into topics 点击后设置没变化。只读检查测试 vault：最新 schema 5 提案真实生成于 09:29，含 4 个主题、8 条方向；data.json 已有原来的 2 个主题及新 4 个主题的两份，共 10 个。写入并未失败，接受路径缺少手动新增主题路径已有的设置页刷新与成功反馈。原测试只检查 saveData，没有覆盖已打开页面。
- change: 插件保留已注册的 ArxivDailySettingTab，接受并持久化成功后刷新其当前/缓存视图，提示新增数量。若保存成功而页面刷新抛错，记录错误并提示重新打开设置，不让用户误以为需要再次接受。没有清理、覆盖真实设置或提案中的任何内容。
- verification: 新 DOM 回归先复现“文件有新主题、页面仍只有旧主题”，成功提示与刷新异常提示亦先红后绿；32 项弹窗测试、704 项插件全量、插件 typecheck、build 全过，lint 保持 0 errors / 20 既有 warnings。短暂去掉刷新调用时 DOM 回归再次失败，恢复后全量通过。core / CLI 本轮未改、未重跑。
- delivery: 修复版 main.js 已安装到测试 vault，cmp 与构建一致；旧文件保存在 `/tmp/arxiv-daily-before-accept-refresh.nhrexa/main.js`。改动未提交。
- next: 用户重载插件并打开设置查看已有主题，本次提案无需再次 Accept。当前 4 个重复主题保留，清理需另行明确；P7 仍待方向质量及修复版实际交互确认。

## 2026-09-06 — P7 验收通过；P9 完成；新增 P10/P11；owner 交接

- evidence: 用户在真实 Obsidian 里跑完了整条路：提案生成 4 主题 / 8 方向，接受写进设置，**方向高度确认合格**（上一次 L3 的起因是 10 主题 46 方向过细），P7 的桌面验收条件因此满足。同一轮验收暴露三件事：(1) 同一份提案被接受了两遍，测试库 data.json 里 4 个主题各存了两份（第二份 tag 带 `-2` 后缀），接受路径除同名外无任何幂等保护，且复审页每次重开都会默认勾选覆盖量最大的两个主题；(2) 主题卡片折叠标题「名字 + #tag」过长，而 tag 是用户不会去编辑的机器字段；(3) 复审页「未聚类」整段列出论文标题过于嘈杂，与上方内容之间没有任何间隔（该段在样式表里一条规则都没有）。
- change: P7 blocked → done。新增 P9（本轮已完成的验收修正）、P10（提案生成看见已有主题，active）、P11（筛选按具体度裁决，pending）。成功标准第 3 条（首次扫库提议整套主题、接受后即可用、不必手写主题名）勾选。Open questions 里 P7 那半条移除，P5 的相似度下限保留。owner 由 `codex-main-session` 接为 `claude-opus-session`（上一条日志的 next 是等用户验证，工作停在用户侧，无并行 owner）。
- disposition: P9 的四块改动全部保留：接受按主题名去重（`topicNameKey`，忽略大小写与首尾空格，同批重名也只留一条）+ 复审页把已在设置里的主题标 Added、禁选、排除出预选与接受载荷 + 全部跳过时不写设置只提示；设置页彻底移除 tag 的显示与编辑，改名时按「旧 tag 仍是旧名字的机器形式或 `topic-N` 占位」决定是否跟随，并对其他主题去重（tag 重复是 `preflightDailySummaryAssembly` 的硬错误，会让当天整份日报生成失败，而界面已无处修改）；未覆盖证据由标题列表收成一行计数；复审页四处间距。ADR 0014 补一条 consequence 记录列表收成计数**以及其代价**——那些论文的标题此后在产品里任何地方都不再出现。测试库 data.json 里 4 个重复主题已删除（备份 `data.json.bak-20260906-200906-dupetopics`）。
- verification: typecheck 四包干净；plugin 719 passed（42 files）；`packages/core` accept-proposed-topics 9/9；lint 0 error / 20 既有 warning。`packages/node-runtime` 的 `reconstructs filter checkpoints from backup...` 为**既有失败**，已用 stash 回到 `655a867` 复现——测试夹具手搓了一个没有 `directions` 键的 topic 绕过 `normalizeTopic`，真实设置到不了那条路径；该夹具待单独修。构建已装入测试 vault（旧文件备份 `main.js.bak-20260906-*-pretopicui` / `-prereview`）。
- note: 排查 P11 时发现 goal 的 Non-goal「主题名仍只作标签」在代码里是**用 prompt 明文规则**落实的（`paper-filter.system.md`：判断依据是方向，不是主题名），而不是靠不给主题名——给模型的 tag 就是主题名的 slug，拉丁文主题名的词一个不少。原本设想的「把主题名加进 prompt」因此收益近零且与该规则自相矛盾，已从 P11 砍掉，Non-goals 不动、不构成 L3。附带记一笔：中日文主题名派生不出 slug，tag 退化为 `topic-N`，这类主题在筛选 prompt 里没有任何主题级线索——不影响正确性（判断本就只看方向），但中英文主题的信息量并不对等。
- next: P10 Chunk 1——先给组织契约补「一条方向归属已有主题」的行为红测。P11 在 P10 之后做，改动会让已缓存的筛选结果整批失效，实施前再确认一次。P5/P6 仍 pending；C 类工作（把已建好但无调用方的 `library/incremental/{placement,recluster,diff-suggestions}` 接上）仍未立项。

## 2026-09-07 — L2 reshape：接受按方向与稳定目标处理，接手当前 Helm

- evidence: 用户要求 review 后明确“按照你说的开始完成当前 helm”。只读审查与真实函数探针确认：部分接受后其余方向被整主题同名拦截；改名后重复新建；详情评分只见首条方向；P10 不允许零新增，且失配名字回退新建。相关既有测试 217 项通过，未覆盖这些场景。另确认 cues 只供复审但仍可编辑、名额截断后空栏目误称无相关论文。
- change: owner 从已暂停的 claude-opus-session 接为 codex-root。P10 原文件（含用户已有 revision 2 未提交修改）完整保留，标 superseded，启用 P12。P5 未实测向量归入路径由 P12 文字归属与 P14 索引后复审替代；ADR 0014 和词汇表同步。P11 增加实际命中方向参与详细总结的验收；P13 修正名额截断说明；最后回到 P6 真实日报与桌面验收。
- disposition: 保留 P1/P2/P3/P7/P8/P9 的已验收结果及用户设置。P9 的整主题同名接受判据在 P12 被逐方向接受记录替换；P3 明确延期的详情评分问题在 P11 补齐。保留单主题归属、名称只作标签、默认 20 篇及 personal novelty 暂停。未提交、不推送、不开 PR。
- next: P12 Chunk 1 的零新增/已有目标红测；同阶段独立开发接受事务，集成后验证实际持久化与复审流程。

## 2026-09-07 — P12 组织契约接受，保存并发复核与范围校正

- evidence: 组织/schema/store 先 35 Red，历史格式和异常名称追加 Red 后 159 Green；core 接受 15 Red→16 Green；插件事务 8 Red→31 Green；UI 15 Red→16 Green；真实 HTTP/ProposalStore 的设置输入与在途修改 2 Red→6 Green。插件接受/复审/事务联合 96 Green。四包 typecheck、boundaries、lint（0 error/20 既有 warning）通过。
- review: 独立审查通过真实 host 复现：接受保存期间直接编辑 live direction，随后整数组提交覆盖编辑。P12 Chunk 2 尚不接受；设置页改用私有草稿与按身份的串行事务，补 DOM 回归。接受记录按 scope 保留各库当前提案，防止 A→B→A 忘记已删除方向。
- regression: 全 workspace 首轮仅两项失败：node-runtime 旧筛选夹具缺 directions/relevanceScore（既有问题，journal 已记录）与空回执改变历史 envelope。夹具补到真实当前合同；无回执时保持旧 envelope；各自 25/25、40/40 Green。
- scope: 核对 ADR 0007 与上一条接手前日志的“C 类自动增量接线未立项”后，撤回本轮额外提出的 P14 自动触发扩展，保留其编号并标 superseded。原成功标准要求后续候选可建议归属并改动，P12 的再次生成及逐候选改归属已承担，不需要把已退休画像的自动索引触发系统带回来；ADR 0005/0007 不改。P5 的向量阈值路径由 P12 文字复审替代。
- next: 完成设置编辑的并发保护及 P12 集成检查，然后 P11/P13，最后 P6 真实模型/日报和用户桌面验收。仍未提交。


## 2026-09-07 — P12 软件验收通过，进入 P11

- verification: core 2133 passed / 2 skipped、node-runtime 45、CLI 85、plugin 772；四包typecheck、boundaries、build通过，lint 0 error/20既有warning。旧source-string两条回归被真实删除确认/只读渲染的Green行为断言替换，未削弱用户约束。
- review: 两处重要问题均关闭：手写编辑经按身份的串行事务保留；接受后等待编辑/失败恢复，再恢复focus/selection/scroll。独立真实host复核3/3通过；reviewer认可软件部分。
- real model: 已有授权与根目录身份核对后，对测试库196篇ready（去重189篇、42组/165篇入组），带现有6主题10方向发出真实请求。HTTP200，58.51秒；实际SSE解析→proposer→schema6校验一次通过，topics=[]、coveredPaperKeys=165。24篇聚类外论文仍未被此提案覆盖。输出位于/tmp/arxiv-helm-live；未接受或写入真实vault设置。
- status: P12 blocked只待P6用户真实桌面确认；三个软件chunk已接受。P11 active；已向用户说明提示词升级将使旧筛选缓存失效。
- next: P11详情方向上下文红测，及prompt v3缓存不能复用的红测。仍不提交、不推送。


## 2026-09-07 — P11 完成，进入 P13

- verification: 详情以B实际命中快照评分、无snapshot时读取全部方向，10 Red→45 Green。旧prompt v3缓存1 Red→Green；分类与cache129项、四文件联合174项、core全量及四包typecheck/boundaries通过。prompt snapshot仅新增两条已审阅裁决规则。独立review重跑详情/cache116项，无Important。
- real model: 合成对照而非真实论文日报，实际已配置端点HTTP200/5.95秒；三个预定场景均符合预期（核心相关的具体方向、同具体度按顺序、无关skip），结果/tmp/arxiv-helm-filter/validation.json。没有对整体推荐准确率作推断。
- change: P11 done，P13 active。名额遗漏说明不改变排序、20默认上限、详情policy或per-paper摘要缓存。
- next: P13正常/降级输出与真实pipeline计数接线的Red，之后P6真实日报和桌面查看。仍未提交。


## 2026-09-07 — P13 完成，P6 真实验收

- verification: pipeline/summarizer 4 Red→Green；assembler/rescue 90基线→23 Red/94 Green→117 Green；四文件联合152 Green。全workspace、四包typecheck、boundaries、plugin build通过，lint0 error/20既有warning。独立review重跑152项，无Important。
- change: 普通、emergency、rescue同源展示限额遗漏，ignored先排除；无遗漏输出不变，单篇摘要cache不变。P13 done，P6 active。
- next: /tmp/arxiv-helm-daily脚本已离线准备，真实共享pipeline使用现有方向、默认20、邮件关闭；输出临时vault。真实arXiv API连通性探针HTTP200。随后提供测试构建与用户桌面验收；P12仍blocked于这一明确条件。未提交。


## 2026-09-07 — P6 真实日报及测试库交付，等待用户桌面确认

- verification: 最终全workspace共3075 passed（core2173、node-runtime45、CLI85、plugin772），2项既有skipped；四包typecheck、boundaries、build通过，lint0 error/20既有warning。三阶段独立review的重要问题全部关闭。
- real run: 首次沙箱内Node网络访问失败，按工具规则获准沙箱外重跑；实际CLI composition root/shared pipeline对2026-09-04完成真实arXiv与LLM运行。保留10篇，全部有有效topic-direction标记，命中5条现有library来源方向；每篇均有五项结构化摘要字段。默认20篇、既有主题和方向保持；验收运行关闭邮件。正文与报告先写/tmp，随后复制验收日报。
- delivery: 已获准安装main.js/styles.css和arxiv-daily/daily/2026-09-04.md到/home/tiandc/Desktop/plugin_test，源和目标逐文件SHA-256校验一致；未修改用户主题或凭据。旧构建备份：/tmp/arxiv-helm-before-install-20260907-080704。可复核产物在工作树.artifacts/directions-inside-topics/（忽略目录）；原始请求/凭据只留/tmp私密文件，没有进入Git。
- status: 真实日报成功标准已勾选，P6真实运行与安装两块已接受；用户桌面确认未做，P6保持active、P12保持blocked，goal保持active。下一次继续从用户对复审/设置/日报的反馈开始；不得把本次软件验证冒充用户桌面确认。
- next: 用户重载测试库插件、重新生成提案并查看复审与研究主题设置及真实日报；通过后记录确认、解除P12并关闭P6/goal。个人novelty仍按原Non-goal暂停；不提交、不推送、不开PR。

## 2026-09-07 — P6 日报可读性反馈修正与交付

- evidence: 用户查看2026-09-04日报后要求方向与来源默认折叠、方向不重复主题并逐行列出。整份日报方向/来源占6477字符，五项摘要合计17452字符；本轮是P6局部验收修正，P11/P13的筛选、限额与真实性规则继续有效。
- change: 原生Obsidian callout收起方向、完整来源清单及方向机器标记；方向按条显示，正文列表加入空行，计数明确为“附独立论文总结”。方向解析兼容原裸标记和新引用内标记，仍校验身份、日期、位置及重复。rescue合同新增每篇固定行并同步中英提示词，避免模型重建编码或旧骨架；输出继续逐行校验。
- verification: 输出/解析13 Red→86 Green；固定行传输3 Red→Green；独立审查发现旧来源引用会吞并callout，4 Red复现后以空行分隔并校正规范位置。最终全workspace3088 passed（core2186、node-runtime45、CLI85、plugin772），2项既有skipped；四包typecheck、boundaries、plugin build、diff检查通过；lint0 error/20既有warning。独立复核无未关闭问题。
- desktop: 隔离的真实Obsidian确认默认折叠、点击展开、方向列表、机器标记隐藏、实时预览，以及4种旧来源组合；渲染器无异常。早期桌面脚本误选同一文件隐藏的另一渲染视图，校正选择器后完成上述检查；没有改动桌面验收框架或用户布局。
- delivery: 原日报离线重排，10篇标题/作者/链接/方向/来源与50个摘要字段逐一保持一致，未重跑真实模型。已获准备份并安装测试库日报与main.js，源/目标及旧文件备份SHA-256核对一致。记录与截图：`.artifacts/daily-readability/`；旧文件：`/tmp/arxiv-daily-readability-2cjb3eez/installed-backup-0mxzmck9/`。
- remaining: 用户尚未确认整套复审/设置/日报体验通过，P6与goal保持active、P12保持blocked。额外阅读建议为集中展示3个空主题、按阅读层次压缩长摘要；公式、单位及对象名转写疑点须核对原文，本次保留原文案。仍不提交、不推送、不开PR。

## 2026-09-08 — P6 统一未来日报的阅读规则

- scope: 用户澄清只改善今后生成的日报，CLI和插件必须一致，不修改任何存量日报；并授权落实额外阅读建议。在P6追加Chunk4，保留上一块已有通用折叠，不引入历史迁移或日期特例。
- change: 共享assembler/rescue把有论文的主题保持相对顺序，空主题汇总到末尾并继续区分未匹配与因每日上限未展示；核心结果默认可见，另外四项摘要进入原生abstract折叠，完整五字段与来源仍可回读。parser只展开指定摘要callout，并停止跨H2读字段；rescue传输固定footer且仍逐行严格校验。
- generation: 中英短摘要提示词优先关键结论与必要限定，保留原始单位、天体和数据集标识；prompt contract由1升2，结果schema仍1，未完成任务的旧摘要缓存不再复用。此规则不重写已完成的日报。
- extraction: 真实arXiv HTML的MathML显示树与TeX辅助annotation被一起取textContent，导致公式和对象名重复。共享提取器按DOM结构选取单一数学表示，并覆盖无章节HTML和/abs备用路径；原始HTML缓存保持不变。原文2609.03779明确为峰值通量密度0.75 mJy/beam，模型误译通过单位保留约束处理，未做词语硬替换。
- verification: 共享分组/解析/缓存9 Red、rescue7 Red后相关279 Green；MathML及备用路径分块Red→Green，96项相关回归。CLI与插件实际pipeline各观察布局Red→Green，只替换外部HTTP响应，真实写入/索引路径保留；两文件46项通过。最终全workspace3109 passed/2既有skipped，四包typecheck、boundaries、CLI/plugin build、diff检查通过，lint0 error/20既有warning；独立代码审查无Critical/Important/Minor。
- model / desktop: 3个公开原文真实模型小样通过，中文完整样例主结果147字符、英文45词，单位片段保留0.75 mJy/beam及入选数量；这是有限样例，非整体科学准确率测量。用新生成的中英normal/emergency/rescue共6个排版样例在隔离Obsidian验证：主要结果可见、两类辅助信息默认折叠、方向逐项显示、空主题置后、点击展开与实时预览均通过，渲染器无异常。
- delivery: CLI构建可运行，测试插件main.js已备份安装并校验；本轮只更新程序，26个既有日报文件（含备份）SHA-256均未变。证据与预览在`.artifacts/future-daily-readability/`；旧插件在`/tmp/arxiv-future-daily-ksjfscjy/plugin-backup-on8fc_v4/`。完整P6用户验收仍未记录，P6/goal保持active，P12保持blocked；不提交、不推送、不开PR。

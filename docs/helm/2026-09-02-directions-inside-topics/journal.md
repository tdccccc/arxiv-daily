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

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

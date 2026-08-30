## 2026-08-30 — start

- evidence: PR 40 已合入 main。用户反馈不是检索引擎本身，而是选库 → 嵌入 → 授权 → 索引 → 搜索被拆成五段，建索引只在命令面板。现状：选文件夹后主按钮永远是 Review & authorize；授权后文案写「Metadata and abstracts authorized」；本地嵌入其实不需要授权也能 `indexPersonalLibraryFullText`。
- change: 新建本 goal。P1 只改设置页文献库一行的说明和主按钮。不撤回 PR 40。
- disposition: 检索、打开 PDF、generation 索引保持现状。发现闭环 / email helm 不改。
- next: 先写失败测试：本地模式下选文件夹后应出现 Build index，而不是必须授权。

## 2026-08-30 — P1 done, start P2

- evidence: 设置页文献库行按 embedding.mode 给出下一步。local + 已选文件夹主按钮为 Build index；remote 未授权为 Review & authorize，授权后为 Build index。Revoke 收回 Manage。plugin 设置/连接测试 62 项 + 新增判定 3 项与 typecheck 通过。
- change: P1 done，激活 P2。勾选成功标准 1–5。
- disposition: 命令面板索引入口保留。P2 再处理索引中的进度与「去 Dashboard 搜索」的可见下一步。
- next: P2 让建索引进行中与完成后的下一步可感知。

## 2026-08-30 — P1b compact settings, hold P2

- evidence: 用户反馈 Personal library 选项太多、挤在一起。默认 local 下仍能看到远程 URL/key/dimension 和 sidecar URL；Embedding / PDF parsing 还是独立分组，legacy display() 顺序也和 1.13+ 不一致。
- change: Embedding 与 Better PDF parser 并进 Personal library，放到 LLM 后面。默认三行：Library（主按钮）、Embedding 下拉、Better PDF parser 开关。远程字段仅 remote 时出现，sidecar URL 仅打开后出现。按钮一行右对齐。P2 索引进度改 pending，等用户看过这页再做。
- disposition: 不撤回 PR 40。不改检索、段落证据、generation。发现闭环 / email helm 不改。
- next: 用户在 plugin_test 重载后看设置页；通过再做 P2。

## 2026-08-30 — P1c drop Manage menu

- evidence: Library 行仍是三按钮 + 一个 Manage… 二级菜单（Review directions / Preview library files / Scan library / Reload catalog / Revoke authorization）。用户决策：彻底去掉 Manage，不要二级菜单。
- change: 删除 Manage… 按钮与 `openLibraryManageMenu`；授权状态下 Revoke 变成 Library 行上的普通按钮（非 CTA）。原菜单另外四项移到命令面板：方向复审复用已有 `review-personal-library-directions`；新增 `preview-personal-library-files`、`scan-personal-library`（Scan personal library folder (rebuild catalog)）、`reload-personal-library-catalog`（Reload personal library catalog from disk）。读过 main.ts 后确认 scan 与 reload 不等价——scan 走文件夹重新识别并重写目录（有网络与进度），reload 只把存盘目录读回内存——故注册两条命令而非一条。设置页里已无人调用的 preview / scan / reload 包装方法删掉，plugin 主类方法原样保留。
- disposition: `renderLibraryConnectionControls` 被 declarative 与 legacy display() 共用，只改这一处即覆盖两条路径。不改检索排序、段落证据、generation。发现闭环 / email helm 不动。工作区保持未提交，叠在 P1b 之上。
- next: 用户重载后确认 Library 行只剩目标按钮；若找不到扫描/重载入口，在行下加指向命令面板的说明文字，而不是把 Manage 加回来。

## 2026-08-30 — P1c 收尾：Personal library 整块下移

- evidence: 用户理由是排序问题而非内容问题——文献库是附加功能，不该排在主流程（LLM → 分类 → 主题 → 输出与排程）前面。此前 P1b 把它放在 LLM 之后，等于让附加功能挡在必填项中间。分组顺序在两条渲染路径上各写一份：declarative 走 `definitions.ts` 的 `buildSettingDefinitions` 数组字面量顺序，legacy 走 `tab.ts` `display()` 里 `sectionHeading` 调用的语句先后。P1c 主体那次共用 `renderLibraryConnectionControls` 所以一处即可，这次不共用。
- change: Personal library 整块（Library 行、Embedding 下拉、远程嵌入四字段、Better PDF parser 开关、两个 sidecar 字段）原样搬到 Output & schedule 之后、Email delivery 之前，两条路径分别改。块内行内容、条件显隐、按钮、文案一字未动。
- disposition: 两条路径最终顺序一致：LLM / arXiv / Research topics / Output & schedule / Personal library / Email delivery / Advanced / Help & feedback（legacy 首段标题仍写 "AI model"、分类段仍写 "arXiv"，本次不动措辞）。不改检索排序、段落证据、generation。发现闭环 / email helm 不动。工作区保持未提交。
- next: 用户在 plugin_test 重载后确认文献库落在排程与邮件之间；若仍觉得偏靠前，下一步是折叠为默认收起的区块，而不是继续挪位置。

## 2026-08-30 — P1d 远程授权改为就地确认

- evidence: 用户认为「我都选了这个目录了，还要我授权？」—— 选目录这个动作本身已经表达了同意，Library 行上再放一个 Review & authorize 按钮像是把同一件事问两遍。但把全文发到第三方嵌入接口和本地读文件不是同一件事（ADR 0005 §5 明确把 connection 与 model authorization 分开），所以最终决定是保留授权、只改询问时机（方案 A + A1）。
- change: Library 行的主按钮只要选了文件夹永远是 Build index，Review & authorize 按钮删除；`librarySetupNextStep` 去掉 `authorize` action，改成 `index` + `remoteConsentPending` 标志，按钮层拿不到授权分支，描述文字仍能区分远程未授权/已授权/指纹失效。全部询问收敛到一个 `requestRemoteFullTextConsent()`（granted / declined / undisclosable），由四个时机先到先弹：切到 remote（确认后才落 mode 再写授权，取消一个字节都不改）、选完文件夹（切换时没有文件夹可披露就推迟到这里）、改动已授权接口（指纹失配就重问，取消恢复已授权的值）、remote 未授权时点 Build index（遗留配置兜底，取消中止建索引）。Revoke 走 A1：先确认「撤销授权并切回本地嵌入」并说明已有索引失效需重建，确认后撤销 + mode 回 local，因此不存在「remote 但未授权」的死状态。顺带修了两个披露不实的问题：`libraryAuthorizationDisclosure` 的 depth 原本取自既有 grant，首次远程授权前会把 full-text 披露成 metadata-only，改为取自当前 scope；嵌入接口地址原本复用 chat 的 URL 构造，弹窗里写的是 `.../chat/completions`，而全文实际被 POST 到 `{baseUrl}/embeddings`，改成显示真正的目的地（脱敏逻辑不变，指纹计算不受影响，存量 grant 不失效）。披露文案补上用途（相似度向量）、其它内容不离开本机、可随时撤销。
- disposition: **ADR 0005 / 0008 不改**。授权的四个要素一个没动 —— 仍是全文深度（ADR 0008 §2）、仍披露具体接口地址、仍绑定接口指纹（ADR 0008 §5：改接口即重新询问）、仍可撤销；变的只是「什么时候问」，而 ADR 0005 §5 本来就把「settings 和 UI 怎么表现同意」列为 non-decision，所以这次改动落在 ADR 授权的自由度之内，不是推翻它。接口指纹的定义也保持不变（只含 baseUrl，不含 model）：model 不是目的地，把它塞进指纹会让所有存量 grant 一次性失效；重新询问的判定写成「保存后指纹是否失配」，将来若扩宽指纹定义会自动覆盖 model。不改检索排序、段落证据、generation。发现闭环 / email helm 不动。工作区保持未提交。
- validation: plugin 668 tests 全绿、typecheck、build、boundaries 通过，产物已覆盖到 plugin_test（md5 一致，main.js 晚于所有改动源文件）。全量 `npm test` 另有一条与本次无关的红：`packages/core/tests/pipeline/pipeline-novelty-stage.test.ts` 单跑即 heap OOM，属本分支已提交的 core 测试，本次未碰 packages/。
- next: 用户在 plugin_test 重载后确认：切 Embedding 下拉会弹披露、取消会弹回 local、Library 行只剩 Change folder / Build index / Revoke。若觉得「切下拉就弹框」太突兀，下一步是把披露并到 Build index 一个时机，而不是把授权按钮加回来。

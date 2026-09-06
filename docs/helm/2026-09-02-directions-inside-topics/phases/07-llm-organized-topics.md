# P7 — llm-organized-topics

goal_ref: ../goal.md
created: 2026-09-05T23:09:26+08:00
updated: 2026-09-06T15:03:31+08:00
revision: 4

## Outcome

一次紧聚类之后，由 LLM 组织出 2–4 个主题、每个 1–2 条方向（只有一个证据小组时允许一个主题），复审按论文覆盖量排序，默认勾选前两个主题且折叠显示。

## Assumptions

- 用户已认可删除粗聚类层、逐簇提取与跨簇综合；不把用户现有方向配置作为提示词范例。
- 小组是证据单元，不是方向；每个小组恰好分配一次，方向成员取其所分配小组的完整并集，代表论文只可来自这些小组。
- 主题建议名与方向可以在同一次组织调用中生成，避免为每个主题再发一次请求；输出无效时沿用有界重试。
- 索引、非 arXiv 论文元数据、生成授权、防改动守卫、进度展示和接受写设置的已有修复继续保留。
- 全局紧聚类使用现有 0.95 分位数作为分组参数；未成组论文仍作为可见的未覆盖论文，不强塞进不相关主题。
- 本轮改动保持未提交；P5/P6 与其他 active initiative 不在本阶段实现。

## Approach

将两级聚类与逐簇生成替换为一次聚类、一次全局组织。提示词看到分组的论文标题、有限摘要和成员数量；模型给出主题名、方向文本及小组归属。严格校验归属和引用后，生产端构造完整成员与指纹。契约记录实际阶段、参数和数量界限，使旧提案失效。

## Chunks

### Chunk 1 — 组织契约、生成器和插件接线

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `cd packages/core && ../../node_modules/.bin/vitest run tests/clustered-direction-proposer.test.ts tests/personal-library-proposal-contract.test.ts` 现有 24 项通过；新增组织行为测试应因旧路径逐簇提取、无法识别组织响应而失败。
- Green check: 相同用例验证少量主题、数量边界、小组完整且唯一归属、成员并集、代表来源、取消与重试、不发送文件路径、非 arXiv 证据、参数/提示词契约漂移。
- regression checks: core 全量、plugin 生成生命周期与进度测试、typecheck、boundaries。
- [x] implementation and tests accepted

### Chunk 2 — 复审页聚焦主要主题

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `npm run test --workspace obsidian-arxiv-daily -- personal-library-interest-profile-modal` 新断言在当前没有默认选择、全部展开且未按覆盖量排序的界面失败。
- Green check: 按去重论文覆盖量降序排列、默认仅勾选前两个、其他标可选、折叠摘要显示篇数与方向数；展开后的改名、方向编辑、选择和接受保持有效。
- regression checks: plugin 全量、接受主题的 core 回归、typecheck；保持已有错误、授权与状态栏修复。
- [x] implementation and tests accepted

### Chunk 3 — 集成、构建与真实页面验收

- change kind: non-behavioral / integration verification
- strategy: proportionate verification
- baseline signal: 交接中的旧版真实运行产出 10 主题、46 方向，用户判断过细；历史绿不是新版验收。
- Green check: core / plugin / CLI 测试、typecheck、lint、boundaries；构建插件并安装到既有测试 vault。
- regression checks: 每个行为 chunk 至少一次有针对性的变异核验；使用真实 Obsidian 检查生成粒度、折叠选择、接受后设置内容。
- [x] automated checks and build accepted
- [ ] user desktop acceptance

## Observed checks

- 初始行为红测：两组证据产生 6 次调用，期望 1 次；组织后 proposer 23 项 + decoder 61 项 + fingerprint 5 项通过。成员并集与跨组代表的变异均被对应断言拦住并恢复。
- 复审基线 13 项；排序/默认/折叠/进度红测 9 项失败后转绿。进一步复现方向取消勾选仍写入 settings，7 项红测后修复；最终 modal 28 项，plugin 全量 670 项通过。排序与真实持久化过滤各做过变异验证。
- 生成指纹仅是 opaque hash，加载端原先未对当前生成器校验；因此 proposal schema 4→5，store 对合法同 scope 的 v4 文档返回无当前提案，保留文件直到重新生成成功；损坏/跨 scope 文档仍拒绝。store/review/proposer 38 项通过。
- 薄证据按真实去重成员数量判断，不能把展示的代表论文数量当全部证据；4 红→9 绿，相关 core 回归 105 项与 modal 28 项通过。
- 最终与 P8 一并验证：core **2043 passed / 2 skipped**、plugin **700 passed**、CLI **85 passed**；四包 typecheck、boundaries、lint（0 errors / 20 既有 warnings）通过。
- 只读复核发现短摘要和长摘要混排时，截断标记让 render(0) 并非最小消息，可能误报 evidence-too-large。302 篇边界用例先红；改为独立 abstractTruncated 标记和码点安全前缀后转绿，proposer 24 项。旧共享标记测试同步改守新契约，最终 core 全量重跑通过。
- 插件最终 build 已通过，main.js / styles.css 已安装到 `/home/tiandc/Desktop/plugin_test/.obsidian/plugins/arxiv-daily/`，与构建产物逐字节一致。安装前备份在 `/tmp/arxiv-daily-before-topics-and-cap.l3uAoa/`（临时目录）。真实模型粒度与桌面验收 NOT RUN，不据自动测试把 P7 标 done。
- 后续桌面反馈：用户已于 2026-09-06 09:29 真实生成 schema 5 提案，4 个主题、各 2 条方向。Accept 实际落盘成功，但没有刷新已打开的设置页，也无成功提示；重复点击把 4 个主题追加两次，磁盘现有 10 个主题。新增真实设置页 DOM 回归确认“文件已写、页面仍旧”，再为插件保留设置页实例，接受成功后调用既有 refreshSettings 并显示添加数量。刷新异常不再将已保存动作伪装成失败。
- Accept 修复验证：原有弹窗 28 项基线通过；目标刷新/提示用例先红，修复后弹窗 32 项、插件全量 704 项、插件 typecheck、build 通过，lint 0 errors / 20 既有 warnings。移除刷新调用时新增 DOM 回归确实失败，已恢复。修复版 main.js 已重新安装并 cmp 一致，旧 main.js 备份于 `/tmp/arxiv-daily-before-accept-refresh.nhrexa/main.js`。未修改真实提案或设置数据；方向质量与修复版实际交互仍待用户确认。

## Phase verification

- 历史已提交的画像退休、唯一 tag 与旧日报解析保持可用。
- 暂存测量脚本与两个 zz-probe 调试测试读过后删除，保留两项生成守卫/进度回归。
- 桌面验收需要用户实际查看；若等待用户，则在 goal 中标记本阶段 blocked，继续独立的 P8，不把单测当成桌面验收。

## Abort / reshape triggers

- 组织消息超过既有 60k 代码单位上限时先检查证据选择，不能静默丢组或截掉归属。
- 少量主题仍不能表达用户研究范围时，记录真实输出并重新评估抽象层次，不继续调整聚类比例去控制方向数量。
- 跨主题重复成员、无法验证的代表引用或绕过授权/防改动守卫时，停止接受结果并修复合同。

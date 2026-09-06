# P10 — proposal-sees-existing-topics

goal_ref: ../goal.md
created: 2026-09-06T22:36:44+08:00
updated: 2026-09-06T22:36:44+08:00
revision: 1

## Outcome

在设置里已有主题时重新扫库，提议出来的是**没被覆盖的**主题，以及挂到已有主题下的新方向；不再造出与已有主题范围重叠的新主题。

## Assumptions

- 组织调用一次就能同时判断「这簇属于已有主题」与「这簇要新建主题」；不需要先算相似度再决定。ADR 0014 §3 那条未实测的相似度下限因此**不进入本 phase**，本 phase 的判据是模型读已有主题的文字，不是向量阈值。
- 已有主题的名字与方向文本足够描述其范围。若用户的主题只有一个名字、方向为空，模型能拿到的只有名字——这种输入下重叠仍可能发生，属于已知局限而非本 phase 的失败。
- 提案文档 schema 再升一版不会有兼容负担：库索引与提案文档从未进过发布 tag（goal Constraints）。

## Approach

给组织阶段的输入加一份「研究者已经在跟的主题」（名字 + 各条方向文本），提示词要求：只提未被覆盖的主题；某个证据小组明显是已有主题的延伸时，作为**那个主题下的一条新方向**产出，而不是新主题。提案 schema 因此要能表达「这条方向归属已有主题 X」，复审页据此把它显示成对已有主题的追加而不是新主题，接受时写进那个主题的方向列表。

## Chunks

### Chunk 1 — 组织契约能表达「归属已有主题」

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `packages/core` 新增解码测试——组织响应里一条方向声明归属某个已有主题时，当前解码器不认识该字段而拒绝/丢弃；先红。
- Green check: `cd packages/core && npx vitest run tests/personal-library-topic-organization.test.ts tests/personal-library-proposal-contract.test.ts`
- regression checks: `npm run test --workspace packages/core`；schema 升版后旧版提案的重生成路径仍可用。
- [ ] implementation and tests accepted

### Chunk 2 — 已有主题进入组织提示词

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: proposer 测试断言渲染出的组织消息包含已有主题的名字与方向文本，并断言提示词说明「只提未覆盖的」；当前 proposer 入参没有这份数据，先红。
- Green check: `cd packages/core && npx vitest run tests/clustered-direction-proposer.test.ts`
- regression checks: `npm run test --workspace packages/core`；生成契约 fingerprint 随提示词变更升版，旧提案按既有路径重生成。
- [ ] implementation and tests accepted

### Chunk 3 — 插件把设置里的主题喂给生成，复审页显示「加到已有主题」

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 插件测试断言生成调用带上了 `settings.arxiv.topics`，且复审页把归属已有主题的方向渲染成对该主题的追加；先红。
- Green check: `npm run test --workspace plugin`
- regression checks: `npm run typecheck`、`npm run lint`、`npm run test:workspaces`
- [ ] implementation and tests accepted

### Chunk 4 — 接受时把追加方向写进已有主题

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `acceptProposedTopics` 现在只会新建主题；对「归属已有主题」的方向，测试断言它被追加进那个主题的 directions 且不新建主题，先红。
- Green check: `cd packages/core && npx vitest run tests/accept-proposed-topics.test.ts`
- regression checks: `npm run test:workspaces`；`description` 影子字段仍由 `normalizeTopic` 单一写者维护（ADR 0012 §1）。
- [ ] implementation and tests accepted

## Phase verification

- 在测试库里对着已有的 6 个主题重新生成一次提案：产出中不应再出现与 `Photo-z`、`Galaxy Cluster` 等已有主题范围重叠的新主题；重叠部分应表现为对应主题下的新方向。这项由用户在真实 Obsidian 中确认——复审页从未进过桌面验收，测试绿不构成交付证据（goal Constraints）。
- `npm run test:workspaces`、`npm run typecheck`、`npm run lint`、`npm run check:boundaries` 全绿。

## Abort / reshape triggers

- 若模型拿到已有主题后开始把**所有**新证据都塞进已有主题、不再提新主题（相反方向的失败），停下按 L2 重塑：判据要么回到相似度（等 P5 的实测），要么改成「先分类再组织」的两步。
- 若「归属已有主题」的表达迫使提案文档与 `settings.topics` 双向耦合（提案里存了设置的主题 id，设置一改提案就失效），停下重新设计表达方式——提案文档不该持有设置的主键。

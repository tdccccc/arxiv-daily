# P3 — library-satisfies-the-run

goal_ref: ../goal.md
created: 2026-09-01T20:06:03+08:00
updated: 2026-09-01T20:06:03+08:00
revision: 1

## Outcome

手写主题为 0、已确认方向 ≥ 1 时日报正常运行；两者皆无时仍然拒绝，且说清是哪一头缺（ADR 0010）。

## Assumptions

- **管线不需要改**。过滤是并集，库方向单独命中的论文以 `personal-library` 类别照常进报告；挡路的只有运行前那道配置检查。这一点在立 goal 时已核实。
- **库状态作为显式入参传入检查，而不是从设置里读**。CLI 产品没有文献库，不传即行为一字不变——这是 ADR 0010 §2 的要求，也是本阶段最重要的回归判据。
- **「可用方向」取 eligibility 的结果，不是 profile 里的方向条数**。被禁用、代表论文失踪或证据变了的方向不算数，否则会放行一次注定只能产出空报告的运行。
- 拒绝时的措辞要分两种：完全没有文献库，与「有库但此刻没有可用方向」。后者才是 ADR 0010 §4 点名要说清的那种。

## Approach

`validateFilterConfig` 增加一个可选的库状态入参（是否已连库、当前可用方向数）。主题数为 0 时，只有在可用方向数也为 0 时才判失败，措辞按是否连库分两种。插件侧所有入口（命令面板、Dashboard、上手检查、诊断报告）传入真实库状态；CLI 不传。

## Chunks

### Chunk 1 — 检查接受库状态，任一来源满足即放行

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 新增测试——0 主题 + 1 个可用方向时 `validateFilterConfig` 判定通过；0 主题 + 0 可用方向且已连库时失败且理由点明库这一头；0 主题且不传库状态时失败、理由与今天逐字相同（CLI 回归）。红在第一条（今天无条件要求主题）。
- Green check: `npm run test --workspace @arxiv-daily/core -- validation`
- regression checks: 既有 validation 断言逐条仍绿；`npm run test --workspace @arxiv-daily/core -- diagnostics`；`npm run typecheck`。
- **observed**: 红的原文 `expected false to be true` 与 `expected 'No research topics defined' to match /library/i`。绿之后 validation 44/44、diagnostics 4/4。不传库状态那条断言把今天的措辞逐字钉住，是 CLI 的回归护栏。
- exception: 无
- [x] implementation and tests accepted

### Chunk 2 — 插件各入口传入真实库状态

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- Red / baseline signal: 新增测试——插件在 0 主题 + 有可用方向时，运行日报的命令不再被拦下；0 主题 + 无可用方向时仍被拦且提示点明库。红在被拦。
- Green check: `npm run test --workspace obsidian-arxiv-daily -- commands`
- regression checks: Dashboard 与上手检查的既有断言仍绿；插件全量；`npm run typecheck`。
- **observed**: 红的原文 `expected "spy" to be called once, but got 0 times` 与退回提示不含 library。绿之后 plugin 698/698、CLI 71/71、core 各分片全绿、typecheck 四包、lint 0 error、boundaries OK。
- **入口比计划多两处**：除命令面板与 Dashboard 外，上手检查（`getSetupStatus`，四个调用点）与诊断报告也读这道检查——不一起改，库驱动的用户会被上手向导继续要求先写主题、诊断也会报错误的原因。
- **一处自行做的判断**：库状态取值在宿主调用点写成可选调用。各测试里的假插件本就只实现各自需要的方法，新增一个必需方法会让五个文件、四十条测试变红，而它们要验的东西与库无关；缺席时退回主题-only 规则，正是它们原本的行为。类型仍绑在真实插件方法上，改名照样构建失败。
- exception: 无
- [x] implementation and tests accepted

## Phase verification

- **observed 2026-09-01**：core 各分片全绿；plugin 698/698；CLI 71/71；`npm run typecheck` 四包；`npm run lint` 0 error（20 warning，既有）；`npm run check:boundaries` OK。
- **CLI 回归是本阶段的硬判据**：`apps/cli` 不传库状态，其行为必须逐字不变——由不传参那条断言与 CLI 既有测试共同佐证。
- **不构成交付证据**：日报真的跑出内容。本阶段只证明「闸不再拦」，跑通属 P5。

## Abort / reshape triggers

- 若发现除这道检查外还有别处隐含要求主题（例如排程或报告组装），停下重塑（L2）：ADR 0010 的前提「管线已经支持零主题」即不成立。
- 若把库状态接进诊断报告会牵动其输入契约过多，就把诊断留到单独一块，不要把本阶段撑大。

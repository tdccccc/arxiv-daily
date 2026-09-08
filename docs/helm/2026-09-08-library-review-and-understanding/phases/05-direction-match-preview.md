# P5 — direction-match-preview

goal_ref: ../goal.md
created: 2026-09-08T08:18:29+08:00
updated: 2026-09-08T08:22:37+08:00
revision: 2

## Outcome

用户接受前可用当前方向草稿预览样本文献的匹配结果，看到对应方向及分类覆盖情况，且不改变订阅或提案。

## Assumptions

- 使用既有日报筛选prompt与校验合同，原因以实际命中方向说明，不制造未经过证据核验的解释。
- 默认样本最多20篇，包括提案代表论文与库内其他论文；界面明确样本规模，不声称未来日报数量。
- 本地文件若无arXiv分类，标记分类未知，不猜测分类。
- 预览复用已有文献库处理授权与取消范围；用户点击才调用模型。

## Approach

core纯预览函数接收论文、方向文字、当前分类和LLM端口，调用真实筛选合同后返回匹配和分类差异。插件从当前库索引读取样本，进行授权/身份守卫。复审中预览读取当前草稿，结果随文字更改失效；不保存提案、不写设置。

## Chunks

### Chunk 1 — 筛选预览与分类核对

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- files: core library preview module/test; plugin main controller and review modal/tests。
- Red signal: 当前草稿进入真实筛选请求，命中/未命中可区分，astro-ph不覆盖cs.LG；无分类标未知；未知响应引用拒绝。
- Green checks: 新core preview测试，plugin controller/DOM测试。
- [x] implementation and tests accepted — core接口进入行为后3 Red→3 Green；DOM/实际controller各1Red，分类变更守卫1Red；plugin36项通过，四包typecheck通过。

## Phase verification

- 预览前后settings/proposal逐字一致；授权拒绝/库变化/失败时不给旧结果；core/plugin typecheck。

## Abort / reshape triggers

- 将库内命中数量冒充真实日报量：停止并明确样本含义。
- 预览暗中扩大分类或保存方向：拒绝该副作用。

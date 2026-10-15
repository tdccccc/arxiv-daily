# 文献库变化提醒；个人新颖性经实测后撤回（library novelty and change nudge）

status: done
created: 2026-10-15T10:18:36+08:00
updated: 2026-10-16T01:00:00+08:00
revision: 7
owner: claude-code-main（worktree `../arxiv-daily-library-novelty`，分支 `feat/library-novelty-and-change-nudge`）

## Intent

库里新增论文后研究者会被提醒、只针对新增部分更新方向建议（取代 ADR 0007 的增量归类）。个人新颖性曾按「按需检索最相似库论文」复活并实测（P2），2026-10-15 用户看过真实样例后决定撤回：对照论文由检索挑选而非研究者挑选，无法确认是研究者关心的比较对象，降格为「库中相关论文」也价值不足。撤回时保留对库检索普遍有益的改进。

## Success criteria

- [x] 决策成文：新 ADR 记录 novelty 对比基准与增量更新的取代方案；ADR 0007 标记被取代、ADR 0012 的 novelty 未决项收口；`CONTEXT.md` 的 Personal novelty / Direction suggestion 等条目与新决定一致。
- [x] ~~日报（插件与 CLI 两个宿主）对每篇入选论文，按其标题+摘要按需检索库中最相似的少量论文作为对比基准，生成新颖性说明；说明写入日报并在 Dashboard 论文行显示。~~（P2 交付后经 L3 撤回）
- [x] ~~新颖性只在文献库处理授权有效时运行；未授权、库不可用、模型判定「库中无可比论文」或该步出错时，该篇不写说明，日报照常生成。不引入未经实测的相似度下限。~~（同上）
- [x] 新颖性撤回：日报不再生成新颖性（流水线阶段、检索端口、插件/CLI 接线、授权文案中的日报用途、Dashboard 与工作台显示、相关验收）删除；含旧标记的历史日报仍可正常读取、不报错；ADR 0015 与 CONTEXT 记录撤回及原因；检索提速、查询截词、非 arXiv 文档可检索等对库检索有益的改进保留，Dashboard「找相似论文」若有同样的查询超词问题一并修复。
- [x] 建好索引后，本地（不调模型、无需授权）统计尚未被任何一次扫描计入的库论文（ADR 0015 §3）；达到提醒条件时在插件里提示研究者更新方向建议。
- [x] 研究者点更新后，重扫只把新增论文与现有方向文字发给模型，产出「已覆盖 / 新方向或新主题」建议，沿用现有复审与接受流程。
- [x] 休眠的增量归类代码（向量阈值归入、待定池重聚类、attach/split/merge 建议、建议存储与重复命令）删除；technical-report 等过时文档修正。
- [x] 全量 typecheck、相关测试、boundaries、插件 build 通过；Dashboard / 插件提醒的交互改动经真实 Obsidian 检查（或明确记录未做）。

## Non-goals

- 拆分、合并方向的建议；方向锁定（用户 2026-10-15 定：先不做）。
- 恢复保存代表论文或方向成员列表（会推翻 ADR 0012 §4）。
- 邮件中显示新颖性（沿用 2026-08-03 goal 的决定：邮件不变）。
- 建完索引后自动调用模型（9 月撤回过，会卡顿）；提醒之后的重扫仍由研究者手动触发。
- 修改其他 active helm（directions-inside-topics、email relay v2、桌面验收框架、发现闭环、workbench parity、0.5.0 follow-ups）的状态；只在 directions-inside-topics 的 journal 追加一条「novelty 基准已由用户决定」的说明。

## Constraints

- 用户决定（2026-10-15）：Q1 对比基准=按需检索最相似库论文；Q2 覆盖全部日报论文；Q3 增量=本地提醒+只扫新增；Q4 不做拆分合并。
- 阈值不得拍脑袋定（ADR 0014）：novelty 是否有可比论文由模型判断；提醒条件只用计数，计数值需在 ADR 中写明理由。
- 日报流水线（core）目前不依赖文献库模块；接入要通过宿主注入的端口，保持 `check:boundaries` 通过。
- 日报每日上限默认 20 篇；新颖性每篇至多一次额外模型调用，沿用 `personalized-novelty.ts` 的上界与校验。
- 插件与 CLI 行为一致；Dashboard 视图单测不实例化，交互改动的「测试绿」不构成交付证据。
- 默认使用子代理执行；子代理不得再派生代理、不得提交、要写报告。
- 不推送、不开 PR，除非用户逐次明示。
- Commits: none

## Phases

<!-- Single source of truth for phase status. PN ↔ filename NN. -->
1. P1 — 决策成文：新 ADR、旧 ADR 状态、CONTEXT 术语与新决定一致 — status: done
2. P2 — 日报对每篇论文按最相似库论文生成新颖性说明，授权门控、失败不影响日报 — status: done
3. P3 — 库新增论文的本地提醒与只扫新增的重扫，休眠增量代码与过时文档清理（真实环境检查移至 P5）— status: done
4. P4 — 撤回日报新颖性，保留检索改进，历史日报兼容（Obsidian 固定验收与旧日报真实读取移至 P5）— status: done
5. P5 — 真实环境验证：Obsidian 中提醒与复审按钮、真实模型只扫新增的结果质量、旧日报读取、Obsidian 固定验收（含「只点索引也能发现新 PDF」修正）— status: done

## Open questions

- （已定）提醒条件计数值 3，见 ADR 0015 §3。
- （已定）对比基准至多 5 篇，见 ADR 0015 §1。

# 发现闭环与文献库分析（discovery loop and library insight）

status: active
updated: 2026-08-31
owner: current-session

## Intent

原本要把图书馆引导式发现从"每日展示"闭环成"保存待读 → 按方向回顾 → 阅读反馈回流"，先修三处 P0 质量/安全缺陷，随后补文献库分析。**2026-08-31 收窄**：闭环那一半（P2/P3）经用户判断价值不足，已停做并移除代码；本 goal 余下的意图是文献库分析本身（规模加固与库级概览）。这是 2026-08-03-personal-library-guided-discovery（ADR 0004 步骤 1–6）的延续。

## Success criteria

- [x] 方向生成选题不偏向最旧论文：库内 arXiv 论文按发布日新到旧参与选题与聚类输入，超出上限时优先保留新论文。
- [x] 搜索入口校验 manifest modelId，与当前嵌入模型不一致时给出明确错误与重建指引，不静默错排。
- [x] 单篇损坏/写入失败的论文文档不中止整轮索引，也不中止检索；重索引失败覆盖 ready 记录的策略有文档与测试固化，旧文档不留孤儿。
- ~~用户可将日报中发现的论文保存为待读候选，并按方向回顾做出阅读决策（精读/略读/弃）。~~ — 2026-08-31 停做，代码已移除，理由见 journal。
- ~~阅读反馈（dispositions）落库并回流影响后续发现。~~ — 2026-08-31 停做（依赖上一条）。
- [ ] 文献库分析：库级概览（规模/时间/方向覆盖）可用；检索在数千篇库上内存有界。

## Non-goals

- 引文图/共引分析（沿用 2026-08-05 决策）。
- 全文级 LLM 问答。
- 自主 agent 循环、自动确认方向。
- 检索精度体系的重构（ANN 索引等，除非规模测量证明必要）。

## Constraints

- 沿用既有 consent/授权模型与 store 纪律（CAS、备份恢复、语义重放）。
- 不破坏日报管线与已发布行为；core 保持 host-neutral。
- 测试基线：core 需 8 GiB 堆 + 单 fork（本机环境），plugin/CLI 常规。
- 不提交/推送代码需显式用户指令（沿用 P6 约束）。

## Phases

1. P1 — 修复方向选题偏置、搜索 modelId 校验与索引异常隔离 — status: done
2. P2 — reading candidates：保存待读与按方向回顾 — status: dropped（2026-08-31，代码已移除）
3. P3 — reading dispositions：阅读反馈回流影响发现 — status: dropped（2026-08-31，依赖 P2）
4. P4 — 检索规模加固：内存预算与分页/上限 — status: pending
5. P5 — 文献库分析：库级概览与聚类浏览 — status: pending

## Open questions

- ~~P3：dispositions 回流的具体途径~~ — 随 P3 停做，不再是待答问题。
- P5：分析的首个形态（概览 + 聚类浏览）与既有三个检索入口的关系，在 P5 计划时定。

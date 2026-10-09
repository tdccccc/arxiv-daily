# Product interaction acceptance

status: done
created: 2026-10-09T21:39:29+08:00
updated: 2026-10-09T19:43:07Z
revision: 3
owner: /root

## Intent

用可重复的自动验收替代改版后逐项手点：真实操作 Reading workbench 与 Plugin product，验证功能、交互状态和持久化结果；模型 API 提供有界、无人值守的探索测试。

## Success criteria

- [x] 一条命令创建隔离测试数据、构建当前分支、执行所选宿主的真实界面回归，并输出可复查的报告。
- [x] 固定场景覆盖设置保存、日报生成/阅读、重复启动、取消、失败重试、阅读标记及恢复；功能清单标明每项实际验证层级与未执行项。
- [x] 论文来源、业务 LLM 和邮件使用受控外部响应；测试不会操作用户资料或发送真实邮件。测试使用真实产品业务代码与存储。
- [x] 报告区分 passed、failed、blocked、not-run；记录操作、断言、截图、错误和请求证据。缺失环境或未完成探索不能报告为通过。
- [x] 显式启用后可接模型 API 自动观察界面、选择允许的操作、发现可疑交互；有步骤/时间/调用预算、动作校验与可回放证据。
- [x] 新测试基础设施取得相应 Red/Green 或 Green baseline 证据，真实工作台与 Obsidian 验收取得观测结果，相关原有回归保持通过。
- [x] 使用方法、模型配置和每次改版的执行方式可直接照文档复现。

## Non-goals

- 论文筛选相关性、摘要忠实性与学术内容质量评分。
- 发布、合并主线、调整用户真实配置或操作真实文献库。
- 将测试 agent 引入产品业务流程，或重写现有核心测试。
- 无界面证据地承诺穷尽所有操作组合。

## Constraints

- 分支 test/ui-regression，worktree /home/tiandc/Documents/code/arxiv-daily-ui-regression；保留其他 worktree。
- 测试状态放自有临时目录，证据放 output/playwright/acceptance/；复用旧 desktop harness 的会话模块，但不继承旧目标的测试 vault 路径或修改旧 Helm 状态。
- 依照用户最新选择，模型探索采用 API；普通回归不需要模型密钥。
- 每次交互验证以用户动作和可观测结果为准，不用替换业务成功结果来宣称全流程成功。
- 接受的实现按独立意图提交；不推送。

## Phases

1. P1 — 两端真实界面回归、隔离夹具和有界模型探索具备可运行实现与局部验证 — status: done
2. P2 — 完整验收实跑、缺陷回归、功能覆盖清单和一键使用说明完成 — status: done

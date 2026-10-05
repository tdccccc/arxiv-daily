# Workbench core parity

status: active
owner: /root
created: 2026-10-03T00:22:59+08:00
updated: 2026-10-05T13:21:53+08:00
revision: 35

## Intent

让 DSH 中的独立工作台承接 Obsidian 产品的核心文献流程，Agent 对话仍是辅助入口。用户已授权依次补设置与首次使用、个人文献库、方向审核、定时与运行管理。

## Success criteria

- 设置逐项遵循 Obsidian 1.13+ 主路径的分组、名称、顺序、控件类型、选项和条件显示，不自创分类。

- 无配置也能打开工作台，在图形设置中完成首次配置并运行日报；已有配置可修改，密钥不回显，冲突不覆盖。
- 可连接、授权、扫描、索引与检索个人文献库，复用现有共享业务。
- 可审核并确认方向与增量建议，未经确认不参与个性化发现。
- 可查看并配置定时任务、查看运行结果与执行恢复操作。

## Non-goals

- Claude Mod 实现、Obsidian 专属编辑体验、另一套数据库或模型管线。

## Constraints

- 开发留在当前隔离 worktree；用户已授权 P19 验收后合并本地 main，保留 arxiv-daily 名称。
- 不使用 Computer Use，不修改真实文献、定时任务，不调用付费模型或发送邮件进行测试。
- Markdown 为研究记录；共享 CLI/core 与既有 consent、revision、原子写入机制。

## Phases

1. P1 — 图形设置与首次使用闭环 — status: done
2. P2 — 个人文献库管理与检索 — status: done
3. P3 — 方向与建议审核 — status: done
4. P4 — 定时与运行管理 — status: pending

5. P5 — 按 Obsidian 原设置复刻全部条目与宿主操作 — status: done

6. P6 — 设置导航与区域区分 — status: done

7. P7 — 修复密钥显示与模型选择 — status: done

8. P8 — 单一可输入模型选择框 — status: done

9. P9 — 周末日报无更新状态 — status: done

10. P10 — 简化标题与品牌图标提案供审核 — status: done

11. P11 — 纯文字品牌与统一预览版本 — status: done

12. P12 — 模型获取不重建首次引导 — status: done

13. P13 — 统一界面语言与外观设置 — status: done

14. P14 — 统一业务设置定义、编辑规则与操作服务 — status: done

15. P15 — Core结构化运行状态与各端展示 — status: done

16. P16 — Rebase到最新主线并保留两端功能 — status: done

17. P17 — 修复工作台各阅读入口的Markdown与公式显示 — status: done

18. P18 — 阅读附录/生成统计与前进后退导航 — status: done

19. P19 — 合并前验收、用户文档与本地主线合并 — status: done

20. P20 — 设置自动保存与首次引导不重复出现 — status: active

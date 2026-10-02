# P6 — 当前PR的CodeQL结果检查

goal_ref: ../goal.md
created: 2026-10-02T16:04:14+08:00
updated: 2026-10-02T16:04:14+08:00
revision: 1

## Outcome

核实并修复本次PR结果检查报告的5条CodeQL告警，完成本地回归与提交；再次push取得单次授权后，确认远程检查结果。

## Evidence

用户选择A授权push。origin/fix/review-followups已更新到163f2f63be2a70b54ea0b4d47f3660ec7ecb1604，PR51同SHA。10个required检查成功；CodeQL分析workflow本身成功，但结果检查110752751156失败：4 high正则性能告警、1 medium原型属性写入告警。

## Chunks / strategy

1. daily-paper-summary与arxiv-source-adapter摘要提取：观察公共入口基线与受控长输入耗时，改线性解析，保留摘要/标题约定与CRLF行为。缺陷行为Red→Green或性能基线/同输入比较；不以复制源码regex制造测试。
2. paper-index路径/作者拆分：观察Green兼容基线和性能证据，去除可疑多项式模式，保留路径、schema5和共享锁语义。若路径此前已被上一步归一化限制，明确记录为防御性等价改写而非虚构慢输入Red。
3. settings写入：先观察特殊属性路径污染的Red，事先拒绝__proto__/constructor/prototype等危险段，正常设置仍可写，失败不部分修改；测试清理自己的原型哨兵。
4. targeted回归后完整测试、lint/typecheck/build/boundaries/submission/release-tools/smoke，本地分范围提交。用户未授权第二次push，最后展示具体结果再请求。

## Constraints / reshape triggers

- 不更新PR、不合并、不关闭/抑制远程告警，不改CodeQL配置来隐藏问题。
- 只处理当前结果检查的5条告警；历史分支alerts仍单独存在，不宣称全部历史安全问题清零。
- 当前远程CI成功只对应163f2f6，不能替代后续本地修复提交的远程验收。

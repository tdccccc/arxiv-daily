# P7 — integrated-acceptance

goal_ref: ../goal.md
created: 2026-09-08T08:27:00+08:00
updated: 2026-09-08T08:44:00+08:00
revision: 2

## Outcome

逐项验证本 Helm 的产品结果、自动化回归及真实 Obsidian 交互，明确模型语义检查的覆盖范围。

## Assumptions

- /tmp 独立 vault 用于桌面检查，不修改用户工作库配置。
- 真实模型实验只使用已授权库快照与当前已授权端点，不写用户提案或设置。
- 用户正在并行修改日报摘要，全量失败需按当前证据定位所有权。

## Chunks

### Chunk 1 — 集成与质量检查

- change kind: verification; discovered regressions use Red-Green
- strategy: full workspace tests + typecheck + lint + boundaries + build
- [x] checks accepted — 全workspace3148通过/2既有skip；独立审查修正后plugin全量798通过，最后详情展开修正81项及32项再次通过；core最终相关113通过，混合来源补测review27通过；四包typecheck、最终plugin typecheck、boundaries、build、diff check通过；lint0 errors/20既有warnings。

### Chunk 2 — 真实库组织与桌面

- change kind: verification
- strategy: 冻结真实库首次/已有方向场景；现有Obsidian隔离harness，宽窄截图及实际点击。
- desktop cases: 主题入口、概览零新增、覆盖变化提示、草稿保留/保存接受、样本预览、证据打开。
- [x] observed evidence accepted — 冻结真实库189篇，首次8calls→4主题7方向/165成员/24未覆盖；已有方向8calls→163覆盖+2篇新增/24未覆盖。真实Obsidian13项检查/7截图/0控制台错误，含实际保存接受及PDF打开。证据见 `.artifacts/library-review-and-understanding/README.md`。

## Verification Limits

- 真实模型只验证一个已授权库的首次与已有方向两种情境，不宣称跨学科普适精度；单领域500/1000、混合方向、非arXiv及少量新增使用可重复夹具与真实核心逻辑。
- 预览桌面展示使用受控响应；真实模型筛选契约与只读性由core/plugin集成测试验证。
- 本轮桌面为独立Obsidian自动操作及截图检查，不冒充用户本人确认，也不关闭旧Helm的用户验收项。
- 未提交、未推送、未部署用户工作库；构建位于plugin/main.js，测试vault均在/tmp。

## Phase verification

- 将每个success criterion对应到实际结果；缺失证据保持active。
- 测试日志与截图归档为git-ignored artifacts，阶段记录可复核命令。

## Abort / reshape triggers

- 真实模型反复不能形成合理主题：回到新阶段修正，不把结构验证当语义验收。
- 桌面流程出现数据丢失或不可用：补Red并修复后重验受影响流程。

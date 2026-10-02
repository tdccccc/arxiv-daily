# P5 — 旧资料库升级恢复

goal_ref: ../goal.md
created: 2026-10-02T13:18:45+08:00
updated: 2026-10-02T13:18:45+08:00
revision: 1

## Outcome

用户已有资料库可安全切换嵌入模型、修复清单/论文缓存不一致，并显式重新生成旧版建议，无需手动删除文件；错误状态保持可重试。

## Evidence and assumptions

- 用户13:10手测日志证明当前构建仍有发布阻断，弹窗可打开不能代替核心流程验收。
- 只读实际.index快照确认：一个摘要manifest记录为1chunk，而对应paper文件仍是116个旧全文chunk；远程768维索引与当前本地384维模型冲突；建议主/备份均为严格合法v3。
- 不写真实.index文件，不读取data.json/PDF，不操作其他worktree。软件部署后由用户Retry触发恢复。

## Chunks / strategy

1. core索引：先77项基线；混配reuse/模型切换/新模型覆盖旧文件和快照读取Red→Green。复用前核对文件，模型与维度切换隔离暂存后CAS；保留旧模型与失败暂存，严格builder校验不放宽。
2. core建议：先46项基线；旧v2-v5识别、普通覆盖拒绝、显式归档恢复Red→Green。区分合法退休格式与真损坏，归档主/备份后重新校验；并发出现当前草稿拒绝。
3. plugin恢复入口：模型/维度不符或构建失败显示Retry并禁用Review，保留历史计数；保存模型设置成功后刷新验证；明确点击旧建议Regenerate才允许替换。caller取消与旧可用索引保留；main状态/选项透传回归先Red后Green。
4. 全量测试、lint/typecheck/build/boundaries/submission及release-tools/smoke；按core索引、core旧建议、共用UI衔接分别本地提交，备份部署后等待实际库复测。

## Abort / reshape triggers

- 不把缺少模型、网络失败或无文字PDF当作测试通过；旧schema识别不能吞掉真正损坏文件。
- 不以删除真实索引、覆盖旧草稿或放宽generation校验来消除错误。
- 模型切换过程中取消或最终保存失败不得破坏当前索引；真实恢复结果需用户复测。

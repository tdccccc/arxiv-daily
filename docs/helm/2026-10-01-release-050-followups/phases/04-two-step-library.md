# P4 — 单一资料库入口与两步准备流程

goal_ref: ../goal.md
created: 2026-10-02T12:24:13+08:00
updated: 2026-10-02T12:24:13+08:00
revision: 1

## Outcome

Personal library成为唯一资料库设置区域。选择目录后自动准备标题摘要索引；准备完成后Review suggestions首次生成并审核，已有建议直接打开。

## Assumptions

- 用户本轮明确改变此前“选目录不扫描”的决定；自动准备由设置页用户选目录成功触发，不由插件启动触发。
- Research topics仍编辑已采纳的主题和方向，移除该区重复的Topics from your library准备入口。
- 本地默认，不额外弹本地/远程选择；远程发送前仍确认。模型文件130MB是首次估算，缓存读取不应伪装成下载。

## Chunks and verification

1. 设置两种渲染路径及connection presentation：先300项旧基线，再新入口唯一性、选择成功自动索引、取消/失败/远程拒绝不索引、重试、索引活动禁用审核的Red→Green；保留取消和撤销授权。
2. 审核首次生成：旧modal基线90项，再无proposal自动授权/生成、已有草稿不覆盖、授权期间关闭/重开不复活、关闭运行取消的Red→Green；main透传只绑定本次生成的AbortSignal。
3. 模型准备进度：旧12项基线，新增回调顺序、缓存语义、取消后停止通知Red→Green；main接入活动行进度并恢复论文计数。
4. 完整plugin及根检查、lint/typecheck/build/boundaries/submission与相关release工具；独立本地提交、备份部署；真实Obsidian与实际库耗时仍由用户手测。

## Abort / reshape triggers

- 不改变外部发送授权、主题存储格式或来源工作树。
- 本轮不发布，不同步版本；发行草稿只改正确流程，留后续版本提交。

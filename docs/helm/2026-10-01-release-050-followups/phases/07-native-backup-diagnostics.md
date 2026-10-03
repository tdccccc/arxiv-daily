# P7 — 定位macOS原生备份内容异常

goal_ref: ../goal.md
created: 2026-10-02T22:57:28+08:00
updated: 2026-10-02T23:37:11+08:00
revision: 2

## Outcome

为真实CI中一次备份内容异常补充有界、逐阶段、可下载的诊断证据；本地准备后单独申请诊断分支push与Draft PR授权。没有根因或修复证据前不称异常已解决。

## Evidence

用户授权PR51普通merge到main，提交c620caeda05af5bfa5c4236f935b6b30d75fb8b6，树与已验ace7745一致。PR43随之标为merged。main native run37012308927第一次darwin-arm64在原子替换测试读到backup三个NUL而非old，新primary为new。9/10通过，资产组装跳过。失败TAP已下载保留。

同SHA失败任务复测（attempt2）全部成功，组装成功；复测通过没有解释第一次异常。Linux10项基线和四模式各100次诊断均通过，不能作为macOS证明。

## Approach / verification

- 不修改C++，不添加sleep/放宽断言/F_FULLFSYNC等猜测修复。
- 先保持原流程的无中间读取对照，再收集write/close/link/重复link/rename/sync阶段的固定测试字节与inode/nlink；Node syscall对照仅在支持的POSIX平台运行。
- 每模式有界100次，首错停止该模式、非零退出，继续其它模式收集对照；只用自身mkdtemp并在清理前记录快照。
- CI基础native测试即使失败也运行诊断（构建成功且未取消），JSON以always artifact上传；诊断失败仍阻断，不抑制原测试。
- 新诊断工具是观测能力，不是生产修复；以Linux原流程Green、实际诊断执行和可控NUL注入验证，明确macOS真实错误尚未复现于新诊断。
- 单独本地分支fix/native-backup-diagnostics从合并提交创建，不重置main或修改任何现有worktree。

## Gate

推送诊断分支、创建PR、合并诊断、tag与正式发布仍分别依用户明确授权执行。本轮不发布0.5.0。

每平台一轮有界对照后决定下一步：复现则按最早差异定位；未复现则记录未知原因并让用户决定继续调查或发布，不无限重跑或宣称未知问题已修复。

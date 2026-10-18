# 运行状态写入在两次 rename 之间崩溃后，下一次更新会清空全部历史

> 发现日期：2026-10-18 · 来源：main `21fe2ce` 深度代码审查 · 编号 CORE-02
> 范围：`run-state.json`（插件、CLI） · 严重程度：低 · 状态：已修复（分支 `fix/2026-10-18-review`，提交 366025c） · 置信度：已复现

## 问题

`writeAtomic()` 先把 `run-state.json` 改名为 `.bak`，再把临时文件改名为 `run-state.json`。如果进程或系统恰好在两次改名之间崩溃，主文件不存在，`.bak` 保存着最后的正确状态。

`StateStore.load()` 会正确回退读取 `.bak`，所以界面一开始看到的历史是完整的。但每次状态变更（`setRunning`、`setCompleted` 等）都走 `loadAuthoritativeRunState`，它在主文件缺失时直接返回空状态、不看 `.bak`。第一次变更就会用空状态写盘，永久丢掉全部历史：永久失败标记和重试计数没了（这些日期会重新跑 LLM），面板日历用的完成日期和写入篇数也消失。

主文件损坏时代码有意"拒绝写入"（有测试断言），主文件缺失时却直接当空，这种不对称看起来是疏忽。

## 原因

- `packages/core/src/services/state-store.ts:374-380`：`loadAuthoritativeRunState` 主文件缺失时返回 `{}`，不检查 `${path}.bak`；而 `loadRunStateWithFallback`（同文件 382-424 行）会检查。
- `state-store.ts:557-581` `writeAtomic` 存在只有 `.bak` 的窗口，`catch` 里的恢复只覆盖进程内异常，覆盖不了崩溃。

## 复现

审查时用临时脚本模拟"只剩 `.bak`"的状态，调用一次 `setRunning` 后 `.bak` 中的历史全部丢失。

## 可能的解决方案

推荐：`loadAuthoritativeRunState` 在主文件缺失时，若 `.bak` 严格校验通过就返回 `.bak` 内容（随后的保存会重建主文件）；`.bak` 存在但无效时抛错（保持拒绝写入），不要返回 `{}`。

备选：`load()` 发现主文件缺失时把 `.bak` 提升回主文件。Node 宿主若有真正的原子覆盖写，可彻底消除窗口。

## 修复后应补的测试

- 只有有效 `.bak` 时，一次变更后历史完整保留。
- 只有损坏 `.bak` 时，变更被拒绝而不是清空。

## 修复情况

主文件缺失时，状态变更基于严格校验通过的 `.bak`；`.bak` 无效时抛错，不再清空历史。

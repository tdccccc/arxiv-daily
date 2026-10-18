# `arxiv-daily update` 在 Windows 上无法安装，且把预发布版本当成正式版

> 发现日期：2026-10-18 · 来源：main `21fe2ce` 深度代码审查 · 编号 CLI-5
> 范围：CLI `update` · 严重程度：低 · 状态：已修复（分支 `fix/2026-10-18-review`，提交 d34d154） · 置信度：代码推断

## 问题

- Windows 是受支持的 CLI 平台，但 `update` 执行 `spawn('npm', ['install','-g',spec], {shell:false})`。Windows 上 npm 是 `npm.cmd`，不经 shell 时 Node 报 ENOENT（显式 `.cmd` 自 2024 年安全修复后报 EINVAL），命令总是以"Update failed"结束。
- `compareSemver` 对每段用 `parseInt`，`0.7.0-rc.1` 与 `0.7.0` 比较相等（显示"Already up to date"）。
- 从 npm registry 拿到的 `latest` 字符串未校验为 SemVer 就拼进 npm 安装参数。

## 原因

- `apps/cli/src/update-cmd.ts:129`：`shell:false` 且用裸 `npm`。
- `apps/cli/src/version.ts:21` `compareSemver` 忽略预发布标签；`fetchLatestFromNpm` 只检查 `typeof string`。

## 可能的解决方案

推荐：
1. 先用严格正则校验 `latest`（`MAJOR.MINOR.PATCH[-prerelease]`），不合格直接报错。
2. win32 上以 `npm.cmd` + `shell:true` 执行（参数已校验，无注入风险）。
3. `compareSemver` 支持预发布：同版本号时，有预发布标签者较小，预发布标识按 SemVer 规则比较。

## 修复后应补的测试

- `compareSemver('0.7.0-rc.1','0.7.0') < 0`，以及数字/字母预发布标识的比较。
- 非法 `latest`（含空格、shell 元字符）被拒绝。
- win32 下 spawn 参数为 `npm.cmd` 且 `shell:true`（模拟 platform）。

## 修复情况

只接受严格 SemVer 的 registry 版本；win32 上经 shell 运行 `npm.cmd`（版本已先校验）；`compareSemver` 支持预发布版本。

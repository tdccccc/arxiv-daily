# 边界检查脚本对 plugin/src 之外的插件文件静默放行（Node 内置模块规则按路径子串判断）

> 发现日期：2026-10-18 · 来源：main `21fe2ce` 深度代码审查 · 编号 CLI-8
> 范围：`scripts/check-boundaries.mjs` · 严重程度：低 · 状态：已修复（分支 `fix/2026-10-18-review`，提交 7316e5f） · 置信度：已复现

## 问题

"插件导入 Node 内置模块"规则（以及几条 core 规则）只在路径包含 `/src/` 时生效。`plugin/main.ts` 和 `plugin/node-fs-exclusive.ts`（导入了 `node:fs`、`node:path`）都在 `plugin/src` 之外，所以在 `plugin/main.ts` 或任何新的插件根目录文件里加入 Node 内置模块导入，`check:boundaries` 都会通过，悄悄破坏插件的移动端兼容边界。现有的 `node:fs` 例外是隐式的，而不是明确的白名单。

实测：在一份脚本副本中给 `plugin/main.ts` 加入 `node:fs` 导入，检查仍通过。

## 原因

- `scripts/check-boundaries.mjs:53-56`：插件内置模块规则以 `file.includes("/src/")` 为前提。
- 在 Windows 上路径用反斜杠，同样的检查会全部被跳过。

## 可能的解决方案

推荐：插件规则作用于所有 `plugin/` 下的生产 TypeScript（排除 tests、esbuild 配置和构建产物），配一个显式白名单（`node-fs-exclusive.ts`，附注释说明原因）；子串判断前把路径分隔符统一为 `/`。

## 修复后应补的测试

- `scripts/tests/` 中为边界检查增加用例：`plugin/main.ts` 导入 `node:fs` 被报告；白名单文件不报告；Windows 风格路径也能被检查。

## 修复情况

插件 Node 内置模块规则覆盖所有插件生产源码，`plugin/node-fs-exclusive.ts` 按路径白名单只允许 `node:fs`、`node:fs/promises`、`node:path`；路径统一为 `/`；检查函数导出并有夹具测试。

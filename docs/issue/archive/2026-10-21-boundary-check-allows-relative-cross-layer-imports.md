# 分层边界检查只认包名，用相对路径跨层导入时一律放行

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 SCR-03
> 范围：`scripts/check-boundaries.mjs`（`npm run check:boundaries`，CI 的 Root verification 和发版工作流都会跑） · 严重程度：低 · 状态：已修复（分支 `fix/2026-10-21-batch4`，提交 67e27e6） · 置信度：已复现

## 问题

`check:boundaries` 的作用是守住 `core ← node-runtime ← apps/cli`、插件只能用 core 和三个指定的 node-runtime 子路径、core 里不能出现 Node 内置模块和第三方依赖这些分层规则。但这些规则只对“包名形式”的导入生效；只要把同一个依赖写成相对路径，检查就全部放行：

| 夹具中的导入（真实的 `checkBoundaries()`） | 结果 |
|---|---|
| core 里 `import … from "@arxiv-daily/node-runtime"` | 报错：`forbidden dependency` |
| 插件里 `import … from "@arxiv-daily/node-runtime"` | 报错：`broad node-runtime import is forbidden` |
| 插件里 `import … from "@arxiv-daily/node-runtime/src/http-client"` | 报错：`deep workspace import` |
| core 里 `import … from "../../node-runtime/src/http-client"` | **无报错** |
| 插件里 `import … from "../../packages/node-runtime/src/http-client"`（被导入文件含 `node:fs`） | **无报错** |
| core 里 `import … from "../../../apps/cli/src/config"` | **无报错** |
| node-runtime 里 `import … from "../../../apps/cli/src/config"` | **无报错** |

esbuild 和 tsc 对这些相对路径都能正常解析，所以构建不会拦。后果是：任何一次（人写的或 AI 助手写的）顺手的相对路径导入，都能把 Node 专用代码带进 core 或插件包，边界检查仍然显示 “Workspace boundaries OK”。目前仓库里没有这样的违规导入（除了 `node-runtime` 读取 `tools/node-library-runtime/**` 的 JSON，那是有意为之），所以这是一个守卫缺口，不是现存缺陷。插件包会被 `smoke-build.mjs` 对 `require()` 说明符的白名单兜住一部分，core 被拉进 Node 代码则没有任何兜底。

## 原因

- `scripts/check-boundaries.mjs:71`：`if (specifier.startsWith(".") || specifier.startsWith("/")) continue;`——所有相对路径和绝对路径的导入在分层判断之前就被跳过了，之后的 `packageOwner`、`layer.workspace`、`workspaceSubpaths`、内置模块和第三方白名单规则都只看包名。
- 没有任何一步把相对路径解析成仓库内的真实文件，再判断它落在哪一层。

## 可能的解决方案

推荐：对相对导入做解析——用导入方文件所在目录把说明符解析成仓库内的绝对路径，算出它属于哪一层（`packages/core`、`packages/node-runtime`、`apps/cli`、`plugin`，或层外），再套用同一张“谁可以依赖谁”的表：跨层的相对导入一律报错（例外只写进显式白名单，如 `packages/node-runtime/src/library-runtime.ts` 读取 `tools/node-library-runtime/**/*.json`）。同层内的相对导入照旧放行。

备选：只禁止相对路径里出现 `/packages/`、`/apps/`、`/plugin/` 这类跨层目录名。实现简单，但对深层 `../../..` 的写法容易漏判。

## 修复后应补的测试

- 在 `scripts/tests/check-boundaries.test.mjs` 增加上表中四种相对路径写法的用例，断言每一种都被报告；同一层内的相对导入、白名单里的 `tools/` 导入不报告。
- Windows 风格路径（反斜杠）下相对导入的解析同样被检查。

## 修复情况

跨层相对路径、绝对路径和 Windows 反斜杠写法都纳入边界检查；同层导入保持可用，仅对运行时 JSON、构建配置及既有集成测试保留精确例外。7 项失败回归修复后，边界测试 17 项与仓库实际边界检查通过。

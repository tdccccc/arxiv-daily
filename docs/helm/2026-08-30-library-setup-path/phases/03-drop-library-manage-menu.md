# P1c — drop-library-manage-menu

goal_ref: ../goal.md
updated: 2026-08-30

## Outcome

Personal library 的 Library 行最多三个按钮、永远一行：未选文件夹只有 Choose folder；已选 + local 是 Change folder + Build index；已选 + remote 未授权是 Change folder + Review & authorize；已授权是 Change folder + Build index + Revoke。Manage… 按钮和它的五项菜单整个删掉。

## Assumptions

- 用户决策是「彻底去掉 Manage」，不是把它改小或换图标。
- 二级动作（预览文件 / 扫描 / 重载目录 / 方向复审）留给命令面板即可，设置页不再是它们的入口。
- Revoke 是低频破坏性动作，做普通按钮、不做 CTA，避免和主按钮抢注意力。
- declarative（1.13+）与 legacy display() 共用 `renderLibraryConnectionControls`，一处改动两条路径同时生效。

## Approach

`renderLibraryConnectionControls` 里删掉 Manage… 按钮与 `openLibraryManageMenu`，仅在 `status.kind === "authorized"` 时追加一个 "Revoke" 普通按钮，仍走 `runAction("revoke personal library", …)`。设置页原来包装 preview / scan / reload 的三个 tab 方法一并删除（plugin 主类方法保留），改由 `commands.ts` 直接调用 plugin 方法并弹原来的 modal。三条新命令的错误处理照 commands.ts 既有 library 命令：先 `getLibraryConnectionStatus()` 挡住未选文件夹并给可读 notice，再 try/catch + `logger.error` + `errorMessage`。

scan 与 reload 语义不同，注册为两条独立命令：scan 走文件夹、重新识别文件、重写目录（有网络与进度）；reload 只把已存盘的目录读回内存，不碰文件夹。

## Test strategy

- change kind: behavior change（设置页按钮集合 + 命令面板入口）
- strategy: strict Red–Green on settings-declarative-tab + commands
- Red / baseline signal: 旧测试锁死三种状态下都有 "Manage…"，并断言菜单五项标题；新命令 id 查不到
- Green / regression checks: settings-declarative-tab、settings-tab、settings-definitions、library-connection-lifecycle、commands、personal-library-scan-lifecycle、plugin 全量 + typecheck + build

## Tasks

- [x] 删除 Manage… 按钮与 `openLibraryManageMenu`。
- [x] 授权状态下渲染直接的 Revoke 按钮（非 CTA）。
- [x] 新增 `preview-personal-library-files`、`scan-personal-library`、`reload-personal-library-catalog` 三条命令。
- [x] 未选文件夹时给可读 notice，不抛未捕获异常。
- [x] 删除设置页里已无人调用的 preview / scan / reload 包装方法（plugin 主类方法不动）。
- [x] 收尾：Personal library 整块从 LLM 之后移到 Output & schedule 之后、Email delivery 之前（附加功能不排在主流程前）。declarative 与 legacy 两条路径各改一处。

## Verification

- 三种状态按钮文案与数量符合目标形态，且任何状态都不出现 "Manage…"。
- 未授权状态没有 Revoke；授权状态 Revoke 存在、`cta === false`、点击路由到 `revoke personal library`。
- 三条新命令 id/name 已注册；scan 只调 `scanPersonalLibrary`，reload 只调 `reloadPersonalLibraryCatalog`。
- 断开状态下三条命令都只发 "choose a personal library folder in settings first."，不调用任何 plugin 方法。
- 分组顺序在两条路径上一致：LLM / arXiv / Research topics / Output & schedule / Personal library / Email delivery / Advanced / Help & feedback；块内各行内容与显隐条件不变。
- Observed Green (2026-08-30): plugin 全量 642 tests 通过（其中 commands 37 / settings-declarative-tab 53），typecheck 与 production build 通过。
- Observed Green (2026-08-30, 分组下移后): plugin 全量 644 tests 通过（settings-tab 55 / settings-declarative-tab 54），typecheck 与 production build 通过。

## Abort / reshape triggers

- 如果用户重载后找不到扫描/重载入口，先在 Library 行下加一行说明文字指向命令面板，而不是把 Manage 加回来。
- 如果 Revoke 被误点导致远程授权反复丢失，给它加一次确认，而不是收回菜单。

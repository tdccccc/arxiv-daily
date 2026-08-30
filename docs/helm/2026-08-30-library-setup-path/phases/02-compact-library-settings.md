# P1b — compact-library-settings

goal_ref: ../goal.md
updated: 2026-08-30

## Outcome

Personal library 设置默认只露出三行：选库/建索引、Embedding 下拉、Better PDF parser 开关。远程 URL/key 和 sidecar URL 按需展开。按钮在一行里右对齐，不再挤成一团。

## Assumptions

- 本地默认不需要看到远程嵌入字段或 sidecar URL。
- declarative（1.13+）和 legacy display()（&lt;1.13）必须同一套分组与显隐。
- 索引进度与「去搜索」仍留给后面的 P2，本阶段不改通知或 Dashboard。

## Approach

把 Embedding 与 PDF parsing 并进 Personal library，放到 LLM 后面。`embedding.mode === "remote"` 才显示远程字段；sidecar 打开才显示 URL。切换 mode / sidecar 后刷新设置页。文献库按钮 `flex-wrap: nowrap`、右对齐。

## Test strategy

- change kind: behavior change（设置呈现）
- strategy: strict Red–Green–Refactor on definitions + declarative tab + CSS
- Red / baseline signal: 旧测试锁死独立 Embedding / PDF parsing heading，以及 local 模式下仍列出远程字段名
- Green / regression checks: settings-definitions、settings-declarative-tab、library-connection-lifecycle、settings-tab、plugin typecheck

## Tasks

- [x] 默认只渲染 Library / Embedding / Better PDF parser。
- [x] remote 与 sidecar 打开时展开对应字段。
- [x] legacy display() 与 declarative 同一顺序和显隐。
- [x] 文献库按钮一行右对齐、不换行。

## Verification

- 默认定义不含 Embedding / PDF parsing heading，也不含远程 URL 字段名。
- remote + sidecar 打开后九行都在。
- CSS 锁死 `flex-wrap: nowrap` 与 `justify-content: flex-end`。
- Observed Green (2026-08-30): plugin 137 tests（settings-definitions / settings-declarative-tab / library-connection-lifecycle / settings-tab）与 typecheck 通过。

## Abort / reshape triggers

- 如果隐藏远程字段导致无法完成远程授权，停下来先保证授权入口仍从文献库主按钮进入。
- 如果按钮在窄设置栏里被裁切到看不见主 CTA，改回允许换行，但主按钮仍须可见。

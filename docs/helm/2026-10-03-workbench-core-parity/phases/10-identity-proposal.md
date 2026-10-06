# P10 — Header and identity proposal

goal_ref: ../goal.md
created: 2026-10-04T01:33:54+08:00
updated: 2026-10-04T01:33:54+08:00
revision: 1

## Outcome

左上角源码仅显示加大加粗的arxiv daily；提供独立SVG标识设计稿供用户审核，不部署未确认图标。

## Scope and constraints

用户明确要求先看图标并决定是否使用。新标识仅在docs/design/arxiv-daily-identity；不改DSH图标、不接入header、不新增发布包。此前模型和周末修复继续等统一版本。

## Evidence

- Header behavior change: observed old a↗/arXiv Daily/阅读工作台 Red, then 21 UI tests pass; CLI typecheck/diff check pass. cb5d3c0 holds the accepted header change.
- Vector design: hand-authoredSVG主标识、深色、单色和app tile，ImageMagick高分辨率渲染、Pillow合成preview.png，并实际查看预览。没有Computer Use、无生成式位图或第三方字体分发。
- Design thesis: 深青色几何小写a + 暖色每日更新标点，圆润、克制，适配小尺寸和单色。
- Logo adoption awaits user feedback; only preparation is accepted here. A later approval authorizes integration separately.

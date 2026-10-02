# P1 — 收尾修复并部署手测

goal_ref: ../goal.md
created: 2026-10-01T23:06:37+08:00
updated: 2026-10-01T23:20:17+08:00
revision: 2

## Outcome

首次索引与审核窗口修复分别提交，构建通过要求的检查后部署到指定手测目录。

## Assumptions

- 现有未提交实现是需要续接的半成品，不回退源码来制造历史 Red。
- PDF 可读性与 arXiv 识别结果是两个独立条件。

## Chunks

### Chunk 1 — 首次索引按需扫描与准确结果说明

- change kind: bug fix
- strategy: 现有半成品运行接手基线；新增缺口严格 Red → Green。
- baseline: plugin 六个针对测试；初次错误根目录 Vitest 命令的加载错误不算行为 Red。
- expected Red: arXiv metadata-fetch-failed 的可读 PDF 仍可索引；空库引导显式重扫；失败索引不宣称已可搜索。
- Green / regression: plugin 针对测试与 core 索引编排测试，随后完整检查。
- exception: 接手前的 Red 未观察，不声称历史 TDD；对保留实现以接手基线和回归补偿。
- [x] implementation and tests accepted — 9e4bcb2；接手基线105通过，新增回归Red→Green，最终针对110通过。

### Chunk 2 — 审核窗口尺寸与滚动

- change kind: low-impact layout fix
- strategy: 比例验证，现有审核对话框交互回归；不添加复制 CSS 的测试。
- baseline: 现有宽度只作用于内容区，外壳仍受宿主默认宽度限制。
- Green / regression: 实际浏览器几何检查（若环境可用）、现有 modal 交互测试，完整检查；真实 Obsidian 由用户重启后手测。
- [x] implementation and tests accepted — 0df3f95；24项交互测试前后通过，样式审查通过；真实Obsidian布局仍待手测。

## Phase verification

- NODE_OPTIONS=--max-old-space-size=8192 npm test
- npm run lint（历史 0 errors / 20 warnings；报告实测）
- npm run typecheck / build / check:boundaries / check:obsidian-submission
- 逐个检查暂存范围并分别提交修复，草稿不纳入。
- 先查看目标，独占创建不覆盖旧文件的备份，复制 main.js / styles.css / manifest.json；记录校验结果。
- 实际结果见 ../verification.md。上述全量检查通过；初次沙箱测试失败如实保留，授权沙箱外重跑成功。

## Abort / reshape triggers

- 新缺陷超出两项修复或需要改变既定行为时先重新定范围。
- 手测未通过时不进入版本同步；CLI 是否纳入发布依用户明确决定。

# P4 — library-topic-entry

goal_ref: ../goal.md
created: 2026-09-08T08:15:55+08:00
updated: 2026-09-08T08:18:29+08:00
revision: 2

## Outcome

研究主题页直接提供从文献库生成/复审入口，用户不需要寻找命令面板。

## Assumptions

- 复用选库、嵌入方式选择、索引与授权；不增加远程处理授权面。
- 索引完成后打开复审，由用户点击生成并使用已有授权流程。
- 两种 Obsidian 设置渲染入口具有相同行为。

## Approach

主题列表前增加一个明确命令入口，按当前库/索引状态进入选库、索引或复审；取消与失败不继续。库设置原有三个按钮限制保留，入口位于主题区域。

## Chunks

### Chunk 1 — 双设置入口与状态引导

- change kind: behavior change
- strategy: strict Red-Green-Refactor
- files: plugin settings tab/definitions；declarative/legacy tests。
- Red signal: 主题页能找到入口；已建索引直接复审；未连接先选库，取消不打开；索引失败不打开。
- Green check: settings-declarative-tab、settings-tab tests；plugin typecheck。
- [x] implementation and tests accepted — 2入口Red→Green；新版/旧版设置174项通过，含取消、索引失败/重试、旧版DOM点击；plugin typecheck通过。

## Phase verification

- 真实 DOM 点击入口与状态分支；最终 P7 统一桌面查看。

## Abort / reshape triggers

- 新入口绕开授权或默认调用远程模型：停止并复用既有授权流程。

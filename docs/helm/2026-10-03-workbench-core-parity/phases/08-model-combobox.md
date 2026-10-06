# P8 — Model combobox

goal_ref: ../goal.md
created: 2026-10-04T01:13:26+08:00
updated: 2026-10-04T01:15:50+08:00
revision: 2

## Outcome

Get models左侧只有一个可输入模型框，加载选项后从该框展开下拉，选择仍写入同一框。

## Assumptions

- 用户截图纠正P7双控件呈现，保留密钥修复与模型获取接口。
- 支持自由输入、筛选、鼠标与键盘选择；不自动替换当前值。

## Chunks

- bug fix；strict Red→Green：组件测试单一输入、加载展开、选择回填、过滤、键盘和关闭；旧UI集成测试改为单框契约。
- CSS：弹层依附输入框，Get models保持右侧，比例验证构建和结构，不使用Computer Use。

## Verification

- 新组件+settings UI/navigation/HTTP回归、typecheck、DSH组件、build/pack。

## Abort / reshape triggers

- 不得重新引入第二个select或放弃手工输入。

## Acceptance

- d753172：两条组件Red及四条UI契约Red后通过；同一个模型输入支持加载展开、过滤、鼠标/键盘选择、Escape/失焦关闭及自定义值保留。
- 25 focused UI tests、12 DSH component/registry checks通过；typecheck/boundaries/inventory/build/pack通过。仅查看用户提供截图，未进行浏览器视觉自动化。
- 0.1.8 linux/x64本地包可安装。后端与密钥逻辑未改，不重复全量Host测试。

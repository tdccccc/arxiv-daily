# Journal

## 2026-10-03 — Scope

用户在 DSH 固定入口验收后明确授权补齐核心流程。此前 DSH 集成的 GUI onboarding non-goal 仅限旧集成阶段；本 initiative 承接新增业务界面范围，不重开已验收阶段。其他 active initiatives 属于既有 Obsidian/relay 工作，不接管其状态。

## 2026-10-03 — P1 acceptance and P2 boundary

P1 on track：无配置打开、图形编辑保存、立即运行日报已打通。用户追加“仿照 Obsidian settings”后采用左侧分组导航、右侧字段，首次流程复用此页。现有文献库、邮件、嵌入和定时字段不丢失。真实 DOM→配置服务测试暴露 topic id 契约遗漏，已先复现后修复并稳定保留主题身份。

验收证据见P1。0.1.4本地产物可安装；没有自动更新用户当前DSH。后续P2复用CLI文献库工作流及既有授权披露，不复制pipeline；P3方向审核与P4定时管理未实现，整体目标保持active。

## 2026-10-03T12:53:53+08:00 — L3 settings parity

用户拒绝自创“每日发现”等分组，并明确下拉/开关/选项均依照Obsidian。新增P5先补设置复刻，P2暂回pending。P1首次启动与保存证据保留；自定义设置布局由P5替换。基准为1.13+主路径，legacy仅对照不混入。

## 2026-10-03T13:24:49+08:00 — P5 accepted

On track: replaced the custom settings design with the Obsidian 1.13+ controls and section order. P5 evidence and host adaptations are recorded in its phase. New 0.1.5 archive is ready, not automatically installed in the user's running DSH. Restart is required after upgrade.

Return focus to P2: connection, consent and build controls are now available in Settings through P5. Remaining library product work is the dedicated catalog/search/review experience; P3 direction review and P4 richer run management are still pending. Do not redo the accepted settings layer or claim full application parity.

## 2026-10-03T16:37:51+08:00 — P6 navigation

用户确认基本复刻完成，要求左侧导航点击跳转和更明显的区域区分。P5继续done，新增P6视觉/导航增量，暂缓P2。保留全部原设置字段和顺序。

## 2026-10-03T16:40:24+08:00 — P6 accepted

On track: left jump navigation and stronger settings sections added without changing original controls. Verified 34 UI +12 DSH checks and built 0.1.6. Existing P5 behavior preserved; resume P2 library catalog/search next.

## 2026-10-03T16:49:17+08:00 — P7 fix

用户报告Show与Get models不可用。根因：保存后空密钥框仅切type；datalist没有明确展开入口。保留P5/P6，新增P7。用户明确请求密钥显示，修订此前仅显示新输入的宿主限制，采用显式受保护POST显示已存密钥。

## 2026-10-03T17:03:11+08:00 — P7 accepted

On track: saved-key Show/Hide and explicit model selector work in regression tests; DSH0.1.7 built. This supersedes P5's new-input-only reveal limitation in response to the user request. P2 remains the next product phase.

## 2026-10-04T01:13:26+08:00 — P8 model control

用户截图要求模型选择合并到Get models左侧原框。P7显示密钥和获取接口保留，单独select由P8替换，采用可输入combobox。

## 2026-10-04T01:15:50+08:00 — P8 accepted

单模型combobox修复已验收：保留原值，Get models成功直接展开框下选项，移除P7第二select。0.1.8已打包，未修改用户安装。后续继续P2。

## 2026-10-04T01:27:09+08:00 — P9 review fix

用户报告周末日期请求显示失败。确认自动调度已有周末guard，而workbench手工日期入口调用runForDateNow未guard。新增workbench入口与日历展示修复，不改变核心手工/强制路径，不写用户旧状态。用户要求收集修改后统一版本：本轮不build/pack，P2维持pending。

## 2026-10-04T01:28:18+08:00 — P9 accepted, review batch held

周末工作台入口复用core日期规则提前跳过；精确匹配旧周末无bucket记录的展示，不修改真实历史。49项相关回归与typecheck通过。已保存源代码，未构建或打包，不推进P2，等待用户继续反馈后统一版本。

## 2026-10-04T01:33:54+08:00 — P10 wordmark and proposal

用户要求标题只保留加粗加大的arxiv daily，并先审核新图标。标题源码已改（21项UI测试）；四个SVG与预览图供审核，新图标未用于实际UI。暂停等待反馈，继续不打包新版本。

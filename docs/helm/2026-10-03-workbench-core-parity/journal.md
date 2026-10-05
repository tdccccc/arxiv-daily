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

## 2026-10-04T01:42:12+08:00 — Identity alternatives for review

首版标识未获用户认可，按请求准备六个不同方向A–F与单色/小尺寸对照，位于docs/design/arxiv-daily-identity/alternatives。已渲染查看comparison.png并校验12个SVG。仅设计稿，未选定、未接入UI、不打包版本，等待选择后细化。

## 2026-10-04T01:47:51+08:00 — Pure wordmark selected and release requested

用户不再要图标，选择arxiv-daily字标；本轮要求更新统一版本，纳入单模型combobox、周末skip、标题简化。原图标提案仅作历史文档，不接入运行包。

## 2026-10-04T01:51:51+08:00 — P11 shipped for user review

采用arxiv-daily纯字标，已把近期单框模型选择、周末skip、标题改动合并成0.1.9并安装到用户Web profile。270 CLI/20 DSH检查及hash校验通过。保留现有会话等待用户重启，暂停等待效果反馈；P2/P3/P4不自动推进。

## 2026-10-04T02:21:14+08:00 — P12 model fetch regression

用户报告Get models后Getting started重新出现。saveDraft每次替换guide导致状态/布局重新展开。分离静默保存与显式guide更新，两条复现测试已Red→Green。

## 2026-10-04T02:22:20+08:00 — P12 accepted

Get models静默保存不再重绘或重新插入引导，两条用户场景回归通过。0.1.10已安装且hash匹配，保持现有会话等待用户重启，其他功能暂不推进。

## 2026-10-04T17:39:35+08:00 — P13 appearance and language

用户要求排查中英文混用，并新增外观分组容纳主题与界面语言。界面语言与总结语言独立；保留之前Obsidian字段结构，新增用户明确授权的外观组。

## 2026-10-04T18:11:35+08:00 — P13 accepted

用户追加尽量统一core：已把appearance类型/默认/校验和词典放core，前端消费共享定义、宿主存储用锁合并独立偏好。0.1.11已安装，292 CLI/20 DSH检查通过。主题与界面语言集中在设置外观，保存后应用，与summary_language独立。Obsidian原生渲染/存储未强制合并；其后续国际化仍需独立接入。等待用户效果反馈。

## 2026-10-04T18:24:00+08:00 — P14 shared business settings

用户明确授权继续统一模型、topic、邮件等业务设置。选择共享定义+编辑规则+操作服务，宿主保留存储/渲染适配，不做静默配置值同步。

## 2026-10-04T19:18:06+08:00 — P14 accepted

业务设置共享定义、编辑规则和模型/邮件操作已收拢。跨宿主真实适配器合同通过；兼容旧custom reasoning及sender name空白。CLI299/Obsidian771/core34/DSH20检查通过，DSH0.1.12已安装且hash匹配，Obsidian仅构建不部署。配置值不静默同步，宿主事务与授权保留，后续文献库主页面等原路线仍pending，等待用户检查。

## 2026-10-04T21:58:40+08:00 — P15 queued before context compression

用户同意共享层优先优化：先做core结构化运行状态，前端仅适配展示。已保存P15目标/测试边界和resume.md，尚未修改执行逻辑。压缩后直接按P15继续，不重新询问范围。

## 2026-10-04T23:04:36+08:00 — P15 accepted

On track: structured source/pipeline outcomes and additive state/history fields now distinguish awaiting announcements, confirmed no updates and no matches. Actual failures have a separate retry budget; legacy state and pending finalization remain compatible. Core manual weekend policy replaces the workbench-only bypass.

Host integration required an L1 transport adjustment: actual DSH tests exposed that in-process callbacks never crossed the child CLI boundary. Typed IPC now carries results; configured and first-run packed Host paths pass for successful generation, waiting and no updates, including persisted calendar states. Obsidian adapters and bilingual workbench views consume outcomes. First-report onboarding excludes no-update days, and durable errors outrank stale job outcomes.

Observed Red/Green and regressions are recorded in phase15: core279 focused, CLI312 full plus subsequent75/27 focused, Obsidian774 full, DSH20 including actual Host; final artifact Host rerun2/2; typechecks/boundaries/inventory/builds pass. Commits c00cff9,1c1f33b,3922e08. DSH0.1.13 linux/x64 package is ready at extensions/dsh-arxiv-daily/dist/dsh-arxiv-daily-0.1.13.tgz. No production profile/vault was modified; installed Web version remains0.1.12 and Obsidian was only built. No Computer Use, real paid models or email.

P15 is complete within the requested scope. P2/P3/P4 stay pending for subsequent work.

## 2026-10-05T01:53:35+08:00 — P16 start

用户授权将当前分支rebase到最新main。备份eafe972，目标7e1774b；104项关键基线通过。按行为保持型整合执行，主线不变，不push或安装生产插件。

## 2026-10-05T02:15:06+08:00 — P16 rebase accepted

Rebased all77 original commits onto main c2de350; retained backup eafe972. No dropped patches (67 unchanged,10 adapted), plus scoped integration/docs/release checkpoints. Main's removal of standalone profiles required CLI topic acceptance and atomic receipts, not restoration of obsolete stores. Shared settings now preserve direction lists and the daily cap; P15 stays intact. Stable legacy direction IDs and page-text-only library parsing fixed with observed regressions.

CLI339, Obsidian1007, core focused suites, DSH20 actual Host, Claude package7 and native workflow5 passed; typechecks/boundaries/inventory/builds passed. DSH0.1.14 packaged locally; no production install, main edit or push. Detailed coverage/limitations in phase16. P16 complete; remaining product phases stay pending.

## 2026-10-05T11:17:00+08:00 — P17 scientific reading accepted

用户确认公式原文直接显示。概览/摘要/推荐理由曾仅转义文本；现在复用工作台Markdown/KaTeX渲染，标题/目录保留公式边界，跨空行display不再被段落拆分。数学标题重复也已按解析元数据修复。原始Markdown保持不变，无需重新生成内容。

Observed Red/Green and63 focused+38 final regression checks、CLI typecheck/boundaries/build passed. DSH20及最终包实际Host2/2通过，逐项验证本地CSS/WOFF2与CSP。0.1.15已打包但未改用户安装；无Computer Use/真实付费调用。P17完成，其余产品阶段继续pending。

## 2026-10-05T11:35:27+08:00 — P18 reading context accepted

Summary sources及其后的资料现在分隔显示，生成统计位于页尾。Core以兼容callout+结构化JSON保存token/耗时/生成时间，旧缺失值不补造；概览日报统计注明整份日报范围。前进后退保留本会话历史、筛选/锚点/滚动位置，同时保留返回列表。

Observed Red/Green：core332、CLI358、Obsidian1007、DSH20通过，typechecks/boundaries/inventory/build通过。0.1.16本地包已生成，未修改生产安装；无Computer Use/真实模型/邮件/用户文献修改。P18完成，后续产品阶段继续pending。

## 2026-10-05T12:16:11+08:00 — P19 merge acceptance start

用户认为首版接近可用并请求合入主线及更新README。进入完整验收和用户文档整理；仅授权本地main合并，不推送或发布。main初始a1cd713、feature4d2b583均干净，后续GUI阶段保持pending。

## 2026-10-05T12:29:28+08:00 — P19 local main merge accepted

First workbench merge accepted after full gate. main a1cd713 was clean; backup/main-before-workbench-20261005 preserved it, then main fast-forwarded to feature d1ad169. Seven user docs updated and35 links checked. Shared functionality is ready as a first iteration; dedicated library/review GUI remains pending.

Root3881 passed/2 real-corpus opt-in skips; release tools368; DSH20; Claude package7; typechecks/boundaries/inventory/build/submission/smokes/published-manifest all passed; audit0. Lint0 errors/22 allowed warnings. Fixed nested-dependency release checker and DSH runtime-file CI coverage with observed regressions. No push/publish/install, user corpus access or Computer Use. Closing records will be fast-forwarded to main as well.

## 2026-10-05T12:49:49+08:00 — P2 resumed

用户确认将0.5.0已有文献库检索和方向审核接入DSH网页。原生界面与core均未丢失；此次补宿主界面，不重建业务。先P2库浏览/检索/PDF，再P3审核，保留现有设置与运行管理。继续原隔离worktree，无生产安装或真实文献处理。

## 2026-10-05T12:59:01+08:00 — P2 accepted, P3 started

Personal-library HTTP/DOM flow now handles catalog and indexed fallback PDFs without requiring an agent conversation. Core retrieval, source guards and existing settings are reused. CLI387, final adapter46, UI27, DSH20, typecheck/build passed; no real corpus/model. Proceed to P3 parity against native review operations with typed service boundaries and receipt-safe acceptance.

## 2026-10-05T13:14:56+08:00 — P3 accepted; DSH0.1.17 packaged

Connected the existing native proposal workflow to web/DSH: explicit authorization, evidence/drafts, rename/move/remove, preview, partial acceptance and overview. Preserved ordinary topics/receipts; did not restore retired profiles. Readonly review/catalog/PDF no longer queue behind model jobs. Fixed delayed-body operation gating and preview restoration with observed regressions.

Full repository3950 passed/2 real-corpus opt-in skips; types/boundaries/inventory/build passed; lint0 errors/22 warnings. Final DSH20 actual isolated Host, Claude7, documentation34 links and package bytes verified. Package0.1.17 sha2567daeeecadb2802534f746bf187b17f531b9d884cea5fc3231657c7ce4838f294. Core df00de4, service20d90e6, UI7b5e32c, docs/version41a4857. No production install, user corpus/model/email, Computer Use, push or main merge. P2/P3done; P4 remains pending, no active phase.

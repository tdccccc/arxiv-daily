# 合并结果与原生备份诊断准备

## 合并已完成

用户授权普通merge后，PR51合并到main的提交为`c620caeda05af5bfa5c4236f935b6b30d75fb8b6`；PR43也被GitHub标为merged。该提交tree与通过PR检查的ace7745相同。本地main仍976c12b，未reset；标签0.5.0、GitHub Release和npm0.5.0均尚未创建。

## 合并后的异常与复测

[main native run37012308927](https://github.com/tdccccc/arxiv-daily/actions/runs/37012308927)第一次macOS arm64失败：storage.test.cjs第8项原子替换测试在最终读取backup时得到3个NUL，期望old；新primary=new。9/10通过，native资产组装跳过。证据保存于`/tmp/arxiv-main-native-darwin-arm64-evidence/native.tap`及native-build.json。

sourceHash为71122e97faf9a4e14987025db14b6c6902c94737c0acc15df3a776040ff1c8f3。只读检查pwrite/EINTR/短写、NAPI字符串生命周期、关闭与RAII，未定位确定的C++错误；未发现测试目录互删或并行改写证据。此前没有在写入和link之后读取old，所以不能把最终backup异常直接归因rename或APFS。

原失败任务按同SHA复测，attempt2所有六平台和asset assembly成功。第一次失败仍保留，不以复测绿认定根因已解决。Root/CodeQL/relay/companion的合并后检查成功。

## 本地诊断分支

从合并提交创建`fix/native-backup-diagnostics`，不改生产C++：

- 新增独立诊断程序，默认每模式100次、最多100次，第一处错误终止该模式且非零退出；继续其它模式作为对照。
- 模式为native原流程无中间读取、native逐阶段、POSIX Node写入+truncate及Node不truncate。Windows只执行两native模式，避免不支持的Node目录fsync。
- 记录write/close/link/重复link/rename/sync阶段的固定old/new字节、size/dev/ino/nlink、OS/Node/文件系统/native元数据。文件内容最多采样64字节，清理仅自身mkdtemp，不读取真实资料库或配置。
- 新的CI诊断步骤在native构建成功且未取消时执行，即使基础测试失败；结果JSON始终纳入verification evidence。原失败断言保留，诊断失败也阻断job，无continue-on-error。
- 没有预先改硬链接策略、加sleep或追加flush来猜测性修复。

## 实际验证

- 原native测试本机Linux 10/10通过；`/tmp/arxiv-native-local-baseline.tap`。
- 新诊断最初stub下8项契约全部Red；实现后8/8通过，覆盖注入NUL、close后损坏、copy冒充hardlink、相同内容替换inode、错误迭代上限和工作流条件。
- 真实当前native addon诊断：四模式各100/100通过，共400次；`/tmp/arxiv-native-backup-new-diagnostic.json`。Linux x64/Node22.22.2/ext文件系统，不能当作macOS验证。
- 完整release-tools第一次349/350，原工作流守卫只允许无if测试。仅为这个“基础失败后也执行”的诊断步骤加入精确条件约束，并新增不可改为false的负例；其它基础测试依然不许跳过。最后350/350通过，Product unit inventory OK。
- check:boundaries、check:release-version0.5.0及diff检查通过。没有生产源码变更，因此未重复全workspace功能测试、构建或手测部署。

## 未完成及下一步

诊断未在macOS上运行，根因未定，不称数据完整性缺陷已修复。建议授权推送诊断分支并创建Draft PR，获取一轮有界平台对照。如果复现则按首个出错阶段修复；如果未复现则保留未知风险和证据，交用户决定继续调查或进入发布，不无限重跑直到假装证明安全。

没有push诊断分支、创建新PR、合并诊断、打tag或发布，也未触碰其他worktree。此前CodeQL五告警已在ace7745的远程检查通过，与本次native异常是不同结果。

## 后续更新（分支fix/release-050-readiness，2026-10-03）

诊断分支已获授权push并合并：PR52（`test(native): diagnose intermittent backup byte mismatch`）head `53623bb`通过全部10项required检查及CodeQL结果检查（0条新增告警），已以普通merge并入main，合并提交为`bdfec4e13f7188b96af4cd2437b9c4bd1de527b6`。

[PR52 native诊断CI](https://github.com/tdccccc/arxiv-daily/actions/runs/37033667259)实际结果：Linux x64/arm64与macOS x64/arm64各完成四种模式×100次循环，Windows x64/arm64各完成两种native模式×100次循环，**共2000次循环全部通过，零失败**；每个平台的原有10项native测试也全部通过。两个macOS目标均为Node22.17.0/Darwin24.6.0，native源码哈希与原始失败构建一致。

结论未变：根因仍未定位，复跑与本轮有界诊断均未复现问题，不称数据完整性缺陷已修复或已排除。原始失败证据（[run 37012308927 attempt 1](https://github.com/tdccccc/arxiv-daily/actions/runs/37012308927/attempts/1)）保留不覆盖。诊断代码保留在CI中，后续若复现可提供write/close/link/rename/sync各阶段的字节与inode证据，而不只是一个裸测试失败。

本分支（fix/release-050-readiness，从main@bdfec4e创建）已在`docs/releases/0.5.0.md`新增"Known issues"小节记录此项，供发布说明读者知悉；该小节未声称问题已解决，只记录已观察到的事实、已执行的诊断范围和已知的不确定性。goal.md第7项成功标准（定位根因）本轮仍未达成，保持未勾选。

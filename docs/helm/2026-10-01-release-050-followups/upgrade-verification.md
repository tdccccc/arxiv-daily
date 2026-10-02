# 2026-10-02 真实旧资料库升级修复

## 确认原因

用户13:10提供错误后，只读指定测试库`.index`诊断，未读取data.json或PDF，未改真实数据：

- `031e99…`对应manifest记录为1个摘要chunk，paper文件仍116个全文chunk，提取规则版本与更新时间也不同；原复用路径只信manifest，generation严格校验才阻止混配。
- 旧索引为remote:nomic-embed-text:768，当前local multilingual-e5-small-q8；索引层与存储层原本都要求手动删库才能切模型。
- 方向建议主/备份是合法schema3，revision2/1，不是损坏。旧decoder与新retired decoder均验证通过，但此前新store漏识别v2/v3。
- UI只看旧papers数量，整体准备失败后隐藏Retry/误放行Review；重启后旧source revision也可能误判已就绪。

## 修复与证据

1. 索引提交`dd29ff2`：复用前严格核对真实paper文件，混配/缺失重新抽取；切model或dimension先写隔离目录再CAS切换，保留旧数据；读取绑定捕获的manifest。builder严格校验未放宽。77项接手基线通过；混配/模型切换/覆盖/跨模型快照有效Red后修复，最终5文件137项通过（含真实FileStore取消、最终写失败、切回和重试）。
2. 旧建议提交`014786c`：严格识别合法旧v2-v5并返回regeneration-required；只有显式恢复才先归档旧主/备份、再重新读取确认未变化、最后保存新建议。损坏/异scope/并发出现当前草稿均拒绝。46项基线通过，新增有效Red后最终4文件182项通过。
3. UI/host：保留lastRun历史但独立标记preparationError，显示Retry并禁用Review；模型保存成功后核验，失败不触发新状态。旧建议提供明确Regenerate suggestions按钮，普通或自动首次生成不带恢复意图。main核对模型/维度和generation source revision；准备失败与重启均不会把旧计数当就绪。main状态4项Red、恢复意图透传1项Red、重启generation未完成1项Red后，最终相关3文件45项通过；UI9文件442项通过；harness56项通过。

## 全量结果

| 命令 | 观察结果 |
|---|---|
| NODE_OPTIONS=--max-old-space-size=8192 npm test | 最终exit0：core2311、node-runtime67、CLI125、plugin976；3479通过，2项core既有跳过 |
| npm run lint | exit0，0 errors / 20 warnings |
| npm run typecheck | 全工作区exit0 |
| npm run build | exit0 |
| npm run smoke:build | Build smoke OK |
| npm run check:boundaries | Workspace boundaries OK |
| npm run check:obsidian-submission | PASS |
| npm run test:release-tools | 342通过，Product unit inventory OK |
| git diff --check | 通过 |

首次全量3项失败来自旧测试直接Object.create插件却未初始化indexStatus；补齐真实状态对象后定向26项通过，再跑上述完整最终检查。相关未处理取消拒绝随夹具修复消失，不将失败运行当通过。

最终日志：`/tmp/arxiv-upgrade-final-tests.log`、`/tmp/arxiv-upgrade-final-typecheck.log`、`/tmp/arxiv-upgrade-final-lint.log`、`/tmp/arxiv-upgrade-final-build.log`、`/tmp/arxiv-upgrade-final-smoke.log`、`/tmp/arxiv-upgrade-release-tools.log`。详细过程：`/tmp/arxiv-upgrade-index-repair-report.md`、`/tmp/arxiv-upgrade-proposal-repair-report.md`、`/tmp/arxiv-upgrade-ui-repair-report.md`。

## 边界与复测

- catalog从备份恢复是已有保护，不能和阻断混同。个别PDF前两页无标题/摘要仍报告单篇失败，其余论文应完成。
- 旧模型与失败暂存文件保留，占用额外磁盘；本轮不自动清理。旧建议归档只在用户明确重新生成并保存时发生。
- 本轮未实际重建用户资料库、未请求真实LLM/模型下载、未启动Obsidian。不能据单测断言用户库已恢复。
- 部署后用户重启，点击Retry preparation；完成后点击Review suggestions，若提示旧版，点击Regenerate suggestions。核对保存后再次打开仍保留，反馈成功/失败数量与错误日志。
- UI与主流程提交`319c10b`；版本仍0.4.6，发行说明草稿未提交，0.5.0发布仍等实际流程复测。无push/PR/tag/发布，无其他worktree修改。

## 部署

目标：`/home/tiandc/Desktop/plugin_test/.obsidian/plugins/arxiv-daily/`。先看目标，独占创建`main.js.bak-20261002-pre-upgrade-recovery`和`styles.css.bak-20261002-pre-upgrade-recovery`，没有覆盖已有备份。

三件资产源目标hash一致：main.js `78521db050d98f2e801bdaa015424289f2bb5eaa648098b8a0aa2ad4c0b1bfdf`，styles.css `c1b84403be62a5ad5de4e5ed42e387176b0a8607fac395146749f41fefbe696d`，manifest.json `b54f7e409c97338a4b2b2457195f426b16d908a181457de2ffa410e8f9cdae91`。完整记录`/tmp/arxiv-050-upgrade-deploy.json`。

未启动Obsidian，未删除或手工改写用户索引、旧建议或data.json。真正恢复会在用户重启后点击Retry preparation/Regenerate suggestions时执行。

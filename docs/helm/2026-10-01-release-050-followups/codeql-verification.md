# 授权push、CI结果与CodeQL修复

## 已完成的远程动作

用户A授权的一次push已完成：origin/fix/review-followups与PR51均指向`163f2f63be2a70b54ea0b4d47f3660ec7ecb1604`。PR仍为Draft，说明未修改，未合并/tag/发布。

- [Root verification](https://github.com/tdccccc/arxiv-daily/actions/runs/36979970181)：同SHA，Root workspace与Node20.19.0/22.17.0均成功。
- [Native storage verification](https://github.com/tdccccc/arxiv-daily/actions/runs/36979970112)：同SHA，Linux/macOS/Windows各x64/arm64及release asset assembly全部成功。
- `gh pr checks 51 --required`确认10个required检查全部pass。
- CodeQL分析workflow成功，但[结果检查](https://github.com/tdccccc/arxiv-daily/runs/110752751156)失败：5条本次告警（4 high / 1 medium）。未将分析任务成功等同于安全结果通过。

原始结果：`/tmp/arxiv-050-codeql-check.json`、`/tmp/arxiv-050-codeql-annotations.json`。仓库另有历史open alerts，本次没有处理或关闭全部历史告警。

## 本地修复

| 提交 | 对应告警 | 验证 |
|---|---|---|
| 005906f | #14/#15 两处摘要提取正则 | 50项原基线，扩展后72项契约回归；保持daily首内容行与source多行旧语义 |
| 2f01ac0 | #35/#36 作者拆分/路径规范化 | 56项原基线，扩展后66项；schema5与锁不变 |
| 534fb2f | #17 设置路径原型污染 | 18项基线，新增后7项有效Red；最终348项相关测试通过 |

性能前后比较来自真实公共入口，有限输入、不联网、不读用户数据：source摘要16k空白约61.5→0.68ms；作者8192空白约16.817→0.030ms。daily摘要该输入未复现超线性，改后约0.88ms（旧0.056ms），不声称提速；路径此前已先合并slash，约0.02ms前后相当。后两项为明确线性、防御性等价改写，不制造不存在的性能Red。

原型路径测试实际复现Object.prototype哨兵污染，finally清理；修复事先拒绝所有位置的保留键，只遍历自有属性，不触发继承setter，普通设置及安全新叶子兼容。没有改变CodeQL配置或抑制告警。

## 修复后的本地检查

- `NODE_OPTIONS=--max-old-space-size=8192 npm test`：3519通过，2项core既有跳过；core2343/node-runtime67/CLI125/plugin984。
- lint：0 errors / 20 warnings；typecheck、build、check:boundaries、check:obsidian-submission、check:release-version 0.5.0全部通过。
- release-tools342通过；build smoke、packed CLI/native Linux x64 install smoke通过。
- 日志：`/tmp/arxiv-codeql-final-{tests,lint,typecheck,build,smoke}.log`、`/tmp/arxiv-codeql-{release-tools,install-smoke}.log`。
- 详细报告：`/tmp/arxiv-codeql-abstract-report.md`、`/tmp/arxiv-codeql-paper-index-report.md`、`/tmp/arxiv-codeql-settings-report.md`。
- 本机没有CodeQL CLI，修复提交未重新运行远程扫描；不能宣称5条告警已被CodeQL关闭。需要第二次push授权后确认。

## 手测目录同步

原目标`/home/tiandc/Desktop/plugin_test/.obsidian/plugins/arxiv-daily/`先检查再备份，创建`main.js.bak-20261002-pre-codeql-fixes`与`styles.css.bak-20261002-pre-codeql-fixes`，未覆盖旧备份。三件资产校验一致：main.js `6006fd34f4ff91e0b5b033ea971ca6071b0f239d4beb13674b29d60021184d2b`、styles.css `c1b84403be62a5ad5de4e5ed42e387176b0a8607fac395146749f41fefbe696d`、manifest.json `6f9b6e674c8f838acb81edb5fd1b5e0ae766c7105abedd3817fe292938d76a8f`。记录`/tmp/arxiv-050-codeql-deploy.json`，未碰data.json/启动Obsidian/修改其它worktree。

下一步只申请再次push当前分支并等待新检查；PR说明/ready、合并、tag和发布仍分别授权。

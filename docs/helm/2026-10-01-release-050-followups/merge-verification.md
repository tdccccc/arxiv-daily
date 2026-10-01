# 2026-10-02 整分支合并验证

## 范围与行为

用户选择A：把feat/library-directions-drive-daily全部已提交内容合入fix/review-followups。父提交326f84a，来源固定c6dc242（79个父分支尚未包含的提交）。没有进入或修改来源工作树，也未操作claude-code-research-plugi工作树；来源未提交内容不在本次合并中。

新行为：资料库索引仅解析前两页，提取标题与有长度上限的摘要；没有摘要标记时可能使用有界开头文本。全文正文独有词不再参与检索。未识别为arXiv的PDF仍可用本地证据参与搜索和方向生成。主题持有方向列表，库分析可提出整套主题/方向，支持编辑、预览、分次接受与已有覆盖；日报按方向筛选并默认限制20篇。

保留父分支的首次build按需扫描、已扫描目录不自动重扫、扫描/索引可取消、元数据失败PDF仍参与索引、setup完成后永久隐藏、设置失败恢复、共享运行锁、请求契约与检查点恢复。审核对话框保留宽外壳、小窗滚动、扫描入口。

## 合并适配与新增回归

- 13处文件冲突逐一融合；退休的旧profile筛选模块按来源删除，确认生产无遗留调用；CLI/核心两边可靠性测试均保留并适配新directions模型。
- 无derivation旧全文记录（同路径、同hash改名、旧键迁移）原会被错误复用：4项有效Red后修复。旧键重建失败仍可搜索旧全文：2项有效Red后隔离为失败记录，保留磁盘文件；重试成功可恢复摘要结果。当前格式正常复用仍通过。
- 引导仍按旧description判断就绪、焦点指向已删除输入框：方向列表回归Red后修复。
- 完成提示和远程披露仍称全文/仅arXiv方向：文案契约11项Red后通过。保留存储processingDepth token和授权校验，显示实际发送的标题摘要。
- 新索引未立即刷新审核证据：集成回归观察indexedPapers为空的Red，完成索引后刷新已提交标题信息，27项相关回归通过。
- 合并测试夹具和固定指纹变化、加载错误、旧文案断言不冒充行为Red。搜索路径仍只读一个manifest，索引后的额外读取用于刷新审核证据。
- 来源main.ts字面NUL/SOH改成等价转义，只恢复文本工具可读性，不改变指纹字节。

## 实际检查

| 检查 | 结果 |
|---|---|
| core索引/存储/摘要抽取/分块 | 4文件92通过（索引46）；含上述Red→Green |
| core筛选/检查点/运行 | 13文件520通过 |
| CLI | 9文件125通过 |
| plugin审核/设置/引导 | 6文件364通过 |
| plugin索引入口/完成提示/授权 | 4文件79通过 |
| plugin首次索引/生命周期 | 2文件27通过 |
| NODE_OPTIONS=--max-old-space-size=8192 npm test | 最终exit0：core2284通过/2既有跳过，node-runtime67，CLI125，plugin930；合计3406通过 |
| npm run lint | exit0，0 errors / 20 warnings |
| npm run typecheck | exit0，所有工作区通过 |
| npm run build | exit0 |
| npm run smoke:build | Build smoke OK；仅临时配置和构建资产验证 |
| npm run check:boundaries | Workspace boundaries OK |
| npm run check:obsidian-submission | PASS |
| npm run test:release-tools | 337通过，Product unit inventory OK；含更新后的桌面披露契约测试 |

桌面验收脚本的两种legacy depth都应显示标题摘要，旧全文披露必须失败：新增契约先观察5项Red，再修正常量与探测，单文件51项通过。该检查没有启动Obsidian。

最终全量日志：/tmp/arxiv-merge-final2-tests.log。此前全量剩余旧文案断言及一次搜索读取次数断言误改均已修正，失败日志保留在/tmp/arxiv-merge-all-tests.log与/tmp/arxiv-merge-final-tests.log，不计为通过。

其他日志：/tmp/arxiv-merge-final-lint.log、/tmp/arxiv-merge-final-typecheck.log、/tmp/arxiv-merge-final-build.log、/tmp/arxiv-merge-smoke.log。详细子任务证据：/tmp/arxiv-merge-index-report.md、/tmp/arxiv-merge-ui-report.md、/tmp/arxiv-merge-runtime-report.md、/tmp/arxiv-merge-semantic-review.md。

## 验证边界与下一步

- 本轮没有运行真实Obsidian、没有对用户PDF全集重新计时，没有发送邮件或调用真实LLM/arXiv。旧分支历史性能数字不作为这次新测量。
- 源分支部分桌面验收本来未完成；合并并不把它们变成已验收。需要新构建重测主题编辑、方向提案接受/重复接受、已有覆盖、库内预览与日报方向标注。
- 版本仍0.4.6，待新构建手测通过后再同步0.5.0。发行说明草稿已按新行为修订，仍不纳入合并提交。CLI日常配置编辑命令继续留后续版本；来源合入的是方向列表与每日上限的配置支持。
- 本地合并提交与部署结果完成后补记；push、PR更新、合并main、tag和发布均未执行。

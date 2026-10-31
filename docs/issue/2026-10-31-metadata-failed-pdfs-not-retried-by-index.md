# 元数据拉取失败的 PDF 不会被"索引 / Retry preparation"重试，部分失败时也不提示

状态：未修复

> 来源：2026-10-31 main `e0498c94` 代码审查（会话内直接审查，静态推断，未运行）
> 严重程度：低 · 置信度：代码推断

## 问题与依据

- 扫描时向 arXiv 批量拉取元数据一旦失败（限流、503、断网），本次新识别的论文全部记为 `failed: metadata-fetch-failed`（`packages/core/src/library/personal-library-reconciliation.ts:294-326`）。
- `detectPersonalLibraryFolderChange`（132-166 行）明确把 failed 记录视为"文件夹未变"；`packages/core/src/library/library-workflow.ts:172-206` 的 `index()` 只在文件夹有变动时才重扫——所以这些论文不会被重试。
- 完成提示 `plugin/src/library/index-completion.ts:80-86` 只在"一篇都没识别出来"时提到 arXiv 查询失败；部分失败时提示里只有 indexed / reused / failed（这里的 failed 是嵌入失败，不是元数据失败）。
- 失败数只在插件的文献库弹窗（`plugin/src/library/modal.ts:157`）和工作台文献库页（`apps/cli/src/workbench/web/library.ts:77`）里能看到。
- 恢复路径是插件命令 "Scan personal library folder (rebuild catalog)" 或 CLI `library scan`，提示没有指向它。

用户看到的是：往文献库加了 20 篇 PDF，索引时恰逢 arXiv 限流；之后这 20 篇一直搜不到、不参与方向建议；点 "Retry preparation" 没反应，也没人告诉他该重新扫描。

## 原因

设计上"重试 failed 是显式扫描的事"（代码注释如此），但界面没有把这件事告诉用户。

## 建议和验收

- `index()` 开始前若目录里有 failed 记录，顺带只对这些文件重扫（只向 arXiv 查这些 id）；或
- 在完成提示和 Library 行描述里显示 "N 篇 arXiv 查询失败——重新扫描以重试"，并给出入口。

## 修复后应补的测试

- `library-workflow` 测试：目录有 failed 记录、文件夹未变时，`index()` 触发对这些 id 的 resolver 调用，成功后变为 ready。
- 完成文案测试：部分失败时提示包含失败数和重试方式。

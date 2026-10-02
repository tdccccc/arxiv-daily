# 0.5.0本地发布准备验收

## 用户验收与候选版本

用户确认实际Retry preparation成功，并选择A确认建议生成、接受到主题、重新打开保存以及审核窗口操作均正常。Linux核心流程与已有资料库升级恢复通过用户手测；没有推断macOS/Windows真实桌面验收或精确性能数字。

版本与发行说明提交：`71d6453 chore(release): prepare 0.5.0 metadata and release notes`。五个workspace package、内部依赖、root lock、两个manifest与两个versions映射均同步0.5.0；minAppVersion仍1.4.0，未改变业务逻辑。发行说明已向用户展示，文件为docs/releases/0.5.0.md。

## 独立安装与检查

避免改动其他工作树使用的依赖目录，在`/tmp/arxiv-050-release-check-6tt3cmxz`复制当前已跟踪源码和候选元数据/发行说明，执行独立npm ci。逐文件比较确认被测源码及元数据与当前工作树一致（仅工作进度文档不参与运行比较），发行说明也一致。

环境：Linux x64、Node22.22.2、npm10.9.7。该检查不是GitHub CI，也不替代Node20.19/22.17及六native平台门禁。

| 检查 | 结果 |
|---|---|
| npm run sync:release-version -- 0.5.0 | 当前分支同步成功 |
| npm run check:release-version -- 0.5.0 | 当前分支及独立副本均通过 |
| npm ci | 成功，429包；根依赖目录未重装 |
| npm audit --audit-level=moderate | 0 vulnerabilities |
| NODE_OPTIONS=--max-old-space-size=8192 npm test | 3479通过，2项core既有跳过；core2311/node-runtime67/CLI125/plugin976 |
| npm run lint | 0 errors / 20 warnings，exit0 |
| npm run typecheck | 全workspace通过 |
| npm run test:release-tools | 342通过，Product unit inventory OK |
| npm run check:boundaries | Workspace boundaries OK |
| npm run build | 独立副本与当前工作区均通过 |
| npm run check:obsidian-submission | PASS |
| npm run smoke:build | Build smoke OK |
| npm run smoke:install | Native package smoke OK (linux-x64, offline)，CLI package install smoke OK |
| git diff --check / staged核对 | 通过，元数据提交仅11个版本/说明文件 |

根npm test覆盖所有workspace；core既有分批runner明确传--maxWorkers=1。没有把docs/release.md里带参数后仅路由core的命令误算为全workspace通过。

日志：`/tmp/arxiv-050-clean-install.log`、`/tmp/arxiv-050-final-{audit,tests,lint,typecheck,release-tools,build,smoke,install-smoke}.log`、`/tmp/arxiv-050-workspace-build.log`。CLI临时安装使用隔离配置，不调用真实init或写用户配置。

## 发行说明核对

确认两步资料库设置、标题摘要与前两页回退范围、未识别PDF证据、模型切换及旧建议恢复、约130MB模型首次下载、无镜像、arXiv ID/标题查询无独立同意弹窗、远程模型授权、默认20篇上限、Paper Index读1–5写5、topic tag自动管理与自定义tag保留、CLI文件0600/新建目录0700。明确普通日报同样受新方向与上限影响，不承诺未选资料库就完全无变化。CLI配置编辑新命令仍不纳入本次版本。

## 0.5.0手测资产同步

目标仍为`/home/tiandc/Desktop/plugin_test/.obsidian/plugins/arxiv-daily/`。先看文件，独占创建`main.js.bak-20261002-pre-release-050`与`styles.css.bak-20261002-pre-release-050`，复制三件资产并验证hash。main.js与用户通过手测的构建相同（78521db050d98f2e801bdaa015424289f2bb5eaa648098b8a0aa2ad4c0b1bfdf），styles.css同样未变（c1b84403be62a5ad5de4e5ed42e387176b0a8607fac395146749f41fefbe696d）；manifest为0.5.0（6f9b6e674c8f838acb81edb5fd1b5e0ae766c7105abedd3817fe292938d76a8f）。记录：`/tmp/arxiv-050-release-deploy.json`。未打开data.json或启动Obsidian。

## 仍需逐项授权的发布步骤

本地发布准备完成。下一步是用户授权push fix/review-followups，再等待当前提交的10个保护检查（root、两Node版本、六native目标、native assets组装）。随后分别授权更新PR51说明/ready状态、合并main、打0.5.0标签与发布。没有push、修改PR、合并main或发布，也没有清理分支/stash。两个受保护worktree与来源分支未修改。

CLI日常设置修改功能留后续版本，其范围与本地提交授权保留在journal；本次没有把它列为已完成。

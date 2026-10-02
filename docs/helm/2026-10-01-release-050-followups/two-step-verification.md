# 2026-10-02 两步资料库设置验证

## 用户确认的行为

- Personal library统一资料库设置；Research topics只编辑已采纳的主题与方向，重复Topics from your library入口移除。
- 设置页选目录成功后自动准备标题摘要索引，首次需要时扫描；以后索引沿用保存的catalog。未选择/取消选择不准备，插件启动不触发准备。
- 本地默认，不再额外弹出模式选择窗口；已有远程设置在实际处理前确认，拒绝不开始索引。
- 准备中显示进度和Cancel；无可搜索索引时提供Retry preparation；准备好后只提供选择/更换目录与Review suggestions等主要操作，撤销授权仍保留。
- 首次Review suggestions在没有proposal且有索引证据时自动授权/生成，已有proposal只打开。授权等待期间出现的草稿不会覆盖，读取错误不自动生成。关闭审核窗口取消本次生成，不影响其它操作。
- 本地模型提供加载进度；模型库对缓存读取也发download事件，因此只显示加载及“首次可能下载约130MB”，不冒充网络下载。取消后不再向当前界面报告进度；底层模型加载可能继续完成缓存，这是既有库限制。

## 基线与Red→Green

- 设置旧基线4文件300通过；两步新行为最初272项中19项失败，改造后扩展到8文件363通过。另有本地默认不弹模式与异步索引载入后刷新按钮的有效Red→Green。
- 审核旧基线90通过；自动生成新增9项中4失败后修复；关闭活跃生成时signal原为undefined，补齐后通过。最终审核/接受两文件102通过。
- 本地模型旧基线12通过；回调与取消2项Red后最终16通过。主端显示模型准备回归先失败，接线后索引相关28通过。最后去重进度百分比显示后的模型/索引3文件27通过。
- main调用者取消2项有效Red：关闭后仍启动、晚到响应仍保存；绑定本次operation信号后，生成相关15通过。
- 桌面harness更新两步按钮契约，7项Red后55/55通过；没有启动Obsidian。

## 完整验证

| 命令 | 结果 |
|---|---|
| NODE_OPTIONS=--max-old-space-size=8192 npm test | exit0：core2284、node-runtime67、CLI125、plugin956；3432通过，2项core既有跳过 |
| npm run lint | exit0，0 errors / 19 warnings，比原20警告少1 |
| npm run typecheck | exit0，所有工作区通过 |
| npm run build | exit0 |
| npm run smoke:build | Build smoke OK |
| npm run check:boundaries | Workspace boundaries OK |
| npm run check:obsidian-submission | PASS |
| npm run test:release-tools | 341通过，Product unit inventory OK |
| git diff --check | 通过 |

全量运行后仅把模型消息中的重复百分比去掉；相关27项重跑通过并重新构建。日志：`/tmp/arxiv-two-step-tests.log`、`/tmp/arxiv-two-step-{lint,typecheck,build,smoke,release-tools,model-final}.log`。细节报告：`/tmp/arxiv-unify-library-settings-report.md`、`/tmp/arxiv-review-autogenerate-report.md`、`/tmp/arxiv-model-preparation-progress-report.md`。main Red/Green日志：`/tmp/arxiv-library-two-step-host-{red,green}.log`、`/tmp/arxiv-library-model-progress-{red,green}.log`。

## 手测与发布边界

本轮没有实际模型下载、真实模型请求、用户PDF全集计时或真实Obsidian交互。手测重点是选目录自动准备、取消/失败可重试、Review suggestions第一次授权生成与后续直接打开、关闭停止生成、两种窗口大小仍能操作。

命令面板仍保留扫描与手动索引入口，可用于更新资料库；该底层选目录命令与设置页引导是不同入口。新增PDF后应重新扫描再索引。

发行说明草稿已同步两步行为但仍单独留到版本同步提交。版本仍0.4.6；新构建手测通过后继续0.5.0准备。来源分支与受保护工作树没有修改，没有push、PR、tag或发布。

## 提交与部署

功能提交`7914545 feat(settings): consolidate library setup into two steps`。部署目标`/home/tiandc/Desktop/plugin_test/.obsidian/plugins/arxiv-daily/`，先检查文件，再独占创建`main.js.bak-20261002-pre-two-step-library`和`styles.css.bak-20261002-pre-two-step-library`，未覆盖旧备份。

源/目标校验一致：main.js `eed36525182f699fb6f9d6820c258694bd55b16111b3a125d688206782867c00`；styles.css `c1b84403be62a5ad5de4e5ed42e387176b0a8607fac395146749f41fefbe696d`；manifest.json `b54f7e409c97338a4b2b2457195f426b16d908a181457de2ffa410e8f9cdae91`。完整记录`/tmp/arxiv-050-two-step-deploy.json`。data.json未打开或修改，Obsidian未启动。

重启后的最小手测：选择一个小资料库应直接准备；准备中取消后可重试；准备完成后Review suggestions首次授权生成，关闭时停止；已有建议再次打开保持原文；研究主题区不再出现重复准备入口。

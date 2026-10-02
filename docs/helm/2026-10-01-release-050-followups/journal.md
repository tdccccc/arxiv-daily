# Journal

## 2026-10-01 — 接手续作

- 原 agent 已停止，由用户确认；新建发布收尾记录，不修改其他 active initiative。
- 工作树与用户快照一致；先读 git diff、交接与发行流程。未 reset/stash/丢弃现有修改。
- 分工：scan_fix 处理 A 及必要 core 索引单元补齐；modal_fix 仅处理 B 的 modal/CSS；cli_recon 只读调查完成。子任务不得继续派生。
- 逐步执行报告在 /tmp/arxiv-scan-fix-report.md、/tmp/arxiv-modal-fix-report.md、/tmp/arxiv-cli-settings-report.md；最终验证/提交/部署结果由主会话汇总。
- 用户选择 CLI 首批 A：主题全管理与常用设置，交互＋非敏感单项命令；版本安排仍待确认，不开始 CLI 实施。

## 2026-10-01 — CLI 版本安排确认

- 用户第二项选择 A：放后续版本，先完成 0.5.0 收尾和手测；授权届时按独立范围本地提交。此次不修改 CLI 源码、不把新命令写入 0.5.0 发行说明。
- 后续范围：主题查看/增改/启停/删除；分类、时区、LLM、输出、邮件配置、计划时间；交互编辑＋非敏感单项命令，密钥隐藏输入。邮件发送、cron 安装继续走已有入口。
- 已核实：init 默认覆盖整份 TOML，Keep existing 非编辑；无配置/主题修改命令；配置路径用 XDG/APPDATA，--config 已移除；主题目前无 enabled 且默认位置 ID 不稳定。可复用原子私密保存与交互取消，须补严格输入校验、稳定 ID、保留未知字段与脱敏；注释保留取舍在实现方案中明确。
- 后续候选命名：config edit/set/show 与 topics list/add/edit/enable/disable/remove，尚不是已发布接口。只读调查未运行测试或修改真实配置。

## 2026-10-01 — P1 本地验收与部署完成

- Checkpoint: On track。修复 A 在 9e4bcb2 独立提交；修复 B 在 0df3f95 独立提交。提交前核对 staged diff；没有将发行说明草稿混入。
- 保留中断实现，以105项接手Green基线补偿未观察的历史Red；本轮新增可读但元数据失败PDF、空库重扫提示、全失败提示与取消处理均记录有效Red→Green。布局为低影响修改，用24项既有交互测试前后通过与样式审查验证。
- 全量测试初次因沙箱禁止 ~/.arxiv-daily/host-locks 写入而失败；经授权沙箱外重跑exit 0：core 2073通过/2原有跳过，node-runtime 66，CLI 110，plugin 815。lint 0 errors/20 warnings；typecheck/build/boundaries/submission全部通过。
- 部署前查看指定目标，以COPYFILE_EXCL创建main.js/styles.css的20261001-pre-build-scan备份，然后复制三件资产，源与目标SHA-256一致。未打开或写入data.json，未启动Obsidian。
- 可选浏览器几何模拟未完成；额外空白会话已关闭/确认PID不存在；不冒充真实宿主验收。
- P1完成，P2仍pending，等待用户手测与识别数量。整体目标保留active，不开始版本同步。详细验证与发布草稿核对待办见verification.md。

## 2026-10-02 — L3：整分支纳入0.5.0

- 用户确认A：合并feat/library-directions-drive-daily全部已提交改动，认为目前标题摘要索引足够。来源固定c6dc242，共79个当前父分支未含提交；不修改来源分支/工作树。
- 合并前父提交326f84a，已有P1可靠性修复保留；P1完成记录不重写，新模型替代旧全文与独立confirmed profile语义，由P3整合。P2仍等待合并后用户手测。
- 术语与决策合入来源CONTEXT和ADR0012/0013/0014；旧注释/说明与新语义不符处同步。旧全文索引需重建，正文独有词不再参与搜索。
- 合并冲突以保留两侧有效行为为目标；已有实现不伪造历史Red。当前基线和新增缺口Red/Green逐步记录在/tmp/arxiv-merge-*.log/report.md；源码分工互不重叠。
- 范围调整记录和P3计划随本次merge提交保留，避免在未解冲突的Git状态创建独立提交。CLI日常设置编辑仍是后续版本，来源既有CLI方向支持不等于新增编辑入口。

## 2026-10-02 — P3验收、合并与再次部署

- Checkpoint: On track。实际merge提交1f3e0d6，双父326f84a/c6dc242，来源分支仍c6dc242。新增整合回归和全量3406通过/2原有跳过；lint 0 errors/20 warnings，typecheck/build/smoke/boundaries/submission通过，release-tools337通过。
- 旧无版本全文复用、旧键转换失败隔离、方向引导、正文披露、建完索引立即看到审核证据均有本轮Red→Green证据。合并夹具/旧文案错误不冒充Red。详见merge-verification.md及其日志索引。
- 三件资产重新部署到原手测目录，备份suffix为20261002-pre-title-abstract-merge，SHA-256核对一致。未访问data.json或启动Obsidian，未影响来源/其他工作树。
- P3完成，P2保持pending等待新构建手测，整体active。发行说明已改为标题摘要及新主题方向行为但仍是未跟踪草稿，不进行版本同步。没有push、PR、main合并、tag或发布。

## 2026-10-02 — 用户把资料库设置收敛为两步

- 用户指出Topics from your library和Personal library重复，并进一步选择“选择目录后自动构建索引 → Review suggestions”。新指令取代旧的“选目录不扫描”。本轮继续用同一initiative，P1/P3历史验收保留，新P4跟进改变的交互。
- Research topics只管理已采纳的方向；Personal library统一选择、准备进度/取消/重试和审核。首次审核无proposal时在授权后自动生成；有草稿或读取失败不覆盖，关闭本次窗口取消其生成而非其他运行。
- 选择后的本地/远程选择弹窗移除，本地默认，Embedding设置仍可调整；已有远程模式在索引前只询问一次授权。加载模型状态使用真实文件进度，但库API在缓存命中也发download事件，因此不伪称每次下载130MB。
- 分工：modal_fix负责设置/connection，cli_recon负责审核modal，scan_fix负责本地模型加载回调，主负责main桥接/取消/记录。禁止触碰其它worktree或继续派生。
- 旧方案中断前仅有300通过基线与/tmp报告，无源码修改，不丢弃用户工作。新增两步流程/首次自动生成/关闭中止/模型状态均先观察Red，再最小补齐。

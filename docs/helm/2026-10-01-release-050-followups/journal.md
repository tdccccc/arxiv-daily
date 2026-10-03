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

## 2026-10-02 — P4提交并部署

- Checkpoint: On track。7914545提交两步流程与必要进度/取消衔接，暂存范围核对，无发行草稿混入。3432项全量通过/2项既有跳过；lint0 errors/19 warnings；typecheck/build/smoke/boundaries/submission通过；release-tools341通过。
- main.js/styles.css备份后重新部署三件资产，suffix20261002-pre-two-step-library，源目标SHA-256相同；未打开data.json，未启动Obsidian。来源分支保持c6dc242，受保护worktree未修改。
- P4完成，P2继续等待真实宿主手测，尚未同步0.5.0版本或执行远程动作。实际模型下载与库全集计时未测；细节和手测清单见two-step-verification.md。

## 2026-10-02 — 真实手测暴露升级阻断，P5接续

- 用户仅确认弹窗可打开，随后提供13:10错误日志：旧remote:nomic-embed-text:768到local模型被拒绝，203复用后generation严格一致性失败，v3方向建议被误报损坏。P2仍pending，不同步版本、不称已具备发布条件。
- 只读测试.index结构验证混配1摘要chunk/116旧全文chunk，建议主备份均合法v3；未修改真实数据。P1/P3/P4保留历史测试验收，新增P5覆盖此前缺失的升级场景，不重写历史或忽略报错。
- 修复分工隔离：scan_fix处理core fulltext存储/索引，cli_recon处理core旧建议识别/显式归档，modal_fix处理UI状态与恢复按钮，主负责main状态/恢复意图接线。全部新增行为采用实际Red→Green；报告在/tmp/arxiv-upgrade-*-report.md。
- 对用户解释：catalog从备份恢复是已有防护，少数PDF前两页无文字是单篇失败；整库构建与重试入口必须恢复。新模型先隔离暂存再原子切换，旧建议只有显式恢复才归档和重写。

## 2026-10-02 — P5本地验证与部署完成

- Checkpoint: On track。分别提交dd29ff2（索引一致性与模型安全切换）、014786c（合法旧建议识别/归档恢复）、319c10b（主流程与UI恢复入口）。暂存范围逐次审查，发行说明草稿未混入。
- 最终3479测试通过/2既有跳过，lint0 errors/20 warnings；typecheck/build/smoke/boundaries/submission通过，release-tools342通过。首次全量旧fixture缺indexStatus的3失败已补齐并重新全量验证，不计为通过。
- 三件资产备份后部署，suffix20261002-pre-upgrade-recovery，SHA-256核对一致；main.js新hash78521db050d98f2e801bdaa015424289f2bb5eaa648098b8a0aa2ad4c0b1bfdf。未写真实.index/data.json，未启动Obsidian，未触碰来源/受保护worktree。
- P5完成指代码与本地验证完成，用户库实际恢复仍待重启Retry/Regenerate复测；相应成功标准保持未勾选。P2继续pending，无版本同步、push、PR、main合并、tag或发布。

## 2026-10-02 — 用户完整手测通过，开始P2

- 用户先确认Retry preparation“点击了，可以了”，随后明确选择A：建议生成、接受到主题、重新打开保存及窗口操作均正常。按实际反馈接受Linux核心流程；没有把仅弹窗能打开扩张为完整通过。
- P2开始，版本同步0.5.0及一致性检查已通过，未push。目录权限仍仅保证新建配置目录0700，Paper Index实际写5，主题tag完全隐藏，发行说明相应校正。CLI后续编辑命令不纳入。
- 为避免npm ci影响其他工作树，在/tmp/arxiv-050-release-check-6tt3cmxz复制当前已跟踪源码与候选元数据/发行说明，独立npm ci成功（429包，审计0漏洞）。在此副本执行发布门禁；不将此称作跨平台CI验证。

## 2026-10-02 — P2完成，本地发布准备结束

- 71d6453提交0.5.0版本与最终发行说明。独立npm ci/audit为0漏洞；3479测试通过/2既有跳过，lint0 errors/20 warnings；typecheck/build/boundaries/submission、342release-tools、build与CLI安装smoke均通过。源/元数据与被测副本比对一致。
- 用户完整Linux手测与本轮候选检查覆盖本地成功标准，全部勾选，P2及goal done。后续远程发布从push授权点继续，不把本地done写成已发布。详情release-verification.md。
- 手测目录manifest同步0.5.0；main.js/styles.css与用户验收通过的字节一致，保留pre-release-050备份，未触碰data.json或启动Obsidian。
- 后续依次：授权push → 当前提交10个CI门禁 → 授权PR51说明/ready → 授权合并 → 授权tag/发布；CLI配置编辑仍留后续版本。

## 2026-10-02 — 授权push与CodeQL后续

- 用户A授权push并等待CI。实际push成功，PR51与origin/fix/review-followups指向163f2f63be2a70b54ea0b4d47f3660ec7ecb1604，PR保持Draft，未改说明或合并。
- 10required检查全成功：Root workspace、Node20.19/22.17、六native平台及资产组装。Root run36979970181、native run36979970112均同SHA。CodeQL workflow成功但结果检查110752751156失败，5条本次告警（14/15/17/35/36）。没有把扫描运行成功误报为安全结果通过。
- P1–P5与P2已验本地结果保留；新P6处理远程新增证据，goal恢复active。代码分工：scan_fix两个摘要解析器、cli_recon索引路径/作者、modal_fix设置写入。禁止派生/触碰worktree/远程修改。主负责核对与提交，再次push须单次授权。
- 原始check与annotations保存在/tmp/arxiv-050-codeql-check.json、/tmp/arxiv-050-codeql-annotations.json；历史open alerts不当作本次五条的同义词。

## 2026-10-02 — P6本地修复完成，等待再次push

- 三项独立提交005906f（摘要扫描）、2f01ac0（作者/路径处理）、534fb2f（设置原型路径保护）。性能修复采用公共契约Green和真实前后测量，只有有证据的source/author称实测加速；设置保护有7项有效Red。
- 最终本地3519通过/2既有跳过，lint0 errors/20 warnings，typecheck/build/boundaries/submission/version-check、342release-tools及build/install smoke通过。说明新增元数据/设置安全处理一行。
- 手测目录备份pre-codeql-fixes后更新三件资产，未改data.json或启动Obsidian。详细CI链接、告警对应提交和限制见codeql-verification.md。
- P6仍active：当前远程CodeQL结果只对应163f2f6。没有第二次push授权，没有关闭/抑制告警；请求用户单次授权后重跑再验收。

## 2026-10-02 — PR51合并与main原生测试异常

- 用户分别授权第二次push、按已展示草稿更新PR51/Ready、普通merge。ace7745全部14个checks成功（10required），CodeQL结果0新annotations且五条告警不在PR open列表。
- PR51与PR43均merged到c620caeda05af5bfa5c4236f935b6b30d75fb8b6；merge tree与ace7745一致。未reset本地main，未创建0.5.0标签/GitHub Release/npm版本。远程Tag为空、Release与npm0.5.0查询均不存在。
- main native run37012308927第一次macOS arm64 test8备份读到3NUL，expected old；其余9测试通过，新primary=new。下载TAP证据在/tmp/arxiv-main-native-darwin-arm64-evidence。第一次下载TLS超时，第二次成功。
- 只读C++/测试复核未定位确定缺陷，也未发现fixture并行/互删；测试首次读old太晚，无法区分写入/硬链/改名阶段。Linux10/10与四模式各100次通过不代表macOS通过。
- 作为定位实验执行同SHA失败任务rerun，attempt2全部native/assembly成功；不把这写成第一次异常已解决。报告/tmp/arxiv-native-backup-corruption-review.md、/tmp/arxiv-native-backup-test-review.md。
- 从c620cae创建本地fix/native-backup-diagnostics，P7只准备诊断与测试，不改C++生产逻辑。推送与创建诊断PR尚未授权，正式发布继续停在授权点。

## 2026-10-03 — PR52合并，分支fix/release-050-readiness处理另17条历史告警与发行说明

- PR52（native备份诊断）已获授权push与merge，合并提交`bdfec4e`：2000次诊断循环覆盖6平台全部通过、零失败，原始macOS arm64异常仍未复现、根因未定。详见native-backup-diagnostics.md的后续更新。
- 新分支`fix/release-050-readiness`从main@bdfec4e创建，处理与PR51/52无关、自2026-08-23起即存在的17条open high-severity CodeQL告警（11条polynomial-redos、3条incomplete-multi-character-sanitization、2条bad-tag-filter、1条incomplete-url-substring-sanitization）。14条本地修复（位置扫描替代回溯正则、`trimEnd()`替代`/\s+$/`、`indexOf`扫描替代惰性双定界符正则、闭合标签允许空白、跨行HTML注释匹配），均附回归测试；3条判定误报/不适用并写入具体理由（测试内mock URL判断；两条因DOMParser/linkedom解析结果从不挂载到真实DOM而不可达）。详见codeql-verification.md新增小节。
- 发行说明`docs/releases/0.5.0.md`新增"Known issues"小节记录native备份异常（诚实陈述：单次复现、同SHA复跑通过、2000次诊断循环无复现、根因未知、诊断保留在CI），并在"Security and dependencies"补充本轮17条CodeQL处理结果的一行摘要。
- 同分支另处理两条桌面验收场景（`buildFullTextDocumentParser`已在main随标题摘要索引改造移除）：retarget为基于现有`buildFullTextExtractor()`的场景，其中一条改名为验证"即使手动开启sidecar并指向可达监听端口，索引仍不发起请求、仍返回PDF.js、且记录文档化的info日志"，取代已不存在的"探测失败回退"叙事。
- 本分支未push、未创建/修改PR、未合并到main、未打tag、未发布，也未触碰任何`.worktree`。

# P8 — Unresolved 文件兜底全文索引

goal_ref: ../goal.md
updated: 2026-08-13

## Outcome

个人库中无法识别 arXiv 编号的 PDF（`catalog.files` 中 `status: "unresolved"`，如作者-年份命名文件）也能进入全文知识库和混合检索：以完整 PDF bytes 的 SHA-256 生成 `file:sha256:<digest>`，从首页提取可版本刷新的标题，三个 Obsidian 检索入口显示标题、相对文件路径和相似度。fallback 论文可参与 placement、缓冲池 recluster 和已有方向 attach；P8 在真实 Obsidian 的 Scan、索引、方向更新和代表性查询通过后完成。

## Assumptions

- unresolved 文件以真实文献为主，全文检索对用户有实际价值；它们没有可依赖的 catalog title/authors/abstract。
- PDF bytes 内容摘要是 fallback 身份；路径观测变化后重新读取成功且内容未变时，paperKey 与已有向量保持不变。
- 首页标题提取是启发式能力；提取失败时 UI 可回退到 paperKey，标题规则通过 `titleVersion` 触发重读而不重嵌入。
- scoped source 默认单文件读取上限为 25 MiB；超过上限的 PDF 不能内容识别或全文索引。
- fallback file key 可参与本地向量路径，但不是 `catalog.papers` 中的 canonical arXiv evidence；本阶段不扩展代表论文或 new-direction draft 契约。

## Approach

索引编排把 catalog 中的 unresolved PDF 合并为额外索引单元：必要时读取完整 PDF 计算内容摘要，同内容路径聚合到一个 paperKey；旧 `file:<observationFingerprint>` 文档迁移到内容 key 时复用 chunks/vectors。KB 文档和 manifest 保存 `contentHash`、`title`、`titleVersion`、路径与 observation fingerprints。查询将最大 chunk cosine、标题确定性相似度和正文 compact-token 命中取最大值；增量缓冲池聚类先把每篇论文的 centered chunks 压成归一化质心，漂移参考继续使用完整 chunk sets。插件命令、Dashboard Library matches 和 Find similar papers 的 Library 页使用同一展示 DTO，不渲染 chunk excerpt。

## Tasks

- [x] T1 契约与首轮实现：扩展 KB document/manifest schema，索引 unresolved PDF，提取首页标题并把 fallback 路径接入检索展示。Change kind: feature。Test strategy: 先为标题提取和 unresolved index/reuse/prune 建行为测试；预期 Red 为缺少 fallback 单元和 title 字段，Green 后跑 core/plugin 回归。
- [x] T2 L1 adjust — PDF 识别缓存失效：给内容识别器和 content-derived ready/unresolved catalog record 增加规则版本，规则变化时重验未变文件；filename-derived 记录保持观测复用。Change kind: bug fix。Observed Red: 聚焦集合中 identifier version 测试失败；Green: Planck 型旧 ready 记录转 unresolved，catalog strict round-trip 通过。
- [x] T3 L1 adjust — 稳定内容身份与迁移：fallback key 改为完整 PDF bytes SHA-256；同内容多路径合并，改名保持 key；旧 observation-key 文档原位重绑并复用 chunks/vectors；store clone 保留 `contentHash`/`title`/`titleVersion`。Change kind: bug fix + migration。Observed Red: 内容 key、rename、store clone、title refresh 共四类断言失败；Green: 聚焦 core 66/66。
- [x] T4 检索与增量计算：加入标题/正文 token/向量 max fusion，标题规则版本刷新只更新元数据；recluster 改用论文质心并保留完整 chunk 漂移参考。Change kind: behavior + performance fix。Verification: P8 core 定向 120/120，core 全量 1579/1579。
- [x] T5 Obsidian 表面一致性：命令、Dashboard Library matches、Find similar Library 页统一显示 title + fallback 相对路径或 arXiv paperKey + similarity，不显示 excerpt；索引 Notice 增 `titlesRefreshed`。Change kind: UI bug fix。Observed Red: Similar modal 仍泄漏内部 file key 和 passage；Green: UI 定向 15/15，plugin 全量 448/448。
- [x] T6 验收、报告与安装：双 typecheck、workspace boundaries、`git diff --check` 通过；lint 为 0 errors/69 warnings，因历史 `--max-warnings=60` 返回 1；core 全量按约束临时使用 8 GiB/单 fork 后还原配置。两次 technical-report handoff 均为 updated。该轮构建与安装产物 SHA-256 为 `05f01f9d6eb5ac4dba822bbdb6bf6ba108be6f7982a1a346cdc8747d5e294768`，随后由 T7 标题 v2 构建替换并保留为备份。
- [x] T7 真实 Obsidian 复验：修复版 Scan 已提交 revision 5，确认 legacy evidence 不再中止整轮，Planck 转 unresolved 且旧错误 arXiv paper 消失；全文索引稳定复用轮为 0 indexed / 307 reused / 3 failed / 0 pruned，内容 key 迁移与同内容路径合并已核对。真实 corpus 继续暴露 pdf.js 空 EOL marker 丢失导致 fallback 标题行粘连，以及多个 Pan-STARRS 标题命中统一 1.0 后按 hash 决胜；已修复 host 行界、升标题规则 v2，并把 title 命中改为 0.95 floor、允许正文频率超过 floor。用户确认 Dashboard 搜索栏 `panstarrs` top-5 已符合预期；Catalog Scan 已到 revision 7（161 ready / 144 papers / 214 unresolved / 1 failed）。Find similar papers 的 Library 页质量仍不达标：标题+摘要长查询曾被正文 token 融合干扰，且检索未做 corpus centering；已 L1 修复为检索默认 centering、仅短查询启用 token 融合、标题词法只取查询首段。MNRAS 页眉 `Advance Access publication …` 曾以黑名单过滤（v3），用户明确不接受该方向；已改为字体结构规则 v4（host 提供行级 font/position layout，core 按字号选标题，无页眉黑名单），376 文件语料验证 374 个标题正确。随后升 v5：PDF 文档元数据 `info.Title`（host getMetadata）过垃圾过滤 + HTML 实体解码后优先于字体规则，token 覆盖检查防丢字符的劣质 metadata；376 文件语料 32 处标题变化全部改善（Krause2017_1/Bulbul2024/Euclid 系列等 v4 盲区被 metadata 解决）。用户复验又发现 Find similar 里出现 "LSST science book version 2.0"：Chen2025.pdf 参考文献第一条（LSST Science Book, arXiv:0912.0201）位于内容流 23 前 512 字符，被误识别为 arXiv:0912.0201（与 Planck 同机制）；修复为标题搜索仲裁——direct ID + 文档标题并存时，标题搜索命中不同 ID 则采用搜索 ID，搜不到保留 direct ID，`PDF_IDENTIFICATION_EVIDENCE_VERSION` 升至 2（旧记录重验）。随后"作者入题"收口：v5 token 覆盖检查的全大写覆盖 metadata 回归（改严格超集，Team2026 等恢复正确大小写）与逗号分隔作者列表被扩展附加（NAME_PAIR_LIST 加逗号姓名对分支，Zhao2023 清除作者+摘要附加）；语料 parity 仅 Zhao2023 一处变化、零误伤。剩余已知形态：Benítez2000 类全大写无标记作者与标题同字号（信息边界）。收尾复验：测试 vault 全文索引 revision 30 `79 titles refreshed`（79 个 fallback 标题全部刷新，Beck2021 → "PS1-STRM: neural network source classification and photometric redshift catalogue for PS1 3 π DR1"）；Chen2025.pdf 识别修正为 `arxiv:2512.16010`（LSTM-MDNz，/arXivID 优先 + 识别版本 3），用户确认显示正确；用户确认测试 vault（Obsidian 运行时等价真实环境）复验通过并决定关闭 P8。真实 vault 插件仍为旧版，由用户自行更新。

## Verification

- 已观察：初始聚焦 Red 5 failed / 60 passed；修复后 core 聚焦 66/66、P8 core 120/120、plugin UI 15/15。
- 已观察：首次大范围验收 core 全量 95 files / 1579 tests、plugin 全量 32 files / 448 tests 全过；双 typecheck、workspace boundaries、`git diff --check` 通过；临时 core Vitest 配置已还原并与 `/tmp/vitest.config.mts.bak` 一致。
- 已观察：legacy ID 现场故障的预期 Red 为 core 1 failed / 22 passed、plugin 2 failed / 8 passed，三项均抛 `invalid arXiv ID: "astro-ph/0609591"`；修复后 core 23/23、plugin 10/10，P8/识别回归 core 128/128、plugin 25/25，全量 core 1580/1580、plugin 450/450。双 typecheck、boundaries、`git diff --check` 通过，lint 维持 0 errors/69 warnings；8 GiB/单 fork 临时配置再次还原。
- 已观察：最终构建产物与已安装 `main.js` SHA-256 均为 `05f01f9d6eb5ac4dba822bbdb6bf6ba108be6f7982a1a346cdc8747d5e294768`。
- 已观察：修复版真实 Scan 成功提交 revision 5：115 ready files / 101 papers / 214 unresolved / 47 failed / 0 unrelated / not truncated。Planck 已成为带 identification version 1 的 unresolved，`arxiv:1008.4686` 已从 catalog 消失；47 个 catalog failed 均为 `metadata-fetch-failed`，属于后续 Scan 可重试的 arXiv metadata 阶段，不是全文索引失败。
- 已观察：KB manifest revision 8 含 310 个单元（307 ready / 3 failed）；209 个 fallback key 中 206 ready 均有 PDF bytes `contentHash`，无 ready legacy observation-key；5 组同内容双路径已合并。Planck 为 `file:sha256:b4ac490d…` ready 文档、205 chunks，旧错误 arXiv 文档已消失。最终稳定复用轮 Notice 为 0 indexed / 307 reused / 3 failed / 0 pruned，因此按设计不触发自动方向更新进度。
- 已观察：3 个全文失败均为 scoped source 的 25 MiB 上限：`Engel2025.pdf` 40 MiB、`Hopp2026.pdf` 47 MiB、`Wu2023.pdf` 34 MiB；不影响其余 307 ready 文档。
- 已观察：Chambers PDF 的 pdf.js 首页 items 用空 `str` + `hasEOL: true` 表示行尾，旧 host 在 `if (!str) continue` 处丢弃 marker，使 Draft/Preprint/title/authors 粘成超长行并误选 `ABSTRACT ...`。旧规则下多个标题含 Pan-STARRS 的论文 token 分数均为 1.0，最终按 hash paperKey 决胜。修复后聚焦 core 86/86、plugin 33/33；全量 core 95 files / 1580 tests、plugin 33 files / 451 tests；双 typecheck、boundaries、`git diff --check` 通过，lint 保持 0 errors/69 warnings，临时 core 配置已恢复。
- 已观察：标题修复版构建与安装目标 SHA-256 均为 `b629ae82b4b91b817442344493666118f276b704b2263c7e89fde17a3bc104c5`；被替换版本备份 SHA-256 为 `05f01f9d6eb5ac4dba822bbdb6bf6ba108be6f7982a1a346cdc8747d5e294768`。
- 已观察：Catalog Scan revision 7 → Ready files 161 / Papers 144 / Unresolved 214 / Failed 1 / Unrelated 0 / Truncated No。
- 已观察：用户确认搜索栏 `panstarrs` 可给出预期 top-5；Find similar papers 的 Library 结果主题相关性仍不满意。
- 已观察：L1 similar-ranking 修复后聚焦 core 3 files / 51 tests、相关回归 9 files / 136 tests 全过；core/plugin typecheck 通过。新安装 `main.js` SHA-256 `d21147fe38d9b2070e0aa18dc2a941b432c59d3a298d75078cbc8d131a3f40ee`。
- 已观察：Find similar papers 截图显示多条 fallback 标题为 `Advance Access publication YYYY Month D`（MNRAS 页眉），污染展示与标题融合；v3 用页眉黑名单过滤该 banner，用户明确不接受该方向。
- 已观察：376 文件语料全量验证标题规则 v4（字体结构）：四个用户样例（Mucesh2021/Carrasco Kind2013/Beck2021/Luo2024）标题全部正确；A&A/Elsevier 刊头、arXiv stamp、页边刊头、节标题大于标题、老论文作者同字号断行、Euclid/eROSITA 系列续行、下标合并、预印本编号/DOI 行、小写专名标题（redMaPPer/dustmaps）等反例全部通过；374 个文件得到标题，2 个无文本层 null（正确兜底）。已知局限 3 个文件（Benítez2000 全大写作者粘连、Young2016 类型标签前缀、Bulbul2024 系列标题大写续行截断）。聚焦 core 103/103、plugin 25/25；全量 core 95 files / 1597 tests、plugin 33 files / 453 tests；双 typecheck、boundaries、`git diff --check` 通过；lint 维持 0 errors/69 warnings；core 全量临时 8 GiB 堆配置已还原并与 `/tmp/vitest.config.mts.bak` 一致。构建产物与安装目标 SHA-256 均为 `f036b75ea16da28a20b90aa15fba72cdeef39a83b08c435640b84389a9623bfb`（v3 版备份 `27b1e0b6003aecb3a8633d5ca16bf7b7f5dce93f463bb0f359e9e91f2d34900e`）。
- 待观察：重启后跑全文索引，确认 titles refreshed 使四个样例显示真实标题（不再有 Advance Access 假标题）；再打开 2602.01548 的 Find similar papers，核对 Library similar 标题与主题相关性；通过后关闭 P8。
- 已观察（收尾）：Obsidian 环境提取失败两处根因定位并修复——①`getDocument` 缺 `standardFontDataUrl`/`cMapUrl`（Obsidian 内置 pdf.js 5.3.34 对非嵌入/CID 字体抛 UnknownErrorException，pdfjs-dist 6.2.108 不抛），补 Obsidian 资源路径；②getDocument 把 ArrayBuffer transfer（detach）给 worker，v7 起的 `rawInfoTitle` 读 detached buffer 抛 `Cannot perform Construct on a detached ArrayBuffer`（79 个失败文件 = getMetadata 无 Title 走 rawInfoTitle 的文件），改为传 bytes 拷贝；另 `page.cleanup()` 异常不再传播。测试 vault 全文索引 revision 30：`0 indexed / 355 reused / 3 failed / 0 pruned, 79 titles refreshed` 全成功；KB manifest 221 个 fallback 全 titleVersion 8。识别根因：识别提取只扫 content streams、从不读 Info dict `/arXivID`，Chen2025.pdf 第一个命中是参考文献引用 LSST Science Book 的 DOI（arXiv.0912.0201）→ 错误关联；修复为 `/arXivID`（或 `/arXiv`）→ stream 头 → XMP 的信任顺序，`PDF_IDENTIFICATION_EVIDENCE_VERSION` 2→3 强制重识别；真实文件验证提取 2512.16010，用户重 Scan 后 catalog `Chen2025.pdf → arxiv:2512.16010`，显示 LSTM-MDNz 标题。测试：core 识别相关 3 files / 39 tests、plugin 33 files / 460 tests（core 全量在本机 vitest worker OOM，环境问题，改动为叶子模块）。最终安装到测试 vault 的构建 SHA-256 `46defddf0d16ee81d352501d9e4bc55819bd51b88a81030f9f912da3b1e35997`。用户确认测试 vault 复验通过并决定关闭 P8（真实 vault 插件由用户自行更新）。

## Abort / reshape triggers

- Planck 重 Scan 后仍为 `arxiv:1008.4686`，说明生产插件或 evidence-version 接线未生效，停止索引验收并先核对加载产物与 catalog record。
- 内容未变的改名文件重新 embedding 或出现并存的 legacy observation-key 文档，说明迁移/复用契约未达成。
- 全文索引出现新的整轮中止、manifest/paper 不一致或大量超出 25 MiB 边界之外的 failed，先核对磁盘状态再决定 L1/L2。
- `panstarrs` 仍非预期 top-1，保留实际排名与三路分数证据后重新分类检索问题，不以调阈值掩盖。

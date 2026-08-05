# P6 dogfood run kit (2026-08-04)

Goal: 用真实文献库跑一遍完整个人化链路，对比"手动 topics 单独选"与"手动 topics ∪ 已确认文献库方向"的发现，找出至少一篇**只被文献库方向选中**、研究者认可值得看的论文。

## 准备（已完成）

- `scripts/personalization/validate-real-library.ts`：真实文献库 headless 验证已通过（526 文件 → 25 ready / 498 unresolved / 3 unrelated / 0 failed，24 篇论文，重扫 100% 复用，52ms）。
- `scripts/personalization/analyze-dogfood.ts`：产物分析脚本（读 `papers.json` 的 occurrence provenance / novelty，产出 manual-only / library-only / both 对比）。
- 插件 production build 已从本 worktree 构建：`plugin/main.js`（655kb）、`manifest.json`、`styles.css`。
- 已复制进 **plugin_test vault**（`/home/tiandc/Documents/code/arxiv-daily/plugin_test`）的插件目录；旧 `main.js` 备份为 `main.js.bak-20260804`。该 vault 已配置好 endpoint / API key / categories / topics（Photo-z、Galaxy Cluster），无需再填。

## 用户在 Obsidian 里的步骤

1. **打开 plugin_test vault**：Obsidian → 左下角切换 vault 到 `arxiv-daily/plugin_test`（若插件未加载，重启 Obsidian 或重新启用插件）。
2. **连接文献库**：插件设置 → Personal library → Select directory → 选 `~/Nextcloud/work/Article`。
3. **Scan library**（不需要授权即可跑）：识别策略 v2 后预期 ready ≈ 46 / unresolved ≈ 40 / papers ≈ 44（86 个文件中 46 个可识别：22 文件名 + 21 PDF 正文页眉 + 标题搜索补足；剩余 40 个是扫描版 PDF 无文本层，物理限制，保持 unresolved 不阻塞）。扫描会读取 PDF 受限元数据/文本层做识别，并对有标题无 ID 的文件向 arXiv API 发标题搜索。
4. **授权**：按 modal 确认 scope = `~/Nextcloud/work/Article`、depth = metadata-and-abstract、endpoint = 当前配置的 endpoint（lan 上的 baseUrl），点击授权。
5. **Review directions**：命令面板 → "Review personal library directions"；检查 Proposed 候选，修正名称/描述/cues/代表论文后**确认**（至少确认 1 个方向）。
6. **跑日报**：Dashboard → Run Today（或 Run Pending）。等运行完成，日报出现。
7. **把结果告诉我**：运行完成后我来跑分析脚本，给出对比结论；你复核其中 library-only 的论文是否值得看。

## 注意

- 全程不需要我碰 API key；授权状态保存在你的 vault 里。
- 若 endpoint 不可达或授权 modal 有任何异常，先停下来告诉我，不要强行继续。
- 日报数据会写进你选的 vault（真实 vault 会经 Nextcloud 同步）——选真实 vault 前确认可接受。

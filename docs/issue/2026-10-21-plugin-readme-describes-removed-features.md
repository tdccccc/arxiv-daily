# 插件 README 和新手教程描述了已不存在的功能：日报里的"关注/重点"复选框、API 密钥显示为 Configured、回溯天数/高级项/文件类型等设置

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 OBS-07
> 范围：`plugin/README.md`、`plugin/README.zh-CN.md`、`docs/getting-started.md`、`docs/getting-started.zh-CN.md` · 严重程度：低 · 状态：未修复 · 置信度：代码推断（按代码和提交历史逐条核对，未在真实 Obsidian 里点验）

## 问题

下面三处文档和当前代码不符，用户按文档找不到对应功能：

1. **"Daily selections / 每日勾选"整节**（`plugin/README.md:74-83`、`plugin/README.zh-CN.md:74-81`）说每篇论文有两个可在 Markdown 里勾选的复选框，勾选后会同步回论文索引。现在的日报根本不写这两个复选框，勾选这件事也没有任何代码在监听。提交 `fab53e5`（"stop writing markdown status controls"）已让新日报不再写复选框，`3caa94c`（"stop legacy checkbox sync"）已拆掉运行时同步；`DailySelectionSyncService` 现在只剩定义和测试，插件、CLI、DSH 都没有实例化它。星标/标记的唯一入口是 Dashboard 和命令（`Set paper mark…`）。
2. **API 密钥"只会显示为 `Configured`"、"用 Replace / Clear 修改"**（`plugin/README.md:19`、`plugin/README.zh-CN.md:19`、`docs/getting-started.md:30`、`docs/getting-started.zh-CN.md:30`）。设置页的密钥输入框是带"显示/隐藏"按钮的密码框（`plugin/src/settings/sensitive-input.ts`），点眼睛图标就能看到完整密钥；整个插件没有"Configured"、"Replace"、"Clear"这些文案。"只会显示为 Configured"还是一条安全承诺，现在不成立。
3. **"设置概览"表**（`plugin/README.md:91-102`、`plugin/README.zh-CN.md:91-102`）列出"回溯天数（≤ 5）"、"可处理的文件类型、处理深度"、"Advanced：请求延迟、缓存保留时长、分段/字符上限、日志级别"。设置页里没有这些项：回溯窗口固定为 5 天（常量 `LOOKBACK_DAYS`），文献库行只有选文件夹/索引/撤销授权和嵌入设置，Advanced 只有"日志级别"。

## 原因

- 功能先后被移除（见上面两个提交），README 与教程没有同步删除。
- 设置界面改过之后没有回头核对"设置概览"表。

## 可能的解决方案

推荐：

1. 删除"Daily selections / 每日勾选"整节，并在 Dashboard 一节说明星标按钮和 `Set paper mark…` 命令是标记论文的入口。
2. 密钥那几句改成如实描述："保存在本机插件数据中，输入框默认遮蔽，可点眼睛图标查看"，去掉"只会显示为 Configured"和 Replace/Clear。
3. 设置概览表按现有设置页重写（LLM、arXiv categories、Research topics、Output & schedule 只列实际存在的项、Personal library、Email delivery、Advanced 只写日志级别）。

## 修复后应补的测试

- 文档改动不需要代码测试；可以加一条文档检查，列出 README 里提到的设置名，确认它们都出现在 `packages/core/src/settings/schema.ts` 的业务设置清单里。

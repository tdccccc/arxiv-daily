# 插件的 data.json 不是合法 JSON 时，插件整个加载失败：打不开设置，也没有备份可恢复，只能手动删文件并丢掉全部配置

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 OBS-08
> 范围：Obsidian 插件启动、设置持久化 · 严重程度：低 · 状态：未修复 · 置信度：代码推断（插件一侧已用临时测试复现；`loadData()` 遇到非法 JSON 会抛错，是按 Obsidian 的文档与已知行为推断，未启动 Obsidian 验证）

## 问题

`data.json`（在 `.obsidian/plugins/arxiv-daily/` 下）保存全部插件设置：LLM 与邮件的 API 密钥、研究主题、输出目录、个人文献库连接与授权、文献库提案的接受记录。它的内容经过 `settingsAndStateFromPersistedData` 等多层容错（字段缺失、类型不对、路径不合法都会回退到默认值并给出警告），但文件本身不是合法 JSON 时没有任何容错：`onload()` 的第一步 `await this.loadData()` 抛出 `SyntaxError`，`onload` 整体失败。

用户看到的是 Obsidian 报"Failed to load plugin arXiv Daily"，插件整个不可用：没有设置页、没有命令、没有 Dashboard，所以无法在界面里修复。恢复的唯一办法是手动编辑或删除 `data.json`；删除就意味着丢掉所有设置和密钥，要重新走一遍引导。插件的其他持久文档（运行状态、论文索引、文献库目录）都有临时文件加备份的保护，唯独这个最重要的配置文件没有任何第二份副本，也不会把损坏的文件改名保留。

可能的触发：写入中途被强制终止、同步工具（iCloud/Dropbox/Syncthing/Git）合并出冲突标记或只同步了一半、手工编辑出错。

临时测试（已删除）：`loadData` 抛 `SyntaxError("Unexpected end of JSON input")` 时，`await plugin.onload()` 以同一个 `SyntaxError` 拒绝；此时 `logger` 都还没有创建。

## 原因

- `plugin/main.ts:408-409`、`2505-2531`：`onload` 的第一个 await 是 `loadSettingsAndState()`，里面直接 `await this.loadData()`，没有 try/catch。
- `plugin/main.ts:2843-2860` `persistSettings` 每次用 `saveData` 整体覆盖 `data.json`，不保留上一份。

## 可能的解决方案

推荐：`loadSettingsAndState` 捕获 `loadData()` 的异常。捕获后：把原文件复制为 `data.json.corrupt-<时间戳>`（通过 `app.vault.adapter` 读取原文），用默认设置继续启动，并弹出一条明确的 Notice 告诉用户配置文件损坏、原文件保存在哪里；同时在启动日志里记录。

备选：`persistSettings` 在覆盖前先把现有的 `data.json` 复制成 `data.json.bak`（仅在内容有变化时），加载失败时优先尝试备份。

## 修复后应补的测试

- `loadData` 抛 `SyntaxError`：插件仍能完成 `onload`，使用默认设置，损坏文件被保留，出现提示。
- 正常的 `data.json` 与缺失文件的现有行为不变。

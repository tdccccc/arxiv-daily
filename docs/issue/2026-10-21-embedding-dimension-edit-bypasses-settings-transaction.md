# 设置页里改"Embedding dimension"绕过了设置事务：索引状态不重新检查，保存失败不回滚，填错的值悄悄丢弃

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 OBS-06
> 范围：Obsidian 插件设置页（远程嵌入的维度输入框；Obsidian 低于 1.13 的旧设置页里还有"Embedding API key"输入框同类写法） · 严重程度：低 · 状态：未修复 · 置信度：已复现

## 问题

设置页里其他所有设置项都通过 `SettingsChangeService`（候选值校验、串行保存、失败时给出旧值），唯独"Embedding dimension"直接改内存里的设置对象，再调用 `saveSettings()`：

1. **文献库索引状态不重新检查。** 其他嵌入字段（模式、Base URL、模型）保存成功后会调用 `plugin.refreshLibraryIndexTrace()`，发现"索引是用另一个维度建的"，把文献库行标成需要重新准备，并禁用审阅入口。改维度时不会。结果是：文献库行仍显示"已索引 N 篇"，"Review suggestions"可点，Dashboard 的文献库搜索也照常发起，直到第一次查询才被核心拒绝（远程模型的标识里含维度，查询时报"knowledge base was built with model … rebuild the index before searching"）。代码里 `refreshLibraryAfterEmbeddingChange` 的键列表明明包含 `embedding.dimension`，现有单元测试（"rechecks the library only after saving embedding.dimension"）也通过，但测试走的是 `tab.changeSettingValue("embedding.dimension", 768)`，真实的输入框根本不走这条路，所以测试没有保护到真实行为。
2. **保存失败不回滚。** 磁盘写入失败时，设置页弹出"save embedding dimension failed"，但内存里的维度已经是新值，输入框也显示新值。之后在这次会话里建索引/搜索都按这个未保存的值运行；重启 Obsidian 后悄悄变回旧值。
3. **非法输入没有任何反应。** 输入 0、负数、小数或非数字时，`change` 事件什么都不做：没有提示，输入框保留非法内容，设置不变。

实测（临时测试，已删除，真实的 `SettingsChangeService` 和 `ArxivDailySettingTab`，渲染真实的 `renderEmbeddingDimensionRow`）：

- 把 1536 改成 3072：持久化被调用，`settings.embedding.dimension = 3072`，`refreshLibraryIndexTrace` 调用 0 次。
- 持久化抛错"disk full"：`settings.embedding.dimension` 仍是 2048，输入框 2048。

## 原因

- `plugin/src/settings/declarative-rows.ts:866-874`（1.13+ 的行）：`change` 回调里 `tab.plugin.settings.embedding.dimension = parsed; await tab.plugin.saveSettings();`，不经过 `tab.changeSettingValue`，所以既没有 `refreshLibraryAfterEmbeddingChange`（`plugin/src/settings/tab.ts:366-370`），也没有 `SettingsChangeError` 的恢复值；`parsed` 不合法时整个回调静默返回。
- `plugin/src/settings/tab.ts:1966-1975`（旧设置页）：同样的直接赋值；同文件 1936-1941 的"Embedding API key"文本框在每次按键的 `onChange` 里 `s.embedding.apiKey = v.trim(); await this.plugin.saveSettings();`，既每个字符保存一次，又没有 try/catch，保存失败时是未处理的 Promise 拒绝。
- `plugin/src/settings/change-service.ts:85-104` `persistCurrent` 的注释说明它是为"复杂编辑器已经改过内存对象"而设，不会恢复编辑前的快照，所以失败时没有可回滚的东西。

## 可能的解决方案

推荐：维度输入框和旧设置页的 API key 输入框都改成 `tab.changeSettingValue("embedding.dimension", parsed)` / `("embedding.apiKey", next)`（API key 用与其他密钥一致的 `renderSensitiveInput`，失败时恢复）。非法输入时设置 `setCustomValidity` 并在输入框旁提示"Enter a positive whole number"（与"Daily paper limit"行一致）。

备选：保留现有写法，但在保存成功后补调 `refreshLibraryIndexTrace()`，保存失败时手动还原旧值；不推荐，等于把事务重新实现一遍。

## 修复后应补的测试

- 用真实的行渲染：改维度成功后 `refreshLibraryIndexTrace` 被调用一次。
- 持久化失败时，内存里的维度和输入框都回到旧值，并出现错误提示。
- 输入 0 / "abc" / 2.5 时出现校验提示，设置不变。

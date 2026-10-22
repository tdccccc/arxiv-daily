# 插件的 data.json 不是合法 JSON 时，插件悄悄按默认设置启动，下一次保存就把原文件覆盖：没有提示，也没有备份，设置和密钥无声丢失

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 OBS-08
> 范围：Obsidian 插件启动、设置持久化 · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-21-batch2`，提交 ddecf07、94b87d2、9a3c99c） · 置信度：已按 Obsidian 1.11.5 桌面版随包的 `app.js` 核对读取代码（未启动 Obsidian 实测）；最初按“`loadData()` 遇到非法 JSON 会抛错”推断，已证明有误

## 问题

`data.json`（在 `.obsidian/plugins/arxiv-daily/` 下）保存全部插件设置：LLM 与邮件的 API 密钥、研究主题、输出目录、个人文献库连接与授权、文献库提案的接受记录。它的内容经过 `settingsAndStateFromPersistedData` 等多层容错（字段缺失、类型不对、路径不合法都会回退到默认值并给出警告），但文件本身读不出来（不是合法 JSON）时没有任何处理。

**更正。** 本文最初按“Obsidian 的 `loadData()` 遇到非法 JSON 会抛出 `SyntaxError`，`onload` 整体失败，用户看到 Failed to load plugin arXiv Daily”写成。这是推断，并且是错的（见下面“原因”里 Obsidian 的代码）。实际情况是：`loadData()` 不抛错，对读不出的文件只返回 `undefined`，插件把它当成“还没有设置”，悄悄按默认设置启动，没有任何提示，设置页看起来像第一次使用。之后只要有任何一次保存（改任何一个设置、接入文献库等），插件就用默认设置加这次的修改把 `data.json` 整个覆盖，唯一的一份设置和密钥就此丢失，没有副本可以找回。这比最初写的“插件无法加载、需要手动处理”更糟，因为用户不会意识到有东西坏了。

可能的触发：写入中途被强制终止、同步工具（iCloud/Dropbox/Syncthing/Git）合并出冲突标记或只同步了一半、手工编辑出错。

复现：让 `loadData()` 对一份不完整的 `data.json` 返回 `undefined`（和 Obsidian 一样），`loadSettingsAndState()` 正常完成并使用默认设置，没有副本，也没有提示。最初的临时测试让 `loadData` 抛 `SyntaxError`，那条路径在真实的 Obsidian 里不会出现。

## 原因

- Obsidian 1.11.5 桌面版随包的 `app.js`（`/opt/Obsidian/resources/obsidian.asar` 里）：`Plugin.loadData()` 就是 `this.app.vault.readPluginData(this.manifest.dir)`，后者读 `<dir>/data.json`，走 `Vault.readJson`。整理成可读形式如下：

  ```js
  readJson(path) {
    try {
      return JSON.parse(await this.adapter.read(path));
    } catch (e) {
      if (e.code === "ENOENT") return null;           // 文件不存在：首次运行
      console.error("failed to read JSON", path, e);  // 其它读取失败或解析失败：只打日志
    }
    return undefined;                                  // 不抛错
  }
  ```

  所以 `loadData()` 对不存在的文件返回 `null`，对读不出或解析不了的文件返回 `undefined`，从不抛错。
- `plugin/main.ts` 的 `loadSettingsAndState` 直接把 `loadData()` 的结果交给 `settingsAndStateFromPersistedData`；它对非对象一律返回默认设置且不给警告，于是 `undefined` 和“首次运行”没有区别。
- `persistSettings` 每次用 `saveData` 整体覆盖 `data.json`，不保留上一份。插件的其他持久文档（运行状态、论文索引、文献库目录）有临时文件加备份的保护，唯独这个最重要的配置文件没有。
- 附带发现（未处理）：同一份代码里 `Vault.writeJson` 在写入失败时什么也不做（`catch` 为空），所以 `saveData()` 即使没写成也会正常返回，插件无法靠它得知保存是否成功。

## 可能的解决方案

> 本节和下一节按最初的推断（`loadData()` 会抛错）写成，保留作记录；实际的做法和测试见“修复情况”。

推荐：`loadSettingsAndState` 捕获 `loadData()` 的异常。捕获后：把原文件复制为 `data.json.corrupt-<时间戳>`（通过 `app.vault.adapter` 读取原文），用默认设置继续启动，并弹出一条明确的 Notice 告诉用户配置文件损坏、原文件保存在哪里；同时在启动日志里记录。

备选：`persistSettings` 在覆盖前先把现有的 `data.json` 复制成 `data.json.bak`（仅在内容有变化时），加载失败时优先尝试备份。

## 修复后应补的测试

- `loadData` 抛 `SyntaxError`：插件仍能完成 `onload`，使用默认设置，损坏文件被保留，出现提示。
- 正常的 `data.json` 与缺失文件的现有行为不变。

## 修复情况

`data.json` 读不出来时，插件不再悄悄按默认设置启动、又在下一次保存时把它覆盖。Obsidian 的 `loadData()` 对读不出的文件返回 `undefined`（文件不存在才返回 `null`，仍是首次运行）；现在遇到 `undefined` 而 `data.json` 存在时，插件自己再读一遍：读得出且是合法 JSON，就照常使用（Obsidian 那一次只是临时读失败，不提示）；不是合法 JSON，就复制成同目录下的 `data.json.corrupt-<该文件最后修改的时间>`（文件名不含冒号，Windows 也能用），按默认设置启动，弹出提示并在启动日志里记一行，写明副本的位置；读不出来，或副本写不出来，插件说明原因并拒绝启动，这样就没有机会用默认设置覆盖 `data.json`。原来的 `data.json` 原样留着，所以也可以直接在原文件里修好；之后任何一次保存会覆盖它，但副本此时已经存在。同一个坏文件反复启动只留一份副本，提示仍每次出现，直到文件恢复可读。代价：文件修好之前插件跑在默认设置上，需要从副本里找回设置（里面有明文 API key）；以前悄悄用默认设置启动的“读不出来”（例如权限问题）现在会让插件不启动。没有采用写作方案里备选的“保存前备份成 `.bak`”。尚未处理：顶层不是对象的合法 JSON（例如 `[]`）仍然悄悄回退到默认设置；`saveData()` 写失败时不会报错（见“原因”）。

测试：`corrupt-data-json.test.ts`（18 条，按 Obsidian 的真实行为构造：`loadData()` 对坏文件返回 `undefined`、对不存在的文件返回 `null`、从不抛错）。覆盖：坏文件用默认设置并留下副本、提示内容、文件名合法、重复启动只留一份、文件变了另存一份、取不到修改时间仍留副本、空文件也留副本；Obsidian 第一次读失败而第二次读成功时照常使用；读不出来、副本写不出来时插件说明原因并拒绝启动；文件消失按首次运行；宿主以解析错误拒绝时同样处理，其它错误照旧抛出；正常和缺失的 `data.json` 不触碰目录；真实 `onload` 在坏文件下走完且不保存任何东西，在读不出也留不下副本时根本不启动。

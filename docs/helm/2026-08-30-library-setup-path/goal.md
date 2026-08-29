# 文献库上手路径（library setup path）

status: active
updated: 2026-08-30
owner: claude-code-main-session

## Intent

让第一次使用个人文献库的人能在设置里跟着走完：选文件夹 →（远程才授权）→ 建索引 → 去 Dashboard 搜索。不要再靠命令面板猜下一步。

## Success criteria

- [x] 未选文件夹时，文献库这一行只提示选文件夹。
- [x] 已选文件夹且为本地嵌入时，主按钮是建索引；不要求先授权。
- [x] 已选文件夹且为远程嵌入时，主按钮先是授权；授权后变成建索引。
- [x] 建索引可从设置页启动，进度和结果有通知；不必先打开命令面板。
- [x] 授权文案区分本地（不离开本机）和远程（全文会发到嵌入接口）。

## Non-goals

- 段落摘录、页码跳转、真按篇增量重建。
- 重做整个 Settings 或 Dashboard。
- 改检索排序或索引格式。
- 问答 / Agent。

## Constraints

- 本地嵌入继续不需要 model-processing 授权；远程嵌入继续要求 full-text 授权。
- 不扩大 `ScopedLibrarySource` 权限面。
- 不修改并行 active Helm（发现闭环、email relay v2）。
- 每个行为块有 Red/Green 证据。

## Phases

1. P1 — 设置页文献库一行给出当前下一步（选文件夹 / 授权 / 建索引） — status: done
2. P2 — 建索引过程与完成后的下一步（去搜索）可感知 — status: active

## Open questions

- 设置分组顺序（Embedding 在文献库前还是后）留到 P2，P1 只改文献库这一行的主按钮与说明。

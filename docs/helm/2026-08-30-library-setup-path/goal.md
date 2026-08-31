# 文献库上手路径（library setup path）

status: done
updated: 2026-08-31
owner: claude-code-main-session

## Intent

让第一次使用个人文献库的人能在设置里跟着走完：选文件夹 →（远程才授权）→ 建索引 → 去 Dashboard 搜索。不要再靠命令面板猜下一步。

## Success criteria

- [x] 未选文件夹时，文献库这一行只提示选文件夹。
- [x] 已选文件夹且为本地嵌入时，主按钮是建索引；不要求先授权。
- [x] 已选文件夹后主按钮永远是建索引；远程授权改为切到远程时就地确认，不再是 Library 行上的独立按钮。
- [x] 建索引可从设置页启动，进度和结果有通知；不必先打开命令面板。
- [x] 建索引期间 Library 行显示当前阶段与 `Indexing… (N/M)`，并能就地取消。
- [x] 索引跑完后 Library 行常驻一句「上次建索引的时间 + 可搜到多少篇」，不随通知一起消失。
- [x] 授权文案区分本地（不离开本机）和远程（全文会发到嵌入接口）。
- [x] Personal library 默认只露出选库、嵌入方式、可选 PDF parser；远程字段和 sidecar URL 按需展开，按钮不挤成一团。
- [x] 文献库这一行最多三个按钮、没有二级菜单；预览 / 扫描 / 重载目录 / 方向复审只从命令面板进。
- [x] 切到远程嵌入时就地弹全文披露：确认即完成切换与授权，取消则回到本地、什么都不改。
- [x] 已授权后接口指纹变化会重新询问；取消恢复成已授权的接口值。
- [x] Revoke 先确认，确认后同时撤销授权并切回本地嵌入，不留「远程但未授权」的死状态。
- [x] 遗留的远程未授权配置不卡死：主按钮仍是建索引，点击时先弹披露，取消则不建索引也不改状态。

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
2. P1b — Personal library 设置精简（少选项、不挤） — status: done
3. P1c — 去掉 Manage 菜单，Library 行最多三个按钮 — status: done
4. P1d — 远程授权改为切换时就地确认，Revoke 即切回本地 — status: done
5. P2 — 建索引的过程、取消与结果都落在 Library 行上 — status: done

## Open questions

（无。P2 收尾时确认：不加 Search library 按钮——搜索时本来就看得到，完成提示够了。）

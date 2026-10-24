# Obsidian 保存设置失败可能被宿主吞掉，界面误报成功

状态：已修复（分支 `fix/2026-10-23-followups`，提交 ad714ed）

> 来源：2026-10-21 修复交接中的额外发现，2026-10-23 复核；不计入原 54 条。
> 严重程度：中 · 置信度：宿主代码已核对，尚未模拟真实磁盘故障

## 问题与依据

`plugin/main.ts` 的 `persistSettings()` 只等待 `this.saveData(data)`。本机 Obsidian 的 `app.js` 中，`Plugin.saveData → Vault.writePluginData → Vault.writeJson` 最终调用 adapter.write；`writeJson` 的 catch 消费异常后正常返回。因此磁盘满或权限失败时，上层设置事务不一定收到拒绝，无法回滚和提示。

证据：只读解析 `/opt/Obsidian/resources/obsidian.asar` 内 `app.js`，核对 `prototype.saveData`、`prototype.writePluginData`、`prototype.writeJson`，未改动应用或真实设置。该结论仅针对核对过的宿主实现，其他 Obsidian 版本需重验。

## 建议和验收

在插件的持久化边界验证写入结果或采用能传播错误的适配器方法，并保持外部设置变更通知兼容。用模拟“saveData 正常返回但文件未改变”的宿主验证：保存失败必须提示、内存回滚，旧文件保留。补真实宿主故障注入验证，勿在用户 Vault 上测试。

## 修复情况

插件现在不再依赖 Obsidian 自带的 `saveData`（它会吞掉写入错误），而是自己通过库的文件接口写出同样格式的 JSON，所以磁盘满或权限失败时，保存会真正失败：设置事务拒绝、内存里的修改回滚、设置页弹出已有的失败提示，原来的 `data.json` 不会被换成新内容。已核对 Obsidian 的 `app.js`：`saveData` 只额外记录一个修改时间用于“外部修改”通知，而本插件没有实现 `onExternalSettingsChange`，所以跳过它没有影响。代价：这是对宿主行为的绕开，依赖库接口的 `write`；文件系统本身没有原子写，写到一半失败时磁盘上的文件仍可能被截断，这一点没有额外处理。测试：`save-data-write-errors.test.ts` 5 条（宿主吞错的行为说明、写失败时设置变更被拒绝并回滚且旧文件保留、API 密钥同样回滚、直接保存报错、正常写入内容不变）；未做真实 Obsidian 的磁盘故障注入。

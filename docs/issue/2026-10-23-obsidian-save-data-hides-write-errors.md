# Obsidian 保存设置失败可能被宿主吞掉，界面误报成功

状态：未修复

> 来源：2026-10-21 修复交接中的额外发现，2026-10-23 复核；不计入原 54 条。
> 严重程度：中 · 置信度：宿主代码已核对，尚未模拟真实磁盘故障

## 问题与依据

`plugin/main.ts` 的 `persistSettings()` 只等待 `this.saveData(data)`。本机 Obsidian 的 `app.js` 中，`Plugin.saveData → Vault.writePluginData → Vault.writeJson` 最终调用 adapter.write；`writeJson` 的 catch 消费异常后正常返回。因此磁盘满或权限失败时，上层设置事务不一定收到拒绝，无法回滚和提示。

证据：只读解析 `/opt/Obsidian/resources/obsidian.asar` 内 `app.js`，核对 `prototype.saveData`、`prototype.writePluginData`、`prototype.writeJson`，未改动应用或真实设置。该结论仅针对核对过的宿主实现，其他 Obsidian 版本需重验。

## 建议和验收

在插件的持久化边界验证写入结果或采用能传播错误的适配器方法，并保持外部设置变更通知兼容。用模拟“saveData 正常返回但文件未改变”的宿主验证：保存失败必须提示、内存回滚，旧文件保留。补真实宿主故障注入验证，勿在用户 Vault 上测试。

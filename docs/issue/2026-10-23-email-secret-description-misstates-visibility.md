# 邮件密钥说明声称保存后不可查看，与显示按钮不符

状态：未修复

> 来源：2026-10-21 修复交接中的额外发现，2026-10-23 复核；不计入原 54 条。
> 严重程度：低 · 置信度：界面实现已核对

## 问题与依据

`packages/core/src/settings/schema.ts` 的 `emailApiKey.description` 写着 “not shown again after you save”。插件的 `renderSensitiveInput()` 实际保留掩码值并提供 Show/Hide，工作台也支持确认后显示已保存密钥。

## 建议和验收

文案应说明密钥保存在本机、默认隐藏但可主动查看。核对两端英文及中文说明和显示按钮，不改变既有密钥存储行为。

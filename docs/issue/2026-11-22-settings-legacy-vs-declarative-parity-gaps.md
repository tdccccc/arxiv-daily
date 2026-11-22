# Obsidian 1.13+ 声明式设置页与旧版设置页存在三处功能脱节（已废弃 Sidecar 仍暴露于旧版、模板与自定义分类缺失于新版）

状态：未修复

> 来源：2026-11-22 main `2557b90a` 代码审查（Obsidian 设置页双实现一致性）
> 严重程度：低 · 置信度：代码推断

## 问题与依据

插件目前同时维护两套设置界面：Obsidian 1.13+ 的声明式设置（`plugin/src/settings/definitions.ts` + `declarative-rows.ts`）和 `<1.13` 的命令式旧版设置（`plugin/src/settings/tab.ts` `renderLegacySettings()`）。两套实现存在双向功能不一致：

1. **旧版设置页仍展示完全失效的 `"Better PDF parser"`（Local Parser Sidecar）配置**：
   - 根据 ADR 0013 与 `plugin/main.ts:1621-1635`，文献库索引只抽取标题与摘要并直接使用 PDF.js，**不再探测也不再调用 PDF parser sidecar**（`plugin/tests/fulltext-index-lifecycle.test.ts:307-325` 有明确断言）。
   - 1.13+ 声明式设置（`plugin/src/settings/definitions.ts:189-190`、`plugin/tests/settings-definitions.test.ts:239,481`）已专门隐藏了所有 sidecar 控件。
   - 但旧版 `plugin/src/settings/tab.ts:1960-2015` 仍然完整渲染 `"Better PDF parser"` 开关以及 `"Sidecar capability URL"` / `"Sidecar parse URL"` 两个 URL 输入框，误导 `<1.13` 用户配置并维护一个实际完全不生效的外部服务。
2. **1.13+ 声明式设置页缺少旧版已有的“快速开始主题模板”和“自定义 arXiv 分类输入”**：
   - 旧版 `tab.ts:1676-1692` 提供了基于 `TOPIC_TEMPLATES` 的快速填充研究主题模板下拉框，方便新用户一键套用示例主题；在 1.13+ `definitions.ts` / `declarative-rows.ts` 中没有任何对应入口。
   - 旧版 `tab.ts:1606-1620,1648-1660` 在主分类和附加分类旁提供了 `"Or enter custom category"` 自定义输入框，支持输入预设列表以外的合法 arXiv 子分类（例如细分子领域）；在 1.13+ `declarative-rows.ts:278-311` 中只有固定下拉框，用户若需要预设之外的 arXiv 分类只能手工编辑 `data.json`。

## 原因

1.13+ 声明式设置和后续 ADR 0013 清理分别在各自文件上迭代，缺少两套 UI 之间的功能对账清单（`tab.ts` 残留了已下线的 sidecar 块，而 `definitions.ts` 迁移时漏掉了自定义分类和主题模板）。

## 建议和验收

- 从旧版 `plugin/src/settings/tab.ts:1960-2015` 移除已失效的 `"Better PDF parser"` sidecar 渲染块（与 1.13+ 声明式设置及 ADR 0013 对齐，并同步更新 `plugin/tests/settings-tab.test.ts` 中对应的旧 UI 测试）。
- 明确 1.13+ 声明式设置是否保留自定义 arXiv 分类输入框与主题模板（建议至少补齐自定义 arXiv 分类输入，以免非预设领域用户在 Obsidian 1.13+ 上无法配置自己的 arXiv 分类；若确定精简主题模板，可在文档或代码注释中明确记录并清理旧版冗余）。

## 修复后应补的测试

- `plugin/tests/settings-tab.test.ts`：断言旧版设置页不再渲染 `"Better PDF parser"` / sidecar URL 控件。
- `plugin/tests/settings-declarative-tab.test.ts`：如补齐自定义分类输入，断言 Obsidian 1.13+ 设置页可添加并保存预设外的合法 arXiv 分类。

# 主题设置校验在抓完 arXiv 列表之后才做，配置错误时白费一次抓取

状态：未修复

> 来源：2026-10-31 main `e0498c94` 代码审查（会话内直接审查，静态推断，未运行）
> 严重程度：低 · 置信度：代码推断

## 问题与依据

`packages/core/src/pipeline/pipeline.ts:242` 先执行 `fetchPapersForDate`（大分类可能要读多页 /recent 并补摘要），275-278 行才调用 `validateTopicsForReport` 并返回 `failed_permanent`。注释说"在第一次模型调用之前"，但抓取本身也是代价，而且主题校验不依赖抓取结果。

## 建议和验收

把主题校验挪到 `dailyExists` 检查之后、抓取之前。行为不变，只是失败更早、不发请求。

## 修复后应补的测试

pipeline 测试：主题设置非法时 `listForDate` 未被调用，结果仍为 `failed_permanent`。

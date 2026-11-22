# 个人文献库方向预览与主题整理未剥离 Markdown JSON 代码围栏，且方向预览无纠错重试

状态：未修复

> 来源：2026-11-22 main `2557b90a` 代码审查（个人文献库 LLM 响应解析）
> 严重程度：低 · 置信度：代码推断

## 问题与依据

2026-10-31 审查后（见 `docs/issue/2026-10-31-review-index.md`），日报流水线中的筛选（`paper-filter.ts`）和精读选择器（`detail-selector.ts`）均已统一支持剥离单层 ```` ```json ... ``` ```` 代码围栏并带有纠错重问。但个人文献库的两条 LLM JSON 解析路径仍直接调用裸 `JSON.parse(raw)`：

1. `packages/core/src/library/personal-library-direction-preview.ts:50-56`：
   ```ts
   const raw = await input.llm.call(request.messages, { ... });
   throwIfCancelled(input.signal);
   let value: unknown;
   try { value = JSON.parse(raw); } catch { throw new Error("Invalid direction preview response"); }
   ```
   - 既不剥离单层 Markdown 代码围栏，也没有任何格式纠错重试；只要模型习惯性输出 ```` ```json\n[...]\n``` ````，用户在文献库审核弹窗中点击预览研究方向就会直接报错 `"Invalid direction preview response"`。
2. `packages/core/src/library/personal-library-topic-organization.ts:74-79`：
   ```ts
   try {
     value = JSON.parse(raw);
   } catch {
     return { ok: false, reason: "not-json" };
   }
   ```
   - 同样未剥离单层 Markdown 代码围栏，会无谓消耗一次 LLM 纠错重试或最终导致整次主题整理失败。

## 原因

这两处个人文献库模块未复用 `packages/core/src/pipeline/` 中已有的单层 JSON 代码围栏剥离逻辑（例如 `stripSingleJsonCodeFence`），方向预览也未复用筛选器的纠错重试策略。

## 建议和验收

- 将单层 Markdown 代码围栏剥离抽为共享工具并在 `personal-library-direction-preview.ts:53` 与 `personal-library-topic-organization.ts:76` 中复用（仅接受单层标准 ```` ```json ... ``` ```` 围栏，拒绝夹杂额外前后说明文本，保持原有严格边界）。
- 可选为 `previewPersonalLibraryDirection` 增加 1 次格式纠错重试，与 `paper-filter.ts` 一致。
- 验收：模型返回被 ```` ```json ... ``` ```` 包裹的合法结果时，方向预览与主题整理均能一次解析成功。

## 修复后应补的测试

- `packages/core/tests/personal-library-direction-preview.test.ts` 与 `packages/core/tests/personal-library-topic-organization.test.ts`：
  - 返回带单层 ```` ```json ... ``` ```` 围栏的合法 JSON 时成功解码；
  - 返回附带多余说明文字或损坏 JSON 时仍拒绝。

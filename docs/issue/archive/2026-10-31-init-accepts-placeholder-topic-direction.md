# CLI init 允许把占位说明文字当成研究方向写入配置，并把"之后再改"说成可选

状态：已修复（分支 `fix/2026-10-31-review`）

> 来源：2026-10-31 main `e0498c94` 代码审查（会话内直接审查，静态推断，未运行）
> 严重程度：中 · 置信度：代码推断

## 问题与依据

`apps/cli/src/init.ts`：

- 251-254 行：主题描述的默认值是一段说明文字 "Describe in natural language what papers belong in this topic (problems, methods, objects; what to exclude)."，主题名默认 "My research (edit me)"；
- 778-790 行：这一步用 `initialValue` 预填这段说明文字，回车即采用；输入为空时又回退到另一句说明文字；
- 1253-1258 行：原样写入 TOML 的 `description`；`apps/cli/src/config.ts:353-359` 把它变成该主题唯一的研究方向；
- `validateFilterConfig` 只检查方向非空（`packages/core/src/settings/validation.ts:146`），所以放行；
- 332-339 行：结尾提示 "Optional: refine [[arxiv.topics]] in the config"。

用户按默认走完向导、运行 `arxiv-daily run --today`，模型拿这段说明文字去给论文归类——得到一份"今天没有相关论文"或随机匹配的日报，日报小节标题还是 "My research (edit me)"，没有任何地方解释为什么。

## 原因

用 `initialValue` 承载提示文字；完成提示没有区分"用户写了方向"和"还是占位文字"。

## 建议和验收

- 提示文字改放 `placeholder`，`initialValue` 留空；校验拒绝空值和占位原文，要求用户写一句真实的研究方向（plain 模式同样处理）。
- 若仍想允许跳过，结尾文案必须明确"先编辑主题再生成第一份日报"，不能写 Optional；主题名同理。

## 修复后应补的测试

`apps/cli/tests` 的 init 测试：

- 方向一步直接回车 → 看到要求填写的校验提示，不能进入下一步；
- 输入真实描述 → 写入 TOML；
- 结尾文案在方向仍为默认时不再出现 "Optional"。

## 修复情况

init 的"研究方向"一步不再预填说明文字（示例改为灰色 placeholder），回车留空或仍填旧占位句子都会被拒绝并提示写自己的方向；主题名默认改为 "My research"；步骤前的说明写明"只有命中方向的论文才会进日报"；结尾提示改为"需要时在 `[[arxiv.topics]]` 添加更多主题或方向"，不再说 Optional。

测试：`apps/cli/tests/cli-init.test.ts` 新增"要求真实的研究方向"（空串和占位句子都被重问，最终写入真实描述，结尾不含 Optional）；原来对该提示一律回答空串的用例补了真实答案。

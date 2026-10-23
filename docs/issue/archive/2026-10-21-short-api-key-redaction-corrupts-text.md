# API 密钥是 `local`、`none`、`test` 这类短词时，工作台和 CLI 输出会把正文里的同一个词替换成 `[REDACTED]`，设置自动保存还会把它永久写进研究方向

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 CLI-02
> 范围：CLI 阅读工作台所有 JSON 响应、CLI 的 stdout/stderr、设置自动保存 · 严重程度：中 · 状态：已修复（分支 `fix/2026-10-21-batch3`，提交 8dd06c4） · 置信度：已复现

## 问题

用本地模型服务（Ollama、LM Studio、vLLM 等）的用户必须在 `[llm] api_key` 里填一个非空的占位值（`init` 会拒绝空密钥），常见写法是 `ollama`、`lm-studio`、`EMPTY`、`none`、`local`、`dummy`、`test`。工作台和 CLI 会把配置里长度不小于 4 的所有密钥当作"秘密"，对每一个 JSON 响应和每一段输出做全文子串替换。占位密钥恰好是一个普通英文词时，正文里出现同一个词的地方都会被改写：

- 工作台读到的文档标题和正文：`api_key = "local"` 时，标题 `A local approach to calibration` 在 `api/documents`、`api/document`（标题、元数据、渲染后的 HTML、标题锚点 `id`）里都变成 `A [REDACTED] approach to calibration`，正文 `Local methods work locally on non-local data.` 变成 `Local methods work [REDACTED]ly on non-[REDACTED] data.`。论文标题、摘要、总结、日历消息等其他 JSON 响应同理。
- `arxiv-daily status`、`arxiv-daily papers` 的 stdout 同样被替换，依赖这些 JSON 的脚本和伴生扩展拿到的是被改写的数据。
- 设置里 `embedding.mode` 这类枚举值也会被替换（`"mode":"[REDACTED]"`，此时下拉框退回第一个选项）。

更严重的是设置自动保存会把被替换的文本写回磁盘。设置表单显示的是被替换后的方向文本，表单每次保存都会提交全部主题和方向。用真实界面（Chrome）复现：

```toml
api_key = "local"
...
[[arxiv.topics]]
name = "Cosmology"
description = "photometric redshift estimation in the local universe"
```

打开设置，只把"每日论文上限"改成 25，自动保存之后 `config.toml` 变成：

```toml
[[arxiv.topics]]
id = "topic-1"
name = "Cosmology"
tag = "cosmo"
description = "photometric redshift estimation in the [REDACTED] universe"
detail = false

[[arxiv.topics.directions]]
id = "cli-direction-202eed34a0da1370"
origin = "migrated"
text = "photometric redshift estimation in the [REDACTED] universe"
```

用户的研究方向被永久改写，之后每日筛选拿着"`[REDACTED] universe`"去匹配论文，且没有任何提示。主题名称、方向、标签里出现密钥子串的都会受影响。触发条件窄（占位密钥是词，并且恰好出现在主题文字里），但一旦触发就是静默且永久的。

## 原因

- `packages/core/src/utils/redaction.ts:15-17`：`redactText` 对每个密钥执行 `text.split(secret).join("[REDACTED]")`，没有词边界，也不区分密钥所处的位置；`redaction.ts:62` 只要求长度不小于 4。
- `apps/cli/src/workbench/server.ts:72-75`：`json()` 对整个响应体 `redact(JSON.stringify(value))`，包括文档、论文、设置这类数据负载，而不只是错误信息和运行输出。`server.ts:63` 的 `rememberSecrets` 还会在每次保存设置时把密钥重复追加到同一个数组里（数组只增不减）。
- `apps/cli/src/main.ts:125-129`：CLI 的 stdout/stderr 包装层对所有输出做同样的全文替换。
- `apps/cli/src/workbench/web/settings.ts:175`：设置表单把 textarea 里显示的文本当作用户输入原样提交；`apps/cli/src/workbench/settings-adapter.ts:25-66` 把它当作新值写进 TOML。

## 可能的解决方案

推荐：把"防密钥泄露"限制在会出现密钥的通道，不再对数据负载做子串替换。

1. `server.ts` 的 `json()` 只对错误信息（`error` 字段）和运行输出（`run.output`）做 `redactText`，文档、论文、设置、日历的数据负载不替换。设置接口本来就不返回密钥字段（只有 `apiKeyConfigured`），`api/settings/secret` 是用户显式查看。
2. `redactText` 的子串替换只用于足够长、足够像密钥的值（例如长度不小于 8），更短的值只在"赋值形态"（`api_key=…`、`Bearer …`、JSON 里 `"api_key":"…"`）里替换，这些规则已经存在。
3. CLI 的 `status` / `papers` 这类数据输出不走替换层，或者只替换错误流。

备选：保持现在的全文替换，但设置表单保存前拒绝提交含 `[REDACTED]` 的文本，至少避免写盘。这样仍然会在界面上显示被改写的正文。

## 修复后应补的测试

- `api_key = "local"`，文档标题和正文含 `local`：`api/documents`、`api/document`、`api/papers` 原样返回。
- 同样配置下，设置表单读到的方向文本不含 `[REDACTED]`；改一个无关设置并保存后，`config.toml` 里的方向文本不变。
- 密钥确实出现在错误信息或运行输出里时仍被替换。
- `arxiv-daily status` 在占位密钥是普通词时输出原样的主题描述。

## 修复情况

密钥是 `local`、`none`、`test`、`ollama` 这类占位词时，正文不再被改写。`redactText` 只豁免明确列出的本地服务占位词（`local`、`none`、`test`、`ollama`、`EMPTY`、`dummy`、`lm-studio`、`not-needed`，忽略大小写）；其他非空真实密钥无论长度都会打码。占位词在 `api_key=…`、`Bearer …`、JSON 里的 `"api_key"` 等敏感赋值形态中仍会被打码。多个密钥使用单次替换，且保留已有 `[REDACTED]` 标记，避免短密钥再次改写脱敏结果。工作台的 JSON 响应也不再整体替换：只对 `error`、`message`（日历里的运行错误）和 `output`（运行输出）三个字段脱敏，文档、论文、状态、设置原样返回，所以设置表单读到的方向文本不会带 `[REDACTED]`，自动保存也不会再把它写进 `config.toml`；`embedding.mode` 之类的枚举值同样恢复。`arxiv-daily status` / `papers` / `library` 的结构化输出也只处理诊断字段，真实短密钥不会破坏 JSON 语法、字段名和文献正文；日志对占位密钥原样输出，工作台记住的密钥列表也不再在每次保存设置时重复追加。取舍：占位词不会在普通正文中被替换；已经被旧版本写坏的方向文本（含 `[REDACTED]`）不会自动恢复，需要手动改回；`Bearer …`、`token: …` 这类赋值形态的规则仍对 CLI 输出生效（与密钥长度无关，不在本次范围内）。

测试：`packages/core/tests/redaction.test.ts`（占位词不改写、赋值形态仍替换、1–8 个字符真实密钥、正则特殊字符、多个密钥与重复脱敏、占位词与真实密钥并存）；`apps/cli/tests/cli-inspect.test.ts`（`status` / `papers` 原样输出）；`apps/cli/tests/workbench-redaction.test.ts`（`api/documents`、`api/document`、`api/papers`、`api/settings`、`api/status` 原样返回；设置表单改一个无关设置并保存后 `config.toml` 里的方向文本不变；真实密钥仍从持久化的运行错误、日历消息和分块写入的运行输出里被遮住）。

收尾补验：短真实密钥在持久化错误、日历消息和分块运行输出中仍被打码；单字符密钥与研究方向文字重叠时，设置表单保存仍保留原文且不写入 `[REDACTED]`。新增核心测试在修复前 10 项失败，修复后通过。

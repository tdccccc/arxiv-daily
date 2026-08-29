# P7 — 远程嵌入可选开关

goal_ref: ../goal.md
updated: 2026-08-10

## Outcome

ADR 0008 落地：远程嵌入作为可选开关——core 提供 OpenAI 兼容的 `RemoteEmbeddingModel`（经 HostAdapters.http），插件侧有独立嵌入设置区（provider/baseUrl/apiKey/model/mode，含 CLI 配置映射），授权记录扩展为多端点（嵌入端点指纹 + full-text 深度 + 授权 modal 披露），库首次准备时引导选择本地/远程（切换需重建提示），索引/检索按 mode 选择嵌入实现。接受远程的用户首次索引从小时级降到分钟级；本地路径保持默认与隐私语义不变。

## Assumptions

- 远程嵌入契约收敛为 OpenAI 兼容 `POST {baseUrl}/embeddings`（`{model, input}` → `{data:[{embedding}]}`）——主流供应商（OpenAI、BGE 系 API、硅基流动等）通用。
- `EmbeddingModel` port 已 host-neutral；远程实现放 core（复用 http port，与 LlmClient 同模式），CLI 未来可复用。
- 授权指纹扩展：`libraryAuthorizationFingerprint` 在远程模式时纳入嵌入端点 baseUrl；`LIBRARY_PROCESSING_DEPTH` 扩为 `"metadata-and-abstracts" | "full-text"`；深度升级与端点变更都触发重新授权（现有机制）。
- 切换模式 = 模型切换 → KB manifest modelId 守卫自然要求重建（现有机制，引导中告知）。

## Approach

T1 先做 core `RemoteEmbeddingModel`（批量、超时、abort、错误分类、modelId/dimension 声明）+ 测试；T2 设置层（core PluginSettings.embedding 段 + 默认/校验 + CLI 映射 + 插件设置 UI 行）；T3 授权扩展（深度 union + 多端点指纹 + 授权 modal 披露嵌入端点 + 升级流程）；T4 首次引导选择（库设置/首次索引准备时选模式 + 重建提示）；T5 工厂接线（index/search/placement 按 mode 构建，模型切换守卫生效）；T6 收尾验收。

## Tasks

- [x] T1 core `RemoteEmbeddingModel`：OpenAI 兼容 embeddings 客户端（批量 ≤64、超时、AbortSignal、HTTP 错误分类、`modelId`/`dimension` 声明）+ core 测试（mock http）—— 端口增 `prefixPolicy`（e5|none）编排条件化前缀；+9 测试；commit 1d154a4
- [x] T2 设置：`PluginSettings.embedding`（mode `local|remote`、provider/baseUrl/apiKey/model/dimension + initialChoiceDone）+ 默认值 + 校验（validateEmbeddingConfig）+ CLI `[embedding]` 映射 + 插件设置 UI Embedding 区；+4 校验 + CLI 映射测试；commit ed34343
- [x] T3 授权扩展：`LIBRARY_PROCESSING_DEPTHS` union + full-text 档；授权 scope 化（LLM 端点 + 可选嵌入端点），指纹含嵌入端点 digest；授权 modal 披露嵌入端点与全文深度；decode 兼容两种深度；+2 远程授权测试；commit 14bbf76
- [x] T4 首次引导选择：`chooseLibraryRoot` 成功后 `offerEmbeddingModeChoice`（initialChoiceDone 一次；modal 含速度/隐私/重建提示，关闭默认本地）；+3 测试；commit aa82875
- [x] T5 工厂接线：`buildEmbeddingModel` 按 mode 选择（本地 transformers vs remote）；`assertRemoteEmbeddingReady` 门禁（远程需完整配置 + full-text 授权）；index/search 接入，诊断保持本地；+5 测试；commit 8041e8b
- [x] T6 收尾：core 1547/1547、plugin 447/447、tsc/lint/boundaries 全绿；technical-report handoff（T1-T5 均 updated：全文机制段、设置形状、授权段、嵌入宿主段）；每阶段提交完成；goal.md P7 done + status done

## Verification

- T1：mock http 返回标准 /embeddings 响应 → 向量形状/维度正确；超时/4xx/网络错误分类；abort 生效。
- T2：设置读写、校验（apiKey 必填当 remote）、CLI 配置加载 embedding 段。
- T3：授权指纹在远程模式下含嵌入端点；端点变更/深度升级后状态变 authorization-invalidated；modal 披露两行端点。
- T4：首次准备库时出现模式选择；选择后生效；提示含"切换需重建索引"。
- T5：remote 模式下索引/检索走 RemoteEmbeddingModel（mock http 断言调用）；local 模式行为不变（现有测试全过）。
- T6：core 1534+ / plugin 437+ 全绿、lint 0、boundaries OK；handoff 到 `updated` 或 `no-impact`；每阶段提交。

## Abort / reshape triggers

- 若实测供应商 /embeddings 契约差异大（非 OpenAI 兼容）：L2——收敛为 OpenAI 契约并文档化支持范围，不逐家适配。
- 若 CLI 配置映射复杂化：嵌入段 CLI 侧先不映射（CLI 保持本地默认，文档说明），插件侧不受阻。
- 若 full-text 深度引入的授权流程改动过大：先做设置+远程模型（本地授权语义不变），深度升级流程单独小步推进。

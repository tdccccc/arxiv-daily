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

- [ ] T1 core `RemoteEmbeddingModel`：OpenAI 兼容 embeddings 客户端（批量 ≤64、超时、AbortSignal、HTTP 错误分类、`modelId`/`dimension` 声明）+ core 测试（mock http）
- [ ] T2 设置：`PluginSettings.embedding`（mode `local|remote`、provider/baseUrl/apiKey/model/dimension?）+ 默认值 + 校验 + CLI `[embedding]` 映射 + 插件设置 UI 行 + 测试
- [ ] T3 授权扩展：`LIBRARY_PROCESSING_DEPTH` 扩为 union + full-text 档；`libraryAuthorizationFingerprint` 纳入嵌入端点（远程模式）；授权 modal 披露嵌入端点与全文深度；深度升级/端点变更重新授权 + 测试
- [ ] T4 首次引导选择：库准备流程中选嵌入模式（本地/远程，含速度/隐私说明与切换重建提示）+ 测试
- [ ] T5 工厂接线：`indexPersonalLibraryFullText`/`searchPersonalLibraryFullText`/增量按 `settings.embedding.mode` 构建嵌入模型（本地 transformers vs core RemoteEmbeddingModel）；modelId 随配置声明 + 测试
- [ ] T6 收尾：core/plugin 全量测试、tsc、lint、boundaries；technical-report handoff（每被接受 chunk）；每阶段提交；goal.md P7 done + status done

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

# P4 — Automatic Refresh and Integrations

> Phase ID：P4
> 依赖：P1 Reference Library、P2 Profile Agent、P3 Personalized Recommendation
> 对应目标：根据文献库变化和 Dashboard 当前标星状态，批量更新画像，并完善 Zotero/JabRef 和大型文献库使用体验。
> 核心约束：单次文件变化或 Star 操作不立即调用 LLM。

## 1. 交付结果

完成后：

1. Personalized 日报运行前会检查文献库和标星状态；
2. 文献新增、修改、删除后，系统只处理变化内容；
3. 当前星标集合变化后，下一次批量刷新会更新画像；
4. 未发生变化时不会重新构建画像或消耗 LLM；
5. 刷新失败时继续使用上一份有效画像；
6. Zotero/JabRef 用户可通过常见导出文件稳定接入；
7. 桌面端可选支持只读外部目录；
8. 设置和诊断可看到刷新状态、数量和错误。

## 2. 范围

### 2.1 包含

- `refresh-state.json`；
- catalog revision 比较；
- 当前星标集合 fingerprint；
- dirty reason；
- 手动 Refresh；
- Personalized run 前自动检查；
- 自动刷新节流；
- P1 增量 scan + P2 profile build 串联；
- 失败保留旧 profile；
- Better BibTeX/JabRef export 使用体验；
- 可选 desktop external folder/file；
- 大型库性能、诊断和恢复测试。

### 2.2 不包含

- 每次 Star 立即调用 LLM；
- 正负反馈事件日志；
- 根据打开次数学习；
- 时间衰减模型；
- Zotero SQLite；
- 必须运行 Zotero local API；
- PDF 全文解析；
- embeddings；
- exploration；
- 后台无限循环 Agent。

## 3. RefreshState

路径：

```text
arxiv-daily/.index/personalization/refresh-state.json
```

模型：

```ts
interface RefreshState {
  schemaVersion: number;
  lastCatalogRevision: number;
  lastStarFingerprint: string;
  dirty: boolean;
  dirtyReasons: Array<
    "library_changed" |
    "stars_changed" |
    "manual_request" |
    "profile_missing"
  >;
  lastCheckedAt?: string;
  lastAttemptAt?: string;
  lastSuccessAt?: string;
  lastError?: string;
}
```

原则：

- state 只记录当前比较结果，不记录每次点击事件；
- error 文本长度受限；
- 成功刷新后清除 dirty reasons；
- 失败保留 dirty=true；
- profile build 成功后再更新已处理 revision/fingerprint；
- 原子保存；
- future schema 不自动覆盖。

## 4. 变化检测

### 4.1 文献库

复用 P1：

1. 扫描 enabled sources；
2. fingerprint 未变化的文件复用；
3. 生成 next catalog；
4. catalog 内容变化时 revision 递增；
5. 与 RefreshState.lastCatalogRevision 比较。

变化语义：

- 新增文献：画像证据新增；
- 修改文献/笔记：画像证据更新；
- 删除文献：证据撤销；
- 来源删除：撤销该来源的 refs；
- 目录移动：若 canonical document 相同，文献本体不重复；
- 不把任何删除解释为负反馈。

### 4.2 Dashboard 标星

每次 refresh check 直接读取 `PaperIndexStore.listStarred()`：

- 取稳定排序的 arXiv IDs/canonical keys；
- 计算 fingerprint；
- 与 lastStarFingerprint 比较；
- 新标星和取消标星都只表示证据集合变化；
- 不需要修改 Dashboard 每个 Star handler；
- batch star 和 command star 自动被包含；
- fingerprint 只包含 `listStarred()` 返回的文献标识，不编码 to_read/saved/ignored 等状态值。

这种设计避免事件漏记，也避免 Dashboard、commands、batch actions 多处埋点。

## 5. 自动触发策略

首版只实现两个触发点：

### 5.1 手动刷新

命令和设置按钮：

- Scan library and refresh profile；
- 可取消；
- 忽略自动节流；
- 显示 scan/build 结果。

### 5.2 Personalized run 前检查

在 scheduler/manual run 进入 Personalized pipeline 前：

1. 调用 `RefreshCoordinator.ensureFresh()`；
2. 增量扫描来源；
3. 计算星标 fingerprint；
4. 无变化则直接返回；
5. 有变化则构建新 profile；
6. 成功后继续 P3；
7. 失败时记录错误并继续使用旧 profile；
8. 没有旧 profile 时触发 P3 Manual fallback。

不新增独立定时器，避免与现有 scheduler 重叠。每日推荐前检查已经覆盖主要自动学习场景。

### 5.3 节流

为避免 scheduler retry 重复消耗：

- 同一时间只允许一个 refresh；
- 自动检查在短时间内复用同一个 Promise/result；
- 同一 catalog revision + star fingerprint 不重复 build；
- 上一次 build 失败后设置最小自动重试间隔；
- 手动 Refresh 可以绕过间隔；
- 取消不计为成功。

节流参数先使用代码常量，不增加设置项。

## 6. RefreshCoordinator

建议接口：

```ts
interface RefreshCoordinator {
  check(): Promise<RefreshCheckResult>;
  ensureFresh(signal?: AbortSignal): Promise<RefreshResult>;
  refreshNow(signal?: AbortSignal): Promise<RefreshResult>;
}
```

流程：

```text
load refresh state
→ scan reference library
→ load catalog revision
→ load current stars
→ compare processed state
→ unchanged: return
→ changed: build profile
→ success: save refresh state
→ failure: keep old profile and dirty state
```

返回结果至少区分：

- unchanged；
- refreshed；
- skipped_manual_mode；
- cancelled；
- failed_using_previous_profile；
- failed_no_profile。

## 7. P3 接入

在 P3 RecommendationContext 解析前调用 refresh hook：

- Manual mode：跳过自动扫描和画像刷新；
- Personalized mode：ensureFresh；
- refreshed/unchanged：加载当前 profile；
- failed_using_previous_profile：记录 warning，继续；
- failed_no_profile：走 Manual fallback；
- cancellation：整个当前 run 返回 cancelled。

刷新逻辑不写进 `pipeline.ts` 各阶段。推荐由 main/service orchestration 在 build/run 前完成，保持 pipeline 只消费 context。

## 8. Zotero/JabRef 导入完善

P1 已支持通用 BibTeX/RIS/CSL JSON。P4 只提升管理器工作流，不引入数据库耦合。

### 8.1 Source presets

添加来源时提供可选 preset：

- Generic BibTeX；
- Better BibTeX export；
- JabRef BibTeX；
- RIS；
- CSL JSON。

Preset 只影响：

- 推荐扩展名；
- parser compatibility mode；
- attachment 字段识别；
- UI 帮助文本。

Preset 不影响兴趣归纳，不读取 collection/group 作为兴趣方向。

### 8.2 自动导出文件

对于用户设置为自动导出的 `.bib`/`.json`：

- P1 fingerprint 检测文件变化；
- P4 自动 refresh 时重新解析变化文件；
- entry key 稳定时复用未变化记录；
- 导出文件暂时半写入/解析失败时保留旧 catalog，并在下次重试；
- 帮助文档说明 Zotero Better BibTeX 和 JabRef 自动导出的配置方式。

### 8.3 Attachment

- 解析 export 中的 attachment/file 字段；
- attachment 只用于关联 PDF 和标识符；
- 不复制或修改附件；
- 路径不存在时 warning；
- attachment 路径不发送给 LLM。

### 8.4 明确不做

- 不直接读取 `zotero.sqlite`；
- 不要求 Zotero 正在运行；
- 不把 collection/group 直接当兴趣；
- 不在首版写回 Zotero/JabRef。

## 9. 桌面外部目录（可选子任务）

如果实际 dogfood 的文献库位于 Vault 外，增加只读 external source：

```ts
kind: "external_folder" | "external_file"
```

实现边界：

- 仅 Obsidian desktop/CLI；
- mobile 显示 unsupported；
- 每个 source 只允许访问用户明确配置的 root/file；
- 使用独立 `ExternalReferenceReader`，不扩宽通用 StorageAdapter 的写权限；
- 只提供 list/stat/readText/readBinary；
- 禁止 write/remove/rename；
- 规范化并阻止 path escape；
- sources.json 可以保存绝对路径，但日志/诊断默认脱敏；
- LLM evidence 永远不包含绝对路径。

该子任务可以在 P4 内根据 dogfood 是否需要决定是否交付，但接口应预留。

## 10. 大型文献库

目标场景：

- 1,000 条文献；
- 5,000 条文献；
- 多个导出文件和附件；
- 大部分内容不变化。

优化：

- stat/fingerprint 快速跳过；
- export entry 级缓存；
- metadata resolver 只处理新增/变化 IDs；
- profile batch cache 只更新变化 documents；
- 并发有上限；
- 批次间响应取消；
- UI 进度显示 scan/parse/resolve/profile；
- 避免在 Obsidian UI thread 做长同步解析；
- 统计本次 reused/parsed/resolved 数量；
- 日志不打印每篇文献全文。

不在 P4 引入向量数据库。

## 11. 设置、命令和诊断

### 11.1 设置页

增加：

- Refresh profile now；
- Last checked；
- Last successful refresh；
- Library changed / Stars changed 状态；
- Last error；
- Reused/changed/unresolved counts；
- source preset；
- 外部目录平台可用性（若实现）。

### 11.2 命令

- `arXiv Daily: Refresh personalized profile`；
- `arXiv Daily: Show personalization status`；
- `arXiv Daily: Retry failed reference imports`。

### 11.3 诊断

扩展 diagnostics：

- mode；
- source count/kinds；
- catalog revision/document counts；
- profile generatedAt/interest count；
- dirty state/reasons；
- last refresh result；
- unresolved/error count；
- external paths 脱敏。

不得输出：

- 完整路径（默认）；
- title/abstract/note 全文；
- profile prompt；
- API key。

## 12. 建议代码结构

### 12.1 新建

- `plugin/src/personalization/refresh-store.ts`
- `plugin/src/personalization/refresh-coordinator.ts`
- `plugin/src/personalization/star-fingerprint.ts`
- `plugin/src/personalization/source-presets.ts`
- `plugin/src/personalization/external-reference-reader.ts`（可选）
- `plugin/src/personalization/status-view.ts`

### 12.2 修改

- `plugin/src/personalization/reference-library-service.ts`
- `plugin/src/personalization/profile-agent.ts`
- `plugin/src/personalization/source-store.ts`
- `plugin/src/personalization/types.ts`
- `plugin/src/core/adapters.ts`（仅在外部 reader 需要 capability type 时）；
- `plugin/main.ts`
- `plugin/src/services/scheduler.ts` 或 build/run orchestration；
- `plugin/src/settings/tab.ts`
- `plugin/src/commands.ts`
- `plugin/src/services/diagnostics.ts`
- CLI runtime/config（若支持外部目录）；
- tests、mocks 和文档。

## 13. 实施任务

### Task 1：RefreshStore

- [ ] schema；
- [ ] load/save atomic；
- [ ] empty state；
- [ ] dirty reasons 去重；
- [ ] success/failure transitions；
- [ ] error 截断；
- [ ] future schema；
- [ ] output path reload。

### Task 2：Star fingerprint

- [ ] 使用 listStarred；
- [ ] 稳定 canonical keys；
- [ ] 排序和 hash；
- [ ] 星标顺序变化不改变 fingerprint；
- [ ] add/remove 改变 fingerprint；
- [ ] status 变化但 priority 不变时 fingerprint 不变；
- [ ] 空集合稳定。

### Task 3：RefreshCoordinator check

- [ ] 扫描 catalog；
- [ ] 获取 revision；
- [ ] 获取 star fingerprint；
- [ ] 比较 refresh state；
- [ ] dirty reasons；
- [ ] unchanged 快速返回；
- [ ] 同时调用去重；
- [ ] cancel。

### Task 4：Profile build 串联

- [ ] dirty 时调用 P2；
- [ ] build success 后提交 processed state；
- [ ] build failure 保留旧 state/profile；
- [ ] profile missing 自动 dirty；
- [ ] manual refresh bypass throttle；
- [ ] metrics/log；
- [ ] 不在 star action 内调用 LLM。

### Task 5：Personalized run hook

- [ ] Manual 跳过；
- [ ] Personalized pre-run ensureFresh；
- [ ] existing profile fallback；
- [ ] no profile Manual fallback；
- [ ] cancellation；
- [ ] scheduler retry 不重复 build；
- [ ] manual run 与 scheduled run 共用逻辑。

### Task 6：管理器 presets

- [ ] Better BibTeX；
- [ ] JabRef；
- [ ] RIS；
- [ ] CSL JSON；
- [ ] attachment parsing；
- [ ] 自动导出半写入恢复；
- [ ] UI help；
- [ ] docs。

### Task 7：外部目录（若纳入 P4）

- [ ] desktop capability detection；
- [ ] read-only scoped reader；
- [ ] external folder/file source；
- [ ] path escape tests；
- [ ] mobile unsupported；
- [ ] diagnostics redaction；
- [ ] no write methods。

### Task 8：性能

- [ ] 1k/5k fixture benchmark；
- [ ] unchanged scan reuse；
- [ ] changed subset parse；
- [ ] batch cache reuse；
- [ ] cancellation latency；
- [ ] UI responsiveness；
- [ ] memory ceiling observation；
- [ ] log/diagnostic metrics。

### Task 9：状态 UI 和诊断

- [ ] dirty badge/status；
- [ ] last check/success/error；
- [ ] Refresh now；
- [ ] counts；
- [ ] source preset；
- [ ] error detail without sensitive content；
- [ ] diagnostics tests。

## 14. 测试

建议新建：

- `plugin/tests/personalization/refresh-store.test.ts`
- `plugin/tests/personalization/star-fingerprint.test.ts`
- `plugin/tests/personalization/refresh-coordinator.test.ts`
- `plugin/tests/personalization/personalized-refresh-integration.test.ts`
- `plugin/tests/personalization/source-presets.test.ts`
- `plugin/tests/personalization/external-reference-reader.test.ts`（若实现）
- `plugin/tests/personalization/large-library.test.ts`

关键场景：

- catalog unchanged + stars unchanged；
- library add/modify/delete；
- star add/remove；
- same star set different order；
- refresh build success；
- refresh build failure with old profile；
- refresh failure without profile；
- two simultaneous personalized runs；
- scheduler retry；
- manual mode skip；
- Better BibTeX automatic export changes；
- half-written export；
- 5k docs mostly unchanged；
- cancellation；
- external path escape/mobile unsupported。

## 15. 验证

```bash
cd plugin
npx vitest run tests/personalization/refresh-coordinator.test.ts
npx vitest run tests/personalization/personalized-refresh-integration.test.ts
npm test
npm run build
```

手工场景：

| 场景 | 预期 |
|---|---|
| 文献库无变化 | Personalized run 不调用 profile LLM |
| 新增一篇文献 | 只解析新增内容并刷新画像 |
| 删除一篇文献 | 撤销证据，不产生负兴趣 |
| Dashboard 新标星 | 下一次 refresh 增加该论文权重 |
| 取消标星 | 下一次 refresh 撤销额外权重 |
| profile refresh 失败 | 使用上一份 profile 继续推荐 |
| Better BibTeX 自动导出 | 文件变化后自动增量导入 |
| 5k 文献第二次扫描 | 大部分记录复用 |
| Manual mode | 不自动扫描或重建画像 |
| 外部目录在 mobile | 明确提示不支持，不影响插件 |

## 16. 完成标准

- [ ] 文献库变化可触发批量画像刷新；
- [ ] 星标集合变化可触发批量画像刷新；
- [ ] 单次 Star/文件变化不立即调用 LLM；
- [ ] 无变化时不重复 build；
- [ ] 刷新失败保留旧 profile；
- [ ] Personalized run 前自动 ensureFresh；
- [ ] Manual mode 不触发个性化刷新；
- [ ] Better BibTeX/JabRef 常见导出可稳定使用；
- [ ] 大型库未变化内容可复用；
- [ ] diagnostics 不泄露文献内容和路径；
- [ ] 全量测试和 production build 通过；
- [ ] 用户可从“配置来源”到“自动推荐”完成完整闭环。

## 17. 文档

交付时同步：

- README 个性化推荐说明；
- Reference Library 设置指南；
- Zotero Better BibTeX 自动导出示例；
- JabRef 导出示例；
- 隐私说明；
- 常见错误：未解析 PDF、导出半写入、profile build 失败；
- Manual 回退说明；
- 移动端限制；
- release checklist 对新增状态文件的检查。

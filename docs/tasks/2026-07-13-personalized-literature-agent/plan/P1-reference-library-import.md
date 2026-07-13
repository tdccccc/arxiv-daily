# P1 — Reference Library Import

> Phase ID：P1
> 依赖：无
> 对应目标：把用户指定的一个或多个文献来源，转换为统一、可增量更新的 Reference Catalog。
> 本 Phase 不调用 LLM，不生成兴趣画像，也不改变现有日报流程。

## 1. 交付结果

完成后，用户可以：

1. 添加一个或多个 Vault 内目录；
2. 选择是否递归扫描子目录；
3. 添加 BibTeX、Better BibTeX、RIS 或 CSL JSON 导出文件；
4. 手动执行扫描并查看结果；
5. 看到已识别、重复、未解析和失败的文献数量；
6. 重复扫描时只处理发生变化的文件；
7. 在不影响现有 Manual 功能的情况下生成 `documents.json`。

目录和子目录只表示扫描范围。P1 不根据目录名、collection、group 或 tag 划分兴趣方向。

## 2. 范围

### 2.1 包含

- Vault 内目录来源；
- Vault 内单个导出文件来源；
- 平铺目录和递归目录；
- Markdown、BibTeX、RIS、CSL JSON；
- 可从文件名或 sidecar 识别 arXiv ID/DOI 的 PDF；
- arXiv ID 元数据补全；
- 多来源合并与去重；
- 文件 fingerprint 和增量扫描；
- 扫描预览、进度、取消和错误汇总；
- 原子保存 sources/catalog。

### 2.2 不包含

- 外部任意文件系统目录；
- Zotero local API 或 SQLite；
- PDF 全文解析；
- 兴趣画像；
- Dashboard 标星学习；
- Personalized 推荐；
- 目录 grouping 策略；
- LLM 调用。

## 3. 数据模型

### 3.1 ReferenceSource

保持来源配置简单：

```ts
interface ReferenceSource {
  id: string;
  name: string;
  kind: "vault_folder" | "export_file";
  path: string;
  recursive: boolean;
  enabled: boolean;
  createdAt: string;
  updatedAt: string;
}
```

规则：

- `vault_folder` 可以递归；
- `export_file` 忽略 `recursive`；
- path 使用 `StorageAdapter.normalizePath`；
- source 只决定扫描位置，不携带兴趣方向；
- 来源配置保存在 `sources.json`，不塞入主插件 `data.json`；
- 删除来源后，下一次扫描移除该来源贡献，但不删除其他来源中的同一论文。

### 3.2 ReferenceDocument

```ts
interface ReferenceDocument {
  id: string;
  canonicalKey: string;
  arxivId?: string;
  doi?: string;
  title?: string;
  authors: string[];
  year?: number;
  abstract?: string;
  keywords: string[];
  noteExcerpt?: string;
  attachmentPaths: string[];
  sourceRefs: Array<{
    sourceId: string;
    itemPath: string;
    fingerprint: string;
  }>;
  parseStatus: "ready" | "partial" | "unresolved";
  warnings: string[];
  active: boolean;
  updatedAt: string;
}
```

P1 可以保留管理器导出中的 tags/collections 作为普通元数据，但不得把它们转换成兴趣方向或创建 grouping 配置。

### 3.3 Catalog

```ts
interface ReferenceCatalog {
  schemaVersion: number;
  revision: number;
  updatedAt: string;
  documents: Record<string, ReferenceDocument>;
  sourceScans: Record<string, {
    scannedAt: string;
    fingerprint: string;
    files: number;
    ready: number;
    unresolved: number;
    errors: number;
  }>;
}
```

只有 catalog 有效内容变化时才递增 revision。

## 4. 存储路径

基于 `derivePaperInboxPaths(output).indexDir` 派生：

```text
arxiv-daily/.index/personalization/
├── sources.json
└── documents.json
```

要求：

- schema version；
- 原子写入；
- 读取损坏时给出明确错误；
- 扫描失败不得覆盖上一份有效 catalog；
- output 路径变化后重新构造 store；
- 未配置任何来源时不创建 catalog 文件。

## 5. 建议代码结构

### 5.1 新建

- `plugin/src/personalization/types.ts`
- `plugin/src/personalization/paths.ts`
- `plugin/src/personalization/source-store.ts`
- `plugin/src/personalization/catalog-store.ts`
- `plugin/src/personalization/reference-library-service.ts`
- `plugin/src/personalization/reference-scanner.ts`
- `plugin/src/personalization/fingerprint.ts`
- `plugin/src/personalization/deduplicate.ts`
- `plugin/src/personalization/metadata-resolver.ts`
- `plugin/src/personalization/normalizers/index.ts`
- `plugin/src/personalization/normalizers/markdown.ts`
- `plugin/src/personalization/normalizers/bibtex.ts`
- `plugin/src/personalization/normalizers/ris.ts`
- `plugin/src/personalization/normalizers/csl-json.ts`
- `plugin/src/personalization/normalizers/pdf-reference.ts`

### 5.2 修改

- `plugin/src/core/adapters.ts`
- `plugin/src/hosts/obsidian/storage-adapter.ts`
- `plugin/src/hosts/node/storage-adapter.ts`
- `plugin/main.ts`
- `plugin/src/settings/tab.ts`
- `plugin/src/commands.ts`
- `plugin/src/services/progress.ts`
- 对应 tests 和 Obsidian mock。

## 6. 实施任务

### Task 1：Storage stat 与路径能力

为增量扫描增加可选 stat：

```ts
interface StorageStat {
  size: number;
  mtime?: number;
}

interface StorageAdapter {
  stat?(path: string): Promise<StorageStat | null>;
}
```

实现：

- Obsidian adapter 使用 Vault adapter stat；
- Node adapter 使用 `fs.stat`；
- adapter 不支持 stat 时退化为内容 hash；
- 不允许业务代码访问 Obsidian 私有 file object。

测试：

- 文件存在/不存在；
- 文件夹；
- mtime 缺失；
- Node/Obsidian adapter contract。

### Task 2：SourceStore 与 CatalogStore

- [ ] 派生 personalization 路径；
- [ ] 实现 sources load/save/add/update/remove；
- [ ] 实现 catalog load/save；
- [ ] 使用 `writeTextAtomic`，缺失时回退到现有原子写方案；
- [ ] 校验 schema；
- [ ] 未知 future schema 不自动覆盖；
- [ ] 增加空状态 helper。

测试：

- 首次加载；
- 正常保存；
- 损坏 JSON；
- future schema；
- 原子写失败；
- 删除来源后重建。

### Task 3：ReferenceScanner

扫描行为：

- `vault_folder`：扫描当前目录，recursive=true 时递归；
- `export_file`：只读取指定文件；
- 默认支持扩展名：`.md`、`.bib`、`.ris`、`.json`、`.pdf`；
- 默认排除 `.git`、`.obsidian`、`.index`、`node_modules`；
- 排除当前 output.dailyDir 和 output.papersDir；
- 不根据子目录创建任何兴趣或权重；
- 支持 AbortSignal；
- 设置文件数、深度和单文件大小上限；
- 单个文件失败不终止整个来源。

扫描器输出 item 清单，不直接写 catalog。

测试：

- 平铺目录；
- 多级递归；
- recursive=false；
- 默认排除项；
- arXiv Daily 输出目录；
- 空目录；
- 文件上限；
- 中途取消；
- list/stat/read 失败。

### Task 4：Fingerprint

fingerprint 用于判断是否需要重新解析：

- 文本文件：path + size + mtime；mtime 不可用时增加内容 hash；
- PDF：path + size + mtime，P1 不为 fingerprint 读取完整 PDF；
- export file：文件 fingerprint；
- export file 中的单条记录：文件 fingerprint + entry key + 记录内容 hash。

规则：

- fingerprint 相同则复用上次解析结果；
- source path 改变但 canonical document 相同，仍可去重；
- 扫描顺序不影响 catalog revision；
- 只更新扫描时间不能递增 catalog revision。

### Task 5：Normalizers

所有 normalizer 输出统一的 `NormalizedReference`，不访问网络、不写文件。

#### Markdown

提取：

- frontmatter 中的 title/authors/year/arxiv/doi/keywords；
- 正文中的 arXiv URL 和 DOI URL；
- H1 作为 title fallback；
- Abstract/摘要段落的有限长度内容；
- 只保存 note excerpt，不保存全文。

#### BibTeX / Better BibTeX

支持常用字段：

- citation key；
- title、author、year、abstract、keywords；
- doi、eprint、archivePrefix、url；
- file/attachment。

实现前先评估一个轻量、维护中的 parser。若采用依赖：

- 记录许可证；
- 检查 production bundle 增量；
- 为 malformed entry 添加隔离测试。

禁止用无法处理嵌套花括号的单条正则替代 parser。

#### RIS

支持 TY、TI/T1、AU/A1、PY/Y1、AB/N2、DO、UR、KW、L1/L2、ER。

#### CSL JSON

支持单对象或数组，读取 title、author、issued、abstract、DOI、URL、keyword 和 attachment 扩展。

#### PDF reference

- 从文件名识别 arXiv ID；
- 从同名 Markdown/BibTeX sidecar 关联标识；
- 无标识时输出 unresolved；
- 不解析 PDF 正文。

### Task 6：MetadataResolver

P1 只实现 arXiv 元数据补全：

- 收集 catalog 中缺少 title/abstract 的 arXiv IDs；
- 复用 `ArxivFetcher.fetchMetadataByIds`；
- 批量请求；
- 支持取消和已有 retry；
- 部分失败保留 partial/unresolved；
- 不因网络失败丢弃已解析文献；
- DOI 网络 resolver 留作后续扩展接口。

测试：

- 全部成功；
- 部分成功；
- 空结果；
- 网络失败；
- 取消；
- 已有完整元数据不重复请求。

### Task 7：Deduplicate 与 catalog commit

去重优先级：

1. 规范化 arXiv ID；
2. 规范化 DOI；
3. 标准化 title + year + first author；
4. sourceId + itemPath。

合并规则：

- 合并 sourceRefs、keywords 和 attachmentPaths；
- 完整 metadata 优先于 partial；
- 冲突字段记录 warning；
- 同一论文只保留一个 active document；
- 某来源删除文件时，只移除对应 sourceRef；
- 无 sourceRef 后将 document 标记 inactive；
- next catalog 完整验证后一次性原子替换。

### Task 8：ReferenceLibraryService

统一工作流：

```text
load sources
→ scan enabled sources
→ reuse unchanged records
→ normalize changed records
→ resolve arXiv metadata
→ deduplicate
→ validate
→ commit catalog
```

返回扫描摘要：

- sources；
- files；
- ready documents；
- partial；
- unresolved；
- duplicates；
- errors；
- catalog revision changed/unchanged。

支持 preview 模式，preview 不写 catalog。

### Task 9：设置页和命令

设置页增加简单的 Reference Library 区域：

- 来源列表；
- Add folder；
- Add export file；
- path；
- recursive；
- enabled；
- Preview；
- Scan now；
- Remove；
- 最近扫描统计。

命令：

- `arXiv Daily: Scan reference library`；
- `arXiv Daily: Preview reference library`。

约束：

- P1 不显示 Enable personalization 或 Personalized mode；
- 扫描不会调用 LLM；
- 删除来源需要确认；
- path 不存在时显示可操作错误；
- 不能选择整个 Vault root 而不进行明确二次确认。

### Task 10：Plugin wiring

在 `ArxivDailyPlugin` 中增加：

- ReferenceSourceStore；
- ReferenceCatalogStore；
- ReferenceLibraryService；
- build/reload helper；
- output path 变更时重建 stores；
- unload 时取消正在进行的扫描。

保持现有 scheduler、pipeline 和 Manual 模式完全不变。

## 7. 测试文件

建议新建：

- `plugin/tests/personalization/source-store.test.ts`
- `plugin/tests/personalization/catalog-store.test.ts`
- `plugin/tests/personalization/reference-scanner.test.ts`
- `plugin/tests/personalization/fingerprint.test.ts`
- `plugin/tests/personalization/deduplicate.test.ts`
- `plugin/tests/personalization/metadata-resolver.test.ts`
- `plugin/tests/personalization/reference-library-service.test.ts`
- `plugin/tests/personalization/normalizers/markdown.test.ts`
- `plugin/tests/personalization/normalizers/bibtex.test.ts`
- `plugin/tests/personalization/normalizers/ris.test.ts`
- `plugin/tests/personalization/normalizers/csl-json.test.ts`
- `plugin/tests/personalization/normalizers/pdf-reference.test.ts`

更新：

- adapter contracts；
- settings tab；
- commands；
- plugin lifecycle/mocks。

## 8. 验证

```bash
cd plugin
npx vitest run tests/personalization
npm test
npm run build
```

手工场景：

| 场景 | 预期 |
|---|---|
| 一个平铺目录 | 扫描全部支持文件并生成 catalog |
| 一个多级目录 | recursive=true 时扫描全部子目录，但不创建目录兴趣 |
| 两个来源包含同一论文 | catalog 中只有一篇，保留两个 sourceRefs |
| Zotero/Better BibTeX 导出 | 导入文献，不把 collection 当作兴趣 |
| 无法识别的 PDF | 标记 unresolved，其他文献正常导入 |
| 第二次扫描无变化 | 不重新解析，catalog revision 不变 |
| 扫描中取消 | 保留上一份 catalog |
| P1 功能未使用 | 当前插件行为不变 |

## 9. 完成标准

- [ ] 支持一个或多个目录/导出文件；
- [ ] 平铺和多级目录都能扫描；
- [ ] 目录层级不影响兴趣语义；
- [ ] Markdown/BibTeX/RIS/CSL JSON 正常与错误 fixtures 通过；
- [ ] PDF identifier-only 路径可用；
- [ ] arXiv metadata 可补全；
- [ ] 多来源去重正确；
- [ ] 未变化文件不重复解析；
- [ ] 扫描失败或取消不破坏旧 catalog；
- [ ] P1 不调用 LLM；
- [ ] Manual 全量测试和 production build 通过；
- [ ] P2 只通过 CatalogStore 读取文献，不再直接扫描来源。

## 10. P2 接口

P1 向 P2 提供：

- `ReferenceCatalogStore.load()`；
- `ReferenceLibraryService.scan()`；
- catalog revision；
- active documents；
- 每篇文献的规范化 metadata；
- unresolved/error 统计。

P2 不需要了解目录结构、导出格式或文件扫描细节。

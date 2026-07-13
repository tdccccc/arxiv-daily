# P3 — Personalized Recommendation

> Phase ID：P3
> 依赖：P2 的有效 InterestProfile
> 对应目标：在不要求用户配置 topics 的情况下，用兴趣画像筛选当日论文并生成推荐理由。
> 核心约束：Manual 模式行为保持不变。

## 1. 交付结果

完成后，用户可以：

1. 在设置中选择 Manual 或 Personalized；
2. 在已有有效画像时开启 Personalized；
3. 由画像自动给出需要关注的 arXiv categories；
4. 对当日候选论文进行 LLM 相关性判断；
5. 在日报和 Dashboard 中看到命中的兴趣方向和简短推荐理由；
6. 在个性化画像不可用时安全回退，不破坏现有日报流程。

## 2. 范围

### 2.1 包含

- Manual/Personalized mode；
- 从 enabled interests 汇总候选 categories；
- 使用画像 description/keywords 筛选候选；
- 新的 personalized filter；
- 推荐理由；
- 画像兴趣方向映射为运行时 topic；
- 论文索引保存推荐信息；
- 日报和 Dashboard 展示推荐信息；
- Manual 回归和 fallback；
- 候选数量/category 数量上限。

### 2.2 不包含

- 自动刷新画像；
- 文献库扫描触发；
- 每次 Star 后立即更新；
- embeddings；
- 向量检索；
- 学习排序模型；
- exploration 配额；
- 复杂评分 UI；
- 直接搜索整个 arXiv；
- 使用 to_read/saved/ignored 学习。

## 3. 设置模型

在 `PluginSettings` 增加：

```ts
interface PersonalizationSettings {
  mode: "manual" | "personalized";
}
```

默认：

```ts
personalization: {
  mode: "manual",
}
```

规则：

- 新旧用户迁移后都为 manual；
- 切换 personalized 前必须存在有效 profile；
- profile 至少有一个 enabled interest；
- 切换失败不保存 mode；
- 不自动覆盖 `arxiv.categories` 和 `arxiv.topics`；
- Manual 设置继续保留，作为显式回退配置。

首版候选上限使用代码常量，不增加高级设置项，避免设置页复杂化。

## 4. 运行时兴趣设置

现有 summarizer/markdown writer 依赖 `ArxivSettings.topics`。P3 不直接重写用户设置，而是根据 profile 构造本次运行的临时设置：

```ts
interface RecommendationContext {
  mode: "manual" | "personalized";
  arxiv: ArxivSettings;
  profile?: InterestProfile;
}
```

Personalized runtime topics：

- `id`：interest ID；
- `name`：interest name；
- `tag`：由 interest ID/name 生成稳定 slug；
- `description`：interest description + keywords；
- `detail`：true，使 personalized filter 的 detail 判断可生效。

Runtime categories：

- 汇总所有 enabled interest 的 `arxivCategories`；
- 去重；
- 只保留合法 category；
- 设置内部上限，例如 8；
- 没有合法 category 时拒绝 personalized run，并回退 Manual。

该转换只存在于内存中，不写回用户的 manual topics。

## 5. 候选获取

首版只用 arXiv categories 扩展候选范围，keywords 用于 LLM 判断，不实现任意全文搜索。

流程：

1. 从 RecommendationContext 获取 categories；
2. 对每个 category 获取指定日期的论文；
3. 合并 arXiv ID；
4. 补全摘要和 category；
5. 应用总候选上限；
6. 交给对应 filter。

优先复用当前 `fetchPapersForDate` 和 `fetchRecent` 逻辑，避免同时维护两套日期语义。

需要重构：

- `fetchPapersForDate(date, categories, signal)`；
- Manual 传当前 `arxivCategories(settings)`；
- Personalized 传 profile categories；
- 其他 pipeline 阶段使用本次运行的 runtime `ArxivSettings`，不能继续直接读取 `this.deps.arxiv`。

候选上限：

- 单 category 继续受 arXiv recent 页限制；
- 合并后设置内部最大候选数；
- 超限时使用稳定排序截断并写日志；
- 不因一个 category 失败而丢弃其他成功 category。

## 6. Personalized filter

新建独立模块，避免破坏现有 Manual filter：

- `plugin/src/personalization/personalized-filter.ts`
- `plugin/src/prompts/personalization/personalized-filter.system.md`

输入：

- 精简 profile summary；
- enabled interests；
- 每个 interest 的 description/keywords；
- 候选论文 ID/title/authors/abstract/categories；
- 日期和必要输出规则。

输出每篇候选：

- recommended；
- interest ID；
- reason；
- detail。

规则：

- interest ID 必须来自 enabled interests；
- reason 简短、具体，说明论文与画像中哪个方向相关；
- 不推荐论文不需要 reason；
- detail 从严；
- LLM 不得创建新兴趣方向；
- 文献内容视为不可信数据；
- 严格 JSON；
- unknown paper ID/interest ID 丢弃并记录；
- 同一论文只产生一个 primary interest；
- 不输出数值评分，首版只做布尔选择。

## 7. Filter 结果模型

扩展现有 `FilteredPaper` 或定义兼容字段：

```ts
interface PersonalizedRecommendation {
  interestId: string;
  interestName: string;
  reason: string;
  profileGeneratedAt: string;
}

interface FilteredPaper {
  // existing fields
  recommendation?: PersonalizedRecommendation;
}
```

Manual filter 的 `recommendation` 为空。

后续 pipeline 继续使用现有 category/detail 逻辑：

- Personalized 的 `category` 使用 runtime topic tag；
- `isDetail` 使用 personalized decision；
- summarizer 可以按 runtime topics 输出分区；
- writer 不需要知道 profile store。

## 8. 论文索引

为 `PaperIndexEntry` 增加可选字段：

```ts
recommendation?: {
  interestId: string;
  interestName: string;
  reason: string;
  profileGeneratedAt: string;
}
```

修改：

- `PaperIndexUpsert`；
- `upsertEntry`；
- normalize/migration；
- history sync 对未知字段保持兼容；
- Manual 后续再次看到同一论文时，不应无意清除已有 personalized reason，除非有新的明确 recommendation；
- 新 personalized run 可以更新 reason/profile timestamp。

不新增独立 recommendations 日志文件。

## 9. 日报与 Dashboard

### 9.1 日报

把 recommendation 信息传入 daily summarizer：

- 命中的兴趣方向；
- 简短推荐理由。

输出要求：

- 每篇 Personalized 论文保留一行或一段“推荐原因”；
- Manual 输出格式保持原样；
- prompt 只接收精简画像和单篇 reason，不接收完整文献库；
- Snapshot tests 分 Manual/Personalized 两组。

### 9.2 Dashboard

在不重构整个 view 的前提下增加：

- interest name；
- recommendation reason；
- 无 recommendation 时不显示占位；
- reason 使用文本展示，不注入 HTML；
- 保留 Star 操作。

优先把展示逻辑拆成小 helper/component，避免继续增加 `view.ts` 主类复杂度。

## 10. 建议代码结构

### 10.1 新建

- `plugin/src/personalization/recommendation-context.ts`
- `plugin/src/personalization/personalized-filter.ts`
- `plugin/src/personalization/recommendation-service.ts`
- `plugin/src/prompts/personalization/personalized-filter.system.md`

### 10.2 修改

- `plugin/src/settings/types.ts`
- `plugin/src/settings/defaults.ts`
- `plugin/src/settings/migration.ts`
- `plugin/src/settings/tab.ts`
- `plugin/main.ts`
- `plugin/src/pipeline/pipeline.ts`
- `plugin/src/pipeline/paper-filter.ts`：只提取可共享 parser/helper，不改变 Manual 行为；
- `plugin/src/pipeline/summarizer.ts`
- `plugin/src/pipeline/markdown-writer.ts`
- `plugin/src/services/paper-index.ts`
- `plugin/src/dashboard/model.ts`
- `plugin/src/dashboard/view.ts`
- prompts、tests 和 snapshots。

## 11. 实施任务

### Task 1：Settings 与模式切换

- [ ] 增加 PersonalizationSettings；
- [ ] 默认 manual；
- [ ] migration；
- [ ] 设置页 Manual/Personalized；
- [ ] 切换 personalized 前验证 profile；
- [ ] 无 enabled interests 时阻止；
- [ ] 保存失败恢复 UI；
- [ ] 切回 manual 不删除 profile/catalog。

### Task 2：RecommendationContext

- [ ] 读取 mode；
- [ ] manual 返回现有 settings；
- [ ] personalized 加载 profile；
- [ ] 过滤 disabled interests；
- [ ] 构造 runtime topics/categories；
- [ ] 合法 category 验证和去重；
- [ ] category/topic 上限；
- [ ] profile 缺失/损坏 fallback；
- [ ] 不修改 `this.settings.arxiv`。

测试：

- manual；
- personalized；
- disabled interests；
- duplicate categories；
- invalid categories；
- empty profile；
- profile load error。

### Task 3：Pipeline 参数化

把运行期 arXiv 设置作为局部变量贯穿：

- candidate fetch；
- filter；
- index upsert；
- content fetch；
- daily summarize；
- detail summarize；
- writer。

避免在方法内部混用 `this.deps.arxiv` 和 runtime arxiv。

Manual 路径必须继续通过现有 fixtures/snapshots。

### Task 4：Personalized candidate fetch

- [ ] 使用 runtime categories；
- [ ] 多 category 容错；
- [ ] ID 去重；
- [ ] 保留所有 source categories；
- [ ] 候选总量上限；
- [ ] AbortSignal；
- [ ] 日志记录 categories、原始数、去重数、截断数。

### Task 5：Personalized filter

- [ ] 新 prompt；
- [ ] injection guard；
- [ ] JSON parser/validator；
- [ ] known paper/interest validation；
- [ ] reason 截断；
- [ ] unknown item drop；
- [ ] detail 判断；
- [ ] cancellation/error 分类；
- [ ] 无推荐返回空数组。

不要复用 Manual topic prompt 后通过大量条件分支实现 personalized；共享只限 JSON 提取和基础验证 helper。

### Task 6：PaperIndex recommendation

- [ ] schema optional field；
- [ ] upsert；
- [ ] normalize；
- [ ] existing index migration；
- [ ] clear/remove 行为；
- [ ] same paper reason update；
- [ ] Manual 不误删；
- [ ] tests。

### Task 7：日报输出

- [ ] 把 recommendation 加入 summarizer 输入；
- [ ] Personalized prompt 要求保留推荐原因；
- [ ] Manual prompt/snapshot 不变化；
- [ ] 中文和英文输出；
- [ ] reason 缺失时安全退化；
- [ ] detail report 无需重复整段画像。

### Task 8：Dashboard 展示

- [ ] model 暴露 recommendation；
- [ ] row/detail 中显示 interest/reason；
- [ ] 搜索/排序不因新字段改变；
- [ ] reason 文本安全；
- [ ] Star 行为保持；
- [ ] responsive styles；
- [ ] view tests。

### Task 9：Fallback

固定规则：

1. Manual：永远走现有流程；
2. Personalized + valid profile：走 personalized；
3. Personalized + profile 暂时读取失败：记录 error，使用最后成功加载的 profile（若内存存在）；
4. 没有任何有效 profile：回退 Manual，并通知用户重新构建画像；
5. Manual topics 为空时允许生成 0 篇结果，但不能 crash。

不在一次运行中自动重建 profile，自动刷新留给 P4。

## 12. 测试

建议新建：

- `plugin/tests/personalization/recommendation-context.test.ts`
- `plugin/tests/personalization/personalized-filter.test.ts`
- `plugin/tests/personalization/recommendation-service.test.ts`
- `plugin/tests/personalization/personalized-pipeline.test.ts`

更新：

- migration；
- settings-tab；
- pipeline；
- paper-filter；
- paper-index；
- summarizer snapshots；
- markdown-writer；
- dashboard model/view；
- commands/onboarding（如模式入口涉及）。

关键测试：

- Manual snapshot 完全不变；
- Personalized 不需要 manual topics；
- 两个 interest、多 categories；
- category 去重/截断；
- invalid profile；
- LLM malformed JSON；
- unknown interest ID；
- no recommendations；
- recommendation reason 写入 index；
- Dashboard 展示；
- cancellation；
- category 部分失败；
- fallback Manual。

## 13. 验证

```bash
cd plugin
npx vitest run tests/personalization/personalized-filter.test.ts
npx vitest run tests/personalization/personalized-pipeline.test.ts
npm test
npm run build
```

手工场景：

| 场景 | 预期 |
|---|---|
| 无 profile | 不能开启 Personalized |
| 有 profile、无 manual topics | Personalized 正常推荐 |
| 多兴趣画像 | 日报按画像方向组织 |
| 论文与画像相关 | 显示具体推荐理由 |
| 没有相关论文 | 完成 0 篇，不生成错误内容 |
| profile 损坏 | 回退 Manual 并提示 |
| 切回 Manual | 输出与当前版本一致 |
| Dashboard Star | 仍正常工作，P3 不立即重建画像 |

## 14. 完成标准

- [ ] Manual/Personalized 可切换；
- [ ] Personalized 不需要用户配置 topics；
- [ ] 候选范围来自 enabled interests 的 categories；
- [ ] LLM 根据精简画像筛选候选；
- [ ] 每篇推荐有 interest 和 reason；
- [ ] recommendation 可进入 index、日报和 Dashboard；
- [ ] profile 不可用时安全 fallback；
- [ ] Manual tests/snapshots 不回归；
- [ ] 全量测试和 production build 通过；
- [ ] P4 可在运行前刷新 profile，而无需修改推荐协议。

## 15. P4 接口

P3 向 P4 提供：

- `RecommendationService.resolveContext()`；
- 当前 mode；
- profile 有效性；
- pipeline 开始前的可插入 refresh hook；
- Manual fallback 行为。

P4 只负责确保 profile 尽可能新，不改变推荐 JSON 和日报格式。

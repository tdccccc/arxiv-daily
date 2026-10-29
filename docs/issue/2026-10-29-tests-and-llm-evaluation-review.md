# 单元测试、大模型评测与验证缺口审阅

审阅日期：2026-10-29。范围：当前仓库的测试实现、评测脚本、CI 配置，以及现有本地评测产物。

**结论：工程回归测试已有较好的基础；主要缺口在评测结果的隔离与可比性、真实样本的代表性，以及评审模型自身的漏报验证。** 本次离线评测工具测试全部通过，但额外复现了两种现有测试未拦住的结果污染：推理设置变化仍复用旧结果；缩小样本并更换评审模型后，报告仍混入上次运行的记录。

## 1. 审阅边界与证据口径

- 本次仅新增本文。未修改产品代码、测试、配置、Helm 文件或仓库外评测结果，未提交 commit，未调用真实模型或触发真实业务运行。
- 用户正在另一个会话推进 [summary-quality-evaluation](../helm/2026-10-28-summary-quality-evaluation/goal.md)。本文不接管该计划；基线、核对网页、人工校准等按“进行中”记录，不把尚未完成的阶段视为遗漏承诺。
- 开始审阅时 HEAD 为 `d0221978ee5d5fac8e79c7a060dd58f4ab7ce68d`，存在该会话的评测工具未提交改动。审阅期间另一个会话提交了 `96956229ddc351a1ddba716cd929b78e4221d873`，本文引用的代码位置按这个版本核对。相关发现是开发中快照的反馈，另会话完成后应复核是否仍成立。
- 本地基线进度快照：**2026-10-29 21:45:52，Asia/Shanghai**。之后继续生成的结果不在本次结论内。
- 阅读覆盖了测试入口、所有工作流、测试文件清单，并重点检查 LLM 协议、筛选/摘要/评审、检索、持久化、宿主验收的实现及代表性测试。没有逐条人工审查全部测试用例，也没有测量行/分支覆盖率。
- 本次执行了 `node --test scripts/tests/quality-eval-*.test.mjs`：**17 个文件、272 个用例通过，0 失败、0 跳过**。首次沙箱运行因测试内部 `spawnSync /usr/bin/node EPERM` 失败；允许所需子进程后通过，不归类为产品缺陷。日志留于 `/tmp/arxiv-test-review-quality-eval-unsandboxed.tap`，属于临时证据。
- 另用现有测试的模拟模型和临时夹具复现了下文 T01、T02；未新增仓库测试文件。**未运行整个 workspace、桌面验收、真实模型、真实 embedding 或 Python sidecar 测试**，所以本文不宣称项目全部测试通过。

优先级：P1 表示会影响现有质量结论的可信度，应在依赖这些结论作模型/提示词决策前处理；P2 表示重要覆盖扩充；P3 表示持续治理。它们不是生产事故等级。

## 2. 已有测试和评估能力

以下按测试文件计数，排除 snapshot、mock、夹具和辅助脚本，不能换算成代码覆盖率。

| 范围 | 测试文件数 | 已有覆盖及边界 |
| --- | ---: | --- |
| `packages/core/tests` | 153 | 筛选和摘要合同、重试/拒绝/截断、流水线失败恢复、调度、取消、checkpoint、索引、检索、设置、邮件等；多数使用受控输入及模拟依赖 |
| `packages/node-runtime/tests` | 9 | 文件锁、存储、投递崩溃、文献库路径范围、索引组合等 |
| `packages/node-runtime/native/tests` | 1 | 原生存储测试入口；另有备份诊断脚本，CI 覆盖六种 OS/架构组合 |
| `apps/cli/tests` | 52 | CLI 参数、配置、初始化、调度、工作台 HTTP/DOM、异步竞态等 |
| `plugin/tests` | 64 | Obsidian 生命周期、设置、Dashboard、文献库、数据损坏、写入失败等；Vitest 使用 happy-dom 和 Obsidian mock |
| `scripts/tests` | 65 | 构建、发布、工作流与验收基础设施，以及 17 个质量评测工具测试文件 |
| `services/email-relay/tests` | 6 | 鉴权、幂等、并发限流、DO 状态、配额、错误分类、切换和部署合同；文件少不代表覆盖浅 |
| `extensions/dsh-arxiv-daily/tests` | 6 | 插件与真实 DSH Host 集成；CI 验证两个 DSH 版本 |
| `extensions/vscode-arxiv-daily/scripts/*-test.mjs` | 5 | workspace adapter、Dashboard、共享锁、CLI 命令与扩展 smoke |
| `tools/docling-sidecar/tests` | 1 | Python HTTP 协议及文档结构转换，使用 FakeParser |
| **合计** | **362** | 文件数统计，不是用例总数 |

已有能力值得保留，不能误报成“完全没测”：

- **LLM 接口健壮性已有较密集测试。** [llm 测试目录](../../packages/core/tests/llm/)覆盖 OpenAI-compatible/Anthropic 流式结束、输出截断、取消、usage、请求参数、兼容降级和失败分类；[pipeline 测试目录](../../packages/core/tests/pipeline/)还有接口故障如何影响业务提交的测试。这些主要证明客户端行为，不能证明模型选得准或摘要准确。
- **筛选质量工具已经存在且有真实基线。** [quality-eval/README.md](../../scripts/quality-eval/README.md)提供抽样标注、记录评分、真实重跑、重复运行稳定性、分层加权、分主题与阈值分析。它已经说明“重跑补了摘要”的混杂因素，并给出原先有摘要的子集对照。
- **摘要质量分三层。** 离线规则检查格式/数学/章节/回退/数字线索；regenerate 复用产品生成路径；judge 用另一模型对相同缓存输入评准确、完整、清楚，验证引用与输出结构。核对网页已进入当前代码；完成工具不等于已完成人工校准。
- **检索已有固定指标测试。** [fulltext-retrieval-evaluation.test.ts](../../packages/core/tests/fulltext-retrieval-evaluation.test.ts)计算 Recall、MRR、nDCG，并验证 dense/BM25/RRF；不是只有结果长度断言。
- **UI 有真实交互验收。** [acceptance/README.md](../../scripts/acceptance/README.md)说明工作台和 Obsidian 都可运行真实宿主、真实流水线及文件存储，论文/业务模型/邮件使用固定响应。模型驱动的探索是另一层：模型负责操作，任务结果由程序核验；自由探索疑点仍待人工复核。这两层均不构成论文内容质量评测。
- **CI 已有分层。** [lint.yml](../../.github/workflows/lint.yml)跑工具测试、workspace 测试、类型检查、构建及安装 smoke；[ui-acceptance.yml](../../.github/workflows/ui-acceptance.yml)跑工作台真实浏览器旅程；[native-storage.yml](../../.github/workflows/native-storage.yml)跑六平台原生能力及特定宿主组合；邮件、VS Code、DSH 有独立工作流。普通 CI 不调用真实模型是合理边界。

## 3. 优先补齐的测试与评测

### T01 · P1 · 生成/评审缓存没有覆盖推理设置变化【已复现】

**证据。** [summary-regenerate.ts](../../scripts/quality-eval/summary-regenerate.ts)第 162 行起的 `isCurrent` 比较 provider、model、system prompt hash、source hash；[summary-judge-run.ts](../../scripts/quality-eval/summary-judge-run.ts)第 164 行起另比较 summary hash。两处均未比较 `thinkingMode`、`reasoningEffort`，虽然产物记录了它们。现有 [regenerate 测试](../../scripts/tests/quality-eval-summary-regenerate.test.mjs)第 202 行起和 [judge-run 测试](../../scripts/tests/quality-eval-summary-judge-run.test.mjs)第 227 行起，只测试改模型、提示词标识和原文会失效。

**离线复现。** 用现有夹具先生成 3 条日报和 1 篇笔记，然后保持模型名不变：

| 改动 | 新模拟调用次数 | 被复用条数 |
| --- | ---: | ---: |
| 生成模型 `reasoningEffort: medium → high` | 0 | 4 |
| 生成模型 `thinkingMode: false → true` | 0 | 4 |
| 评审模型同时修改以上两个设置 | 0 | 4 |

结果是“换了推理设置”的实验可能根本没有重新运行，新的 `run.json` 又记录新的设置，容易误读。不同输出目录可以规避，但默认续跑仍需保证实验身份一致。

**应补测试。** 分别改变生成端、评审端的有效推理设置，断言旧缓存失效、调用数与更新条目数一致、报告身份与逐条记录一致；无效于该 provider 的设置变化则按产品的归一化规则判定。再覆盖实际请求内容改变、只有密钥轮换、显式 `--regenerate`/`--rejudge`、不变配置续跑。优先复用 core 的有效生成身份，避免评测另造不完整的比较规则。

### T02 · P1 · 报告缺少“本次样本”隔离测试【已复现】

**证据。** [summary-judge-report.ts](../../scripts/quality-eval/summary-judge-report.ts)第 101 行起的 `loadJudgments` 读取 `judgments/` 下全部 JSON；不按 `run.json.sample` 或本次模型/提示词身份过滤。第 143 行起对这些记录汇总。运行器可以重用输出目录并重写 `run.json`，不会删除上次样本的记录。

**离线复现。** 第一次在目录 A 用 `judge-model` 评 4 条；第二次在 A 中改成 `judge-model-2`，用 `--ids` 只评其中 1 条。结果：`run.sample` 只有 1 条、`counts.judged = 1`、报告头为新模型，但 `judgments.json.records` 仍有 **4 条：1 条新模型结果 + 3 条旧模型结果**。质量分布混合了两个实验。

**应补测试。** 同目录先大样本后小样本、改变 seed/日期范围/模型、原文变化导致某条被跳过、上次只运行一半、本次中途停止、`--report-only` 重建。断言报告仅包含当前清单且身份匹配的记录，或对不兼容目录明确拒绝；不能把旧文件自动算入当前结论。校准页面也应使用同一份经过验证的样本清单。

### T03 · P1 · 人工校准只能核对已报问题，测不到 judge 漏报【代码确认；计划进行中】

**证据。** [summary-review.ts](../../scripts/quality-eval/summary-review.ts)第 65 行起只从 `judgment.issues` 生成卡片，第 83 行起优先选 major，再分散到论文与维度；第 366 行起计算 `agree / (agree + disagree)`。这适合回答“报出的问题是否站得住”，不包含 `ok` 摘要，也不是随机总体样本。引用确实存在只证明引用可定位，不能证明推论正确。

**缺少的评测。**

1. 从判为 `ok` 的摘要抽取人工盲审样本，测漏报；同时审查有问题的条目，分别报告误报与漏报。
2. 建立人工确认的正确/错误对照样本：改错数字、互换实验组、把相对提升写成百分点、遗漏关键条件、把相关性写成因果、把作者声称写成事实，以及正确但表达不同的摘要。
3. 按 accuracy/completeness/clarity、major/minor 和日报/笔记分别计算；“拿不准”和未回答继续单列。
4. 固定一组样本重复评审，衡量 judge 自身波动；少量边界样本由第二位人工或独立评审复核。

**验收口径。** 当前的一致率应明确叫“抽查问题的认可比例”，不能命名为总体准确率或漏报率。至少 10 条核对可作为首次校准启动条件，不能证明模型总体可靠。这是对正在执行计划的后续补充建议，不要求本次会话介入标注或调用模型。

### T04 · P1 · 真实质量基线的代表性不足【已有产物确认】

**筛选基线。** [P3 基线记录](../helm/2026-10-15-filter-quality-evaluation/phases/03-first-baseline.md)与本地 `~/quality-eval/2026-10-15-baseline/` 报告一致：48 行、3 个应选正例、45 个负例；3 个正例全部属于 galaxy-cluster，photo-z 没有正例。记录精确率为 3/8，重跑为 3/5，召回率均为 3/3；3 次重跑选中结果一致。**“100% 召回”只说明找到这 3 篇，不能代表不同方向或未来日期。** 其中 28 行重跑增加了摘要，模型也发生变化；不能把全部差异归因于某一个因素。

**摘要基线。** 已存在真实离线报告：142 份日报中 77 份纳入当前格式检查、469 条五字段条目、3 条回退，46 篇笔记纳入检查，其中 30 篇具有全部当前章节。历史评审与重新生成评审各有 4 条完成的试跑，选择偏向已知有问题的论文。上述进度快照中，`summary-baseline-2026-10-29/regenerated/` 已有 **43 个 generated 文件**，`run.json.counts` 仍为 null；`judged/`、`review/` 尚不存在，未找到该评测目录下的 calibration 报告。**这是正在运行的阶段，不能推断为生成失败，也不能提前称为 50 条基线已验收。**

**应补评测。** 在现有计划之外按日期、方向、正负例、标题-only/有摘要、边界论文分层扩充；每个方向都需要足够的正例和难负例。留出不用于改提示词的固定验证集，并使用固定输入作前后配对比较。报告增加区间估计；跨日样本可按日期分组 bootstrap，小样本显示不足，不能仅给点估计。标注缺失若集中在难题上，也要报告两层标注完成率，不能只靠加权掩盖选择偏差。

### T05 · P2 · 筛选重跑缺少与实际日流程的对照验证【实现差异确认】

**证据。** [rerun.ts](../../scripts/quality-eval/rerun.ts)第 172、188、285 行附近，只将已标注论文合为一个请求，调用 `buildPaperFilterRequest → llm.call → decodePaperFilterResponse`。生产 [paper-filter.ts](../../packages/core/src/pipeline/paper-filter.ts)第 91、206 行附近按 400,000 字符预算分批，并有 checkpoint、拒绝后的拆分处理等。两者共享提示词构造和解析，但不等于复现完整运行。

**影响。** 同一天的未标注候选不再出现在模型上下文；顺序、批大小、多日混合、摘要来源都可能改变预测。扩充标注后，还可能出现评测单请求过大、而生产可分批完成的情形。现有“与 core 构造请求相同”测试不能证明评测代表线上分布。

**应补测试/评测。**

- 使用保存的完整日候选及其原始输入，保持生产分批和顺序，仅在计算指标时取有标签的行；用模拟响应验证每批输入、合并结果、遗漏处理与拒绝行为一致。
- 将“生产条件重放”和“所有论文统一补摘要后的模型能力实验”标成两种实验，分别比较。
- 固定模型只改提示词、固定提示词只改模型、固定两者只改摘要可用性，做单因素对照；真实调用留待后续单独执行。
- 除合同版本整数，记录实际 prompt/request 的内容 hash、语料/标注 hash 和代码版本。当前 [rerun.ts](../../scripts/quality-eval/rerun.ts)第 242 行起的身份只有合同版本、模型与模式等，不能辨认“改了文本但未升合同版本”的运行。
- 区分分类召回和最终日报召回：篇数上限、忽略、去重、内容获取失败分别造成多少损失；现有评测有意排除这些环节，不能用其召回率代表最终交付。

### T06 · P2 · 新笔记缺章可能被当作旧格式，缺少生成版本约束【代码确认】

**证据。** [summary-note-checks.ts](../../scripts/quality-eval/summary-note-checks.ts)第 119 行起，以“是否拥有全部当前章节”区分 current/older；第 132 行起只对 current 检查结构。[summary-judge-input.ts](../../scripts/quality-eval/summary-judge-input.ts)第 93 行默认只抽 current 笔记。这能兼容旧格式，但当**今天生成的笔记漏章**时，也可能被归为 older 而不进入默认样本。现有测试验证了兼容策略，没有解决生成版本缺失时的识别问题。

**应补测试。** 一篇已知由当前提示词生成的笔记，逐一删除必需章节、改标题、重复章节、正文为空，均应纳入当前输出的缺陷统计；真正的旧版本仍可单列。对 `--summaries` 的新生成记录可直接使用已知生成身份实施严格校验；历史 Vault 无法确定身份时显示“版本不明”，避免用章节完整性反推版本再豁免缺章。

### T07 · P2 · 检索有算法回归，真实 embedding 质量仍缺固定标准集【测试实现确认】

**证据。** [fulltext-retrieval-evaluation.test.ts](../../packages/core/tests/fulltext-retrieval-evaluation.test.ts)第 29 行起使用 6 维轴向人工向量、8 篇固定论文和 6 类查询；非常适合算法及排序回归，无法发现真实 embedding 模型语义能力退化。[real-corpus-retrieval.test.ts](../../packages/core/tests/real-corpus-retrieval.test.ts)第 73 行起仅在设置 `REAL_KB_DIR` 时运行，检查非空、存在 evidence hit、两个查询 top-5 不完全相同，没有人工相关性答案。真实 migration 测试同样为 opt-in。

**应补评测。** 固定可公开或脱敏的真实论文集合，人工标注相关性等级和查询意图；缓存明确模型/版本的真实向量，离线计算 Recall@k、MRR、nDCG@k。至少包含中英文、同义改写、标题精确匹配、缩写、无相关结果、关键词相似但研究问题不同的难负例。换 embedding/截断/分块策略时再生成向量并对照；保持现有人工向量测试作为算法单测。

### T08 · P2 · 其他 LLM 业务调用缺少与筛选、摘要同等级的质量评测【范围检查】

现有 `quality-eval` 入口集中在论文筛选、摘要和 judge；其他功能有行为测试，但未发现可重复使用、带人工标准答案的质量评测工具：

| 功能 | 已有测试示例 | 应补的质量标准 |
| --- | --- | --- |
| 详细笔记候选选择 | [detail-selector.test.ts](../../packages/core/tests/detail-selector.test.ts)：分数解析、阈值、并列排序、输入范围、错误与取消 | 在固定“值得精读”标签上评 Precision@k/Recall@k、排序相关性；覆盖异常高分、跨方向候选、没有值得精读的论文 |
| 文献库方向提案和主题组织 | [clustered-direction-proposer.test.ts](../../packages/core/tests/clustered-direction-proposer.test.ts)、[personal-library-topic-organization.test.ts](../../packages/core/tests/personal-library-topic-organization.test.ts)：模拟向量/回复、证据引用、数量与合同约束 | 人工认可率、证据支持率、重复/过宽方向比例、库内论文覆盖率；覆盖小库、混合学科、离群论文、已有方向、增量加入一篇论文后的稳定性 |
| 方向匹配预览 | [personal-library-direction-preview.test.ts](../../packages/core/tests/personal-library-direction-preview.test.ts)：输入合同、身份验证、分类范围 | 与同输入的真实日报筛选结果对照，并复用人工相关性标签 |

这些功能即使 JSON 合法、引用 ID 存在，也仍可能提出没有研究价值的方向或错误排列精读优先级。

### T09 · P2 · 原文提取和数字检查缺少独立内容标准；注入防护缺少行为挑战集【评估层次缺口】

使用相同缓存原文重新生成并评审，已经解决了很重要的输入可比性问题。但它测的是**忠实于模型收到的文本**：若提取阶段漏掉表格或关键限制，生成与 judge 可能共同看不到，仍相互一致。

已有 [section-extractor.test.ts](../../packages/core/tests/section-extractor.test.ts)、[source-extractor.test.ts](../../packages/core/tests/source-extractor.test.ts)、PDF/sidecar 测试验证提取和合同；[summary-grounding.ts](../../scripts/quality-eval/summary-grounding.ts)检查数字是否出现，不验证数字属于哪个对象、指标、单位或实验条件。当前已知错误试跑和少量数字抽查有价值，但未形成独立覆盖这些问题的固定标准集。

**应补评测。** 用少量固定 HTML、LaTeX、PDF 及人工标注关键事实，检查“原文件 → 提取文本 → 摘要 → 最终笔记”全过程。包含多栏、表格、图注、公式、单位换算、数量级、百分比/百分点、否定句、正文与结论不一致、同一数字在多个实验中出现。分别记录提取丢失与生成错误，不能一并记到模型名下。

另有提示词包裹/转义测试，例如 [paper-filter.test.ts](../../packages/core/tests/paper-filter.test.ts)第 589 行起和 [detail-selector.test.ts](../../packages/core/tests/detail-selector.test.ts)第 162 行起。它们验证防护文本与结构，不证明模型会拒绝执行论文里的指令。可在上述固定语料加入“要求全部选中”“要求 judge 判全对”“伪造系统指令/评分答案”等对照样本，统计攻击是否影响任务结果。无需把真实模型调用加入普通 CI。

### T10 · P2/P3 · 自动化入口、覆盖度和运行环境仍有空档【配置确认】

| 缺口 | 证据与边界 | 建议新增验证 |
| --- | --- | --- |
| Python sidecar 测试未接入已查 CI | [test_server.py](../../tools/docling-sidecar/tests/test_server.py)有 5 个 unittest 方法；README 给出独立运行方式，但 `.github/workflows/` 和根测试脚本未调用。`product-units.json` 的扫描根也不含 `tools/` | **P2**：变更 sidecar 时跑标准库协议测试；另以固定 PDF 做可选真实 Docling smoke，明确 FakeParser 测试不证明模型解析能力 |
| `npm test` 不是整个仓库的完整验证 | [run-root-tests.mjs](../../scripts/run-root-tests.mjs)无参数只跑 workspaces，有参数转发 core；工具测试、独立服务/扩展、sidecar 不全在其中。CI 对不少独立单元已有单独入口 | **P2**：提供清晰的验证清单或汇总入口，验证各单元是否被运行/有意跳过；不能仅凭根 `npm test` 宣称全部通过 |
| 无覆盖率基线，存在无效占位测试 | 各 Vitest 配置与工作流未配置 coverage report/threshold；[llm/api-testing.test.ts](../../packages/core/tests/llm/api-testing.test.ts)只有 `expect(true).toBe(true)`。连接成功、拒绝场景其实已在其他文件测试 | **P3**：清理占位测试；先对 LLM、pipeline、持久化的关键分支建立覆盖基线和未覆盖清单，再针对高风险纯逻辑少量做 mutation testing；不建议追逐全局 100% |
| 全宿主和平台组合尚非完整矩阵 | 常规工作流主要 Linux/Node 22；Node 20 做类型/构建/smoke；六平台矩阵重点是原生存储及少数组合测试；UI CI 只跑 workbench | **P2**：按承诺支持范围补 macOS/Windows 的 CLI 调度、路径/换行、取消/子进程；Obsidian 保留独立真实宿主发版证据，记录产物 hash、宿主版本和执行时间。无需把全部桌面环境硬塞每个 PR |
| 真实供应商兼容性缺持续小样本验证 | 丰富协议单测主要使用模拟 HTTP/SSE；现有真实基线只证明其记录的特定模型和接入方式能跑 | **P2**：手动或定期受控 canary，每种受支持协议覆盖一次结构化筛选、摘要、流式完成与 usage，并记录版本/响应形态。将脱敏响应回灌离线回归；不能拿“连接测试 ok”替代业务调用成功率 |
| 缺可比的整条流程成本/性能趋势 | 已有 usage/耗时记录、解析超时断言、索引容量测试及 [容量验证记录](../helm/2026-08-17-pdf-hybrid-library-foundation/phases/08-generation-routing-capacity.md)，不是完全没有性能验证；未发现统一的固定负载趋势与质量联报 | **P3**：固定小/中/大负载，记录生成成功率、回退率、重试后总 token、有效结果成本、P50/P95 延迟、索引峰值内存/查询延迟；只比较质量相当的方案 |

## 4. 建议的落实顺序

1. **先保障评测结果可相信：T01、T02。** 两者都有小规模离线复现，不需要花真实模型费用；由正在开发评测工具的会话完成后再复核，避免并行修改相同文件。
2. **完成当前摘要计划既定的基线与人工校准。** 明确把“抽查问题认可率”和“总体质量”分开；随后按 T03 增加 `ok` 条目盲审和人为错误对照。本文不要求重启或打断当前基线。
3. **扩充筛选验证集并固定比较条件：T04、T05。** 优先补缺少正例的方向、难负例、完整日输入；持续保留未用于调参的验证集。
4. **扩大质量链路：T06–T09。** 先解决已知当前生成输出的结构豁免，再覆盖检索、精读选择、方向提案、提取损失与注入挑战。
5. **补自动化空档和长期指标：T10。** 先接入已有却未自动运行的测试，再增覆盖率和性能趋势。普通 CI 保持确定性、离线；真实模型评测作为独立、可留存证据的流程。

## 5. 本次可下与不可下的结论

可确认：项目已有广泛工程测试、可用的筛选评测与摘要评审基础；本次抽跑的 272 个工具测试通过；T01、T02 在模拟依赖下可复现；现有真实筛选样本偏小且方向不均衡，摘要大样本基线与校准仍在进行。

尚不能确认：全仓测试健康、实际代码覆盖率、某模型在未来论文上的准确率、judge 的总体漏报率、真实 embedding 的排序质量、所有支持平台的完整用户旅程，以及正在进行的摘要基线最终结果。新增更多同类 mock 断言无法单独回答这些问题，需要对应的独立样本、标准答案与运行证据。

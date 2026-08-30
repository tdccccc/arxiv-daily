# P6 — release-asset-list-single-source

goal_ref: ../goal.md
updated: 2026-08-30

## Outcome

验收部署进测试 vault 的文件清单，与发布真正上传的文件清单，出自同一个常量；仓库里其余每一份清单副本都被一条测试按名字比对。往发布加一个资产而没有同步验收（或反过来），必须在 `npm run test:release-tools` 上变红，且失败信息指名道姓说出是哪两处、各自列了什么、多了或少了哪个文件。

## 原来的失败模式

`styles.css` 一度不在验收的部署清单里。后果不是验收报错，而是验收**照常全绿**：设置页的全部几何测量读的是测试 vault 里遗留的旧样式表，而不是被测分支构建出来的那一份。识破它靠的是一次巧合——改完 CSS 重跑，数字一像素未变。

当时的修法是把 `styles.css` 加进那个数组。那只堵了这一个洞。清单在仓库里有三处互不相干的硬编码：

- `scripts/desktop-acceptance/build-deploy.mjs` 的 `ARTIFACTS`
- `docs/release.md` 里「The Obsidian release assets remain exactly:」下面的三条 bullet
- `.github/workflows/release.yml` 的 provenance `subject-path`，以及 `gh release create` 的位置参数

三处当时碰巧一致，但没有任何东西强制它们保持一致。第四个资产出现的那天，同样的失败模式换个文件名重演一次：验收静默地量旧文件。

**红之前的证据**（加检查前，在本分支上实测）：

- 只往 `docs/release.md` 加一条 `- \`plugin/extra.js\``：`npm run test:release-tools` → `# tests 235 / # pass 235 / # fail 0`。全绿。
- 只往 `.github/workflows/release.yml` 的 `subject-path` 与 `gh release create` 加 `plugin/extra.js`：同样 `235 / 235 / 0`。全绿。
- 只往 `ARTIFACTS` 加 `extra.js`：5 条既有测试变红，但红在 `deployedArtifactPaths` 的 deep-equal 差异上，说的是「路径数组和写死的三条不一样」，与发布清单毫无关系——那份写死的期望值本身就是清单的第四份副本。

也就是说，危险的那个方向（发布多了资产、验收没跟上，正是 `styles.css` 那次的方向）在两条路径上都是静默通过的。

## Approach

单一来源 + 对其余副本的解析比对，而不是把 `docs/release.md` 变成生成物。发布说明是给人读的，机器只读它其中一段 bullet 列表。

- `scripts/release-assets.mjs`：只有 `RELEASE_ASSETS` 一个冻结常量，**零 import**——验收 harness 要加载它，不能因此拖进 `yaml` 之类的依赖。
- `scripts/desktop-acceptance/build-deploy.mjs` 的 `ARTIFACTS` 改为引用它。验收部署因此不再有自己的一份清单：发布加资产的那一次编辑，同时就是验收加资产的那一次编辑。
- `scripts/release-asset-sources.mjs`：把 `docs/release.md` 与 `.github/workflows/release.yml` 里的清单解析出来，与 `RELEASE_ASSETS` 比对。工作流走真正的 YAML 解析（`yaml`，已是 root devDependency），按 `actions/attest-build-provenance@` 这个稳定锚点定位 attestation 步骤、按 `gh release create` 定位上传步骤，取的是命令的**位置参数**（遇到第一个 `-` 开头的 flag 停），因此把 `README.md` 混进上传列表也会红，而不只是漏掉 `plugin/` 前缀的情形。
- 比对按集合而非顺序：`docs/release.md` 写的是 manifest → main → styles，`ARTIFACTS` 写的是 main → manifest → styles，顺序从来不是承诺。

## 解析失败绝不能表现为通过

这是本次最容易做错的地方。一个「找不到清单就返回空数组」的解析器，会在文档改版当天悄悄变成一条恒真断言——和它要防的失败模式一模一样。

因此每个解析器都是**抛异常**而不是返回空：

- marker 句子不在了 → 抛，并在信息里点名 `RELEASE_DOC_MARKER` 该跟着改。
- marker 出现两次 → 抛（哪一份是被检查的那份不能靠猜）。
- marker 下面的 bullet 变成了散文或表格 → 解析到 0 条 → 抛「empty list is a parse failure, not an empty release」。
- bullet 在列表内但不是 `` - `plugin/<file>` `` 的形状 → 抛，并回读该行原文与行号。
- 同一个资产列两次 → 抛。
- 工作流不是合法 YAML / 没有 `jobs:` / 没有步骤 / attestation 步骤不存在或不唯一 / `subject-path` 为空 / `gh release create` 不存在或不上传任何资产 → 各自抛。

聚合函数 `verifyReleaseAssetSources()` 把这些异常收集成 issue，因此**返回空数组只有一个含义**：每一处都被成功读出了一份非空清单，且它们互相一致。`canonical` 本身为空或不是数组也直接记为缺陷——空清单不是「大家都没有资产所以一致」，是 bug。

## Tasks

- [x] `scripts/release-assets.mjs`：`RELEASE_ASSETS` 冻结常量，零 import。
- [x] `build-deploy.mjs` 的 `ARTIFACTS` 改为引用该常量；部署行为逐字节不变（值相同、只读用法 `map` / `indexOf`）。
- [x] `scripts/release-asset-sources.mjs`：文档 bullet 解析、工作流两份清单解析、集合比对与失败信息、`verifyReleaseAssetSources()` 聚合。
- [x] `scripts/tests/release-assets.test.mjs`：32 例，覆盖四个方向的单边改动与全部解析失败分支。
- [x] `docs/release.md`：在清单下方加一段说明这份清单被机器读、与哪几处绑定——保持文档是人写的散文，不做成生成物。

## Verification

- 单边改动的红（加检查后，同样的临时改动）：
  - 只改文档 → `release asset lists disagree: RELEASE_ASSETS in scripts/release-assets.mjs has [main.js, manifest.json, styles.css] but docs/release.md (the release checklist read by humans) has [manifest.json, main.js, styles.css, extra.js] — docs/release.md (the release checklist read by humans) additionally lists extra.js`
  - 只改工作流 → 两条，分别指向 `.github/workflows/release.yml attestation subject-path` 与 ``.github/workflows/release.yml `gh release create` arguments``，各自 `additionally lists extra.js`。
  - 只改 `RELEASE_ASSETS` → 三条，文档与工作流两处各报 `is missing extra.js`。验收部署那一处**不报**，因为它已经跟着常量走了——这正是单一来源的意思。
- 全部改齐则通过：把 `theme.css` 同时加进常量、文档与工作流两处后，一致性断言转绿，且 `deployedArtifactPaths()` 自动变成四条路径，验收侧一行未改。
- 解析失败与空清单：`read` 注入抛错 / 注入空字符串 / 把文档 bullet 换成散文 / 换成表格 / 常量置空或置非数组——每一种都断言 issue 非空，即结果是失败而不是通过。
- 门禁：`npm run test:release-tools` 267 通过 0 失败（原 235，新增 32）；`npm test` 全 workspace 通过；`npm run check:boundaries` 通过。
- 真实桌面验收跑了，未走单元层代跑：`OBSIDIAN_TEST_VAULT=/home/tiandc/Desktop/plugin_test npm run test:desktop` → `desktop acceptance PASSED`，两个 session 共 16 条断言全绿，几何数字与 P5 记录逐字一致（窄面板 176px / 每行 26.8 字符，三按钮档 176px / 24.3 字符）。这一点本身也是证据：清单换了来源之后，样式表照旧被部署进去，量到的仍是分支的那一份。

## 边界

- `scripts/tests/desktop-acceptance-build-deploy.test.mjs` 里那句 `deployedArtifactPaths names only the three release artifacts` 保留原样、写死三条路径。它现在是清单的一道额外钉子而不是重复来源：加第四个资产时它会红，提醒人确认部署面确实该跟着变。既有断言一条未放宽。
- 未纳入的相邻检查：`scripts/check-obsidian-submission.mjs` 与 `scripts/smoke-build.mjs` 里出现的 `plugin/main.js`、`plugin/manifest.json` 不是资产清单，是对这两个具体文件各自的专项校验（bundle 预算、禁用模式、manifest 字段）。把它们塞进同一份清单会把「每个资产都要做 bundle 预算检查」这个不成立的意思带进来。

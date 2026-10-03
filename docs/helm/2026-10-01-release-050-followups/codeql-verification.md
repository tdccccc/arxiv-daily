# 授权push、CI结果与CodeQL修复

## 已完成的远程动作

用户A授权的一次push已完成：origin/fix/review-followups与PR51均指向`163f2f63be2a70b54ea0b4d47f3660ec7ecb1604`。PR仍为Draft，说明未修改，未合并/tag/发布。

- [Root verification](https://github.com/tdccccc/arxiv-daily/actions/runs/36979970181)：同SHA，Root workspace与Node20.19.0/22.17.0均成功。
- [Native storage verification](https://github.com/tdccccc/arxiv-daily/actions/runs/36979970112)：同SHA，Linux/macOS/Windows各x64/arm64及release asset assembly全部成功。
- `gh pr checks 51 --required`确认10个required检查全部pass。
- CodeQL分析workflow成功，但[结果检查](https://github.com/tdccccc/arxiv-daily/runs/110752751156)失败：5条本次告警（4 high / 1 medium）。未将分析任务成功等同于安全结果通过。

原始结果：`/tmp/arxiv-050-codeql-check.json`、`/tmp/arxiv-050-codeql-annotations.json`。仓库另有历史open alerts，本次没有处理或关闭全部历史告警。

## 本地修复

| 提交 | 对应告警 | 验证 |
|---|---|---|
| 005906f | #14/#15 两处摘要提取正则 | 50项原基线，扩展后72项契约回归；保持daily首内容行与source多行旧语义 |
| 2f01ac0 | #35/#36 作者拆分/路径规范化 | 56项原基线，扩展后66项；schema5与锁不变 |
| 534fb2f | #17 设置路径原型污染 | 18项基线，新增后7项有效Red；最终348项相关测试通过 |

性能前后比较来自真实公共入口，有限输入、不联网、不读用户数据：source摘要16k空白约61.5→0.68ms；作者8192空白约16.817→0.030ms。daily摘要该输入未复现超线性，改后约0.88ms（旧0.056ms），不声称提速；路径此前已先合并slash，约0.02ms前后相当。后两项为明确线性、防御性等价改写，不制造不存在的性能Red。

原型路径测试实际复现Object.prototype哨兵污染，finally清理；修复事先拒绝所有位置的保留键，只遍历自有属性，不触发继承setter，普通设置及安全新叶子兼容。没有改变CodeQL配置或抑制告警。

## 修复后的本地检查

- `NODE_OPTIONS=--max-old-space-size=8192 npm test`：3519通过，2项core既有跳过；core2343/node-runtime67/CLI125/plugin984。
- lint：0 errors / 20 warnings；typecheck、build、check:boundaries、check:obsidian-submission、check:release-version 0.5.0全部通过。
- release-tools342通过；build smoke、packed CLI/native Linux x64 install smoke通过。
- 日志：`/tmp/arxiv-codeql-final-{tests,lint,typecheck,build,smoke}.log`、`/tmp/arxiv-codeql-{release-tools,install-smoke}.log`。
- 详细报告：`/tmp/arxiv-codeql-abstract-report.md`、`/tmp/arxiv-codeql-paper-index-report.md`、`/tmp/arxiv-codeql-settings-report.md`。
- 本机没有CodeQL CLI，修复提交未重新运行远程扫描；不能宣称5条告警已被CodeQL关闭。需要第二次push授权后确认。

## 手测目录同步

原目标`/home/tiandc/Desktop/plugin_test/.obsidian/plugins/arxiv-daily/`先检查再备份，创建`main.js.bak-20261002-pre-codeql-fixes`与`styles.css.bak-20261002-pre-codeql-fixes`，未覆盖旧备份。三件资产校验一致：main.js `6006fd34f4ff91e0b5b033ea971ca6071b0f239d4beb13674b29d60021184d2b`、styles.css `c1b84403be62a5ad5de4e5ed42e387176b0a8607fac395146749f41fefbe696d`、manifest.json `6f9b6e674c8f838acb81edb5fd1b5e0ae766c7105abedd3817fe292938d76a8f`。记录`/tmp/arxiv-050-codeql-deploy.json`，未碰data.json/启动Obsidian/修改其它worktree。

下一步只申请再次push当前分支并等待新检查；PR说明/ready、合并、tag和发布仍分别授权。

## 另一批预先存在的告警（17条，分支fix/release-050-readiness，2026-10-03）

与上面5条「本次新增」告警不同，`gh api 'repos/tdccccc/arxiv-daily/code-scanning/alerts?state=open&per_page=100'`在main@bdfec4e上另外列出17条自2026-08-23起即存在、与PR51/52无关的open high-severity告警：js/polynomial-redos×11（`packages/core/src/{dashboard,delivery,metrics,pipeline}`下的标题/YAML值/路径/数学定界符解析）、js/incomplete-multi-character-sanitization×3与js/bad-tag-filter×2（`history-sync.ts`的HTML注释剥离、`arxiv-parser.ts`的script/style标签剥离）、js/incomplete-url-substring-sanitization×1（`arxiv-fetcher.test.ts`测试内的mock URL判断）。

逐条读取被标记代码与输入来源后：14条在代码中修复（含回归测试，含对抗性长输入计时断言），3条判定为不适用/误报并写入具体理由，未在本地关闭或抑制任何GitHub告警（关闭仍需用户审阅后另行操作）：

- 11条ReDoS：统一改为位置扫描（不依赖可能跨行回溯的`\s+`/`\s*`组合），或删除重复加捕获组前多余的`[ \t]*`（捕获值随后仍会trim），或改用原生`String.prototype.trimEnd()`替代`/\s+$/`，或把惰性`[\s\S]*?`两定界符匹配改写为`indexOf`扫描，手法与main已有的005906f/2f01ac0一致。
- `history-sync.ts`的`<!--.*?-->`改为`<!--[\s\S]*?-->`以匹配跨行注释；但该函数唯一调用方`parseDailyCandidates`按行解析，传入的`heading`字符串在生产路径上不可能包含换行，因此这是防御性修复，无法通过公开API构造出"跨行"场景来验证行为差异。
- `arxiv-parser.ts`的`stripUnsafeTags`闭合标签正则加`\s*`（如`</script\s*>`），修复`js/bad-tag-filter`；同一函数另外两条`incomplete-multi-character-sanitization`（未闭合标签场景）判定为误报：该函数是DOMParser/linkedom解析前的防御性剥离，解析出的document从不挂载到真实DOM，`parseRecent`只读取特定selector的`.textContent`，脚本不会执行、样式不会生效，与`email-render.ts`那种真正渲染为HTML邮件（有`escapeHtml`）的路径不同。
- `arxiv-fetcher.test.ts`的`req.url.includes("export.arxiv.org")`判定为误报：这是测试内mock HTTP client的分支判断，URL完全由被测生产代码构造（固定的arxiv.org端点），不是攻击者可控输入，不存在净化边界。

验证：`npx vitest run`（packages/core）2360通过/2既有跳过；plugin工作区987通过；`npm run typecheck`全部4个workspace通过。17条告警的逐条表格、具体修复提交摘要和3条误报的完整理由见本分支最终报告（父会话转交）；誊写为`/tmp/arxiv-050-codeql-dismissals.json`（{number, rule, path, reason, comment}数组）供用户审阅后决定是否在GitHub上关闭——本地未对这17条或历史其它open alerts执行任何dismiss操作。

### PR53上CodeQL复查（2026-10-04）

PR53的CodeQL结果检查失败：上述正则修法仍被判为4条告警——`history-sync.ts`注释剥离仍可能残留`<!--`（如`<!<!---->--`），`arxiv-parser.ts`对未闭合`<script`/`<style`仍不完整，且闭合标签可带属性样内容（`</script\t\n bar>`）。改为`64939bf`：listing先解析，再从DOM树删除script/style/link元素，交由解析器自身规则判定，不再正则剥离；标题注释改为线性扫描、每次切除后回退3字符重扫、未闭合注释截到末尾，结果不含`<!--`。新增parser测试（奇异闭合标签、title/authors内嵌script/style/link、未闭合script）与history-sync测试（嵌套/未闭合注释）。原拟dismiss的#19/#20因此改为代码修复，只剩#16（测试mock）待合并后附理由关闭。

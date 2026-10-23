# scripts/personalization 里的四个诊断脚本已经无法按注释运行：引用的启动文件和导出的函数都不存在

> 发现日期：2026-10-21 · 来源：main `d999037` 深度代码审查 · 编号 SCR-07
> 范围：`scripts/personalization/*.ts` · 严重程度：低 · 状态：已修复（分支 `fix/2026-10-21-batch4`，提交 d2fa7fd） · 置信度：已复现

## 问题

`scripts/personalization/` 里的 `analyze-dogfood.ts`、`diagnose-proposal.ts`、`raw-extraction.ts`、`validate-real-library.ts` 是 2026-08-04 为个人文献库一期写的一次性诊断脚本。现在：

- 每个脚本的头注释都写着用 `node scripts/personalization/run-<名字>.mjs` 运行，但目录里只有 `.ts` 文件，仓库里仓库里没有任何 `run-*.mjs`，也没有 npm 脚本；只有 2026-08-03 的 helm 文档 `dogfood-run-kit.md` 把其中两个当作已完成的工具提到。按注释的说明根本启动不了。
- 它们不在任何 `tsconfig` 的 `include` 里，也不在 CI 里，所以早就漂移了。用一个临时 tsconfig（继承 `tsconfig.base.json`，只加入这几个文件）跑 `tsc --noEmit`，这四个文件一共报 18 个错误，包括：`diagnose-proposal.ts` 从 `@arxiv-daily/core` 导入的 `proposePersonalLibraryDirections` 已不存在、`raw-extraction.ts` 导入的 `renderPersonalLibraryExtractionUserMessage` 已不存在、`diagnose-proposal.ts` 读 `proposal.candidates`（该字段已不在提案类型上）、传给核心的 logger 不是 `Logger`、`analyze-dogfood.ts` 对 `{}` 取属性、`validate-real-library.ts` 里可能为 `undefined` 的数组访问。
- 其中两个脚本默认指向维护者本机的绝对路径并带着个人信息：`validate-real-library.ts:33` 默认读 `/home/tiandc/Nextcloud/work/Article`（真实的个人文献目录），`diagnose-proposal.ts:22`、`raw-extraction.ts:21` 默认读 `/home/tiandc/Documents/code/arxiv-daily/plugin_test`；后两个会从 `plugin_test` 里的插件数据读出真实的 API 密钥并向“真实配置的接口”发请求（文件头注释明说）。这些脚本如果被修好并直接运行，默认值就会去碰个人数据和真实服务。

没有人受影响，因为脚本根本跑不起来；问题是它们看起来像仓库提供的工具，却已经是死代码，里面还留着个人路径和“用真实密钥调用真实接口”的默认行为。

## 原因

- `scripts/personalization/*.ts`：没有运行入口，没有类型检查，没有测试；个人文献库后续重构（提案类型、导出函数）没有同步更新它们。
- `scripts/personalization/validate-real-library.ts:33`、`diagnose-proposal.ts:22`、`raw-extraction.ts:21`：环境变量缺省时回退到维护者的绝对路径。

## 可能的解决方案

推荐：删除这四个脚本（它们服务的是 2026-08-04 那一次验证，结论已记录在 `docs/helm/2026-08-03-personal-library-guided-discovery/dogfood-run-kit.md`；`analyze-dogfood.ts` 还读已撤回的 `noveltyByReport`，见 `docs/helm/2026-10-15-library-novelty-and-change-nudge/phases/04-withdraw-novelty.md:35`）。如果其中某个仍然想保留（例如 `validate-real-library.ts` 对真实规模文献库的无界面验证），就补上运行入口、改成必须显式传路径（缺省时报错而不是回退到个人路径）、纳入类型检查，并在 `scripts/` 的某个 README 里登记。

备选：只在文件头注释里标明“已过时，不可运行”，并删去个人绝对路径。成本最低，但死代码仍留在仓库里。

## 修复后应补的测试

- 若删除：无需测试。
- 若保留：把保留的脚本加入一个 `tsc --noEmit` 的检查（例如在 `npm run typecheck` 里加一个 `scripts` 的 tsconfig），并加一个测试断言脚本在缺省环境变量时拒绝运行。

## 修复情况

移除四个只服务旧版文献库验证、已没有有效入口的脚本，避免读者继续尝试不存在的运行命令，也不再留下默认读取维护者目录和真实模型配置的工具。旧 dogfood 文档明确改为历史记录，保留当时结论并指向现行入门指南。

验证：仓库 scripts、package 命令和工作流均无对四个脚本的调用，旧已完成阶段中的历史引用保留；已检查删除 diff 和文档链接。按本问题建议，不新增测试、不运行退役脚本或真实服务。

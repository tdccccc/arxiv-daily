# arXiv 源码包解压没有任何上限，一个约 450 字节的 tar.gz 就能让进程内存耗尽崩溃

> 发现日期：2026-10-18 · 来源：main `21fe2ce` 深度代码审查 · 编号 CORE-01
> 范围：日报流水线全文抓取（插件、CLI 都受影响） · 严重程度：高 · 状态：已修复（分支 `fix/2026-10-18-review`，提交 1b1d759、6de78ee） · 置信度：已复现

## 问题

进入日报的论文都会以 `wantFullText=true` 抓取全文。没有 arXiv HTML 版本的论文，流水线会下载作者上传的 e-print 源码包并用 `extractLatexSource` 解析。源码包内容由任意 arXiv 作者控制。

- gzip 解压没有输出上限：400 KB 的 gzip 会被解成 400 MB，实测峰值内存 1.3 GB；更大的压缩炸弹直接耗尽内存。
- `\input` / `\include` 递归展开到 12 层，只防循环、不限总量。几个互相 `\input` 多次的小文件会指数级膨胀（扇出^12）。实测：447 字节、扇出 3 的 tar.gz 约 27 秒后让 node 以 `FATAL ERROR: Ineffective mark-compacts near heap limit` 崩溃（4 GB 堆）；扇出 2 时 224 ms 正常。

V8 堆内存耗尽无法被 `try/catch` 捕获：插件里整个 Obsidian 渲染进程被杀，CLI 进程直接退出。解析是同步的，即使不到崩溃程度也会冻住界面。源码包在解析前已写入缓存，崩溃后运行仍是 running，下一次调度读到缓存包再次崩溃，直到 10 次上限把这一天标为永久失败。还会留下过期的运行锁（见 CORE-09）。

## 原因

- `packages/core/src/pipeline/source-extractor.ts:88-97`：`ungzip(buffer)` 无大小限制。
- `source-extractor.ts:179-205`：`expandLatexInputs` 对每次出现都做文本展开，`seen` 只保护当前路径，没有总输出预算或单文件展开次数预算。
- `parseTar` 把每个 .tex/.bbl/.sty/.cls/.txt 条目都解码成字符串，无总量上限。
- `packages/core/src/pipeline/arxiv-fetcher.ts:582` `fetchBinary` 无响应体大小上限。

## 复现

审查脚本在 `tmp/review-2026-10-18/scratch-core/` 下构造了上述压缩炸弹和 `\input` 扇出包，直接调用 `extractLatexSource` 复现了 OOM。

## 可能的解决方案

推荐给整个解析过程设预算：

1. 用流式 inflate（pako `Inflate` 分块）替换 `ungzip`，累计输出超过上限（如 64–128 MB）即中止；源码下载同样设响应体上限。
2. `parseTar` 限制单条目字节数、总解码字节数和条目数。
3. `expandLatexInputs` 共享一个计数器（总展开字符数、总展开次数），超限后停止展开（如 8 MB / 2000 次）。
4. 超出预算按"源码不可用"处理，流水线退回摘要。

备选：放到 worker / 子进程里并限制内存，同时消除界面冻结，但对 Obsidian 宿主较重。

## 修复后应补的测试

- 压缩炸弹 gzip 被拒绝且不抛出未捕获错误。
- `\input` 扇出 3、深度 12 的包在预算内返回（截断或不可用），耗时有界。
- 正常多文件 LaTeX 包的展开结果不变。

## 修复情况

按方案实施：gzip 流式解压上限 64 MB，tar 条目数与文本字节有上限，`\input` 展开共享预算（8 MB / 2000 次），源码下载上限 50 MB（超限不缓存、不解析）。超出任何预算都按"源码不可用"处理，回退到摘要。

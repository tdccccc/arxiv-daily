# Core 测试堆耗尽诊断 — 2026-08-30

> 复现分支：`feat/library-setup-path`（`packages/` 相对 `origin/main` 无改动）
>
> 症状文件：`packages/core/tests/pipeline/pipeline-novelty-stage.test.ts`
>
> 结论：**不是产品代码缺陷**。泄漏在 Core 测试共用的 markup parser 替身里。

## 复现

```
npm run test --workspace @arxiv-daily/core -- tests/pipeline/pipeline-novelty-stage.test.ts
```

vitest worker 在跑到第 3~10 个用例时耗尽默认约 4 GB 堆，`FATAL ERROR: Reached heap limit`。worker 崩溃后 tinypool 仍向已关闭的 IPC channel 写入，于是终端最后看到的是 `ERR_IPC_CHANNEL_CLOSED` / `Channel closed` 的 unhandled rejection —— 那是崩溃的后果，不是原因。

`scripts/run-core-tests.mjs` 没有设置任何 `--max-old-space-size`，4 GB 是 Node 在本机的默认老生代上限。该脚本把测试文件按 8 个一批分进程跑，这本身就已经在掩盖单进程内存增长：只要一批里凑巧同时包含若干重解析大 fixture 的文件，随时可能复发。

## 增长曲线

用 `--logHeapUsage` 观察，堆占用随**用例数**线性上涨，每个用例约 +250~380 MB，与输入规模无关（该文件每个用例只有 2 篇论文、1 条代表作）：

| 用例 | 堆 |
| --- | --- |
| regenerates every planned paper… | 2459 MB |
| skips the checkpoint save… | 2843 MB |
| degrades a checkpoint lookup failure… | 3091 MB |
| degrades only the affected paper… | 3473 MB → OOM |

单独跑其中任何一个用例都能通过。所以不存在 N×N 比较或按论文数增长的结构，是**每次运行固定泄漏一大块**。

进一步把变量缩到解析器本身（`--expose-gc`，每次测量前强制两轮 GC，887 KB 的 `tests/fixtures/arxiv-recent-astroph.html`）：

| 做法 | 6 次解析后的堆 |
| --- | --- |
| 每次 `new happy-dom Window()` + 解析，随后丢弃全部引用 | 22 → 824 MB（**每次约 +130 MB，GC 后不释放**） |
| 复用同一个 happy-dom `Window` 反复解析 | 22 → 794 MB（同样泄漏） |
| 每次解析后 `await window.happyDOM.close()` | 平坦 |
| `linkedom` 的 `DOMParser` 反复解析 | 平坦（+1 MB） |

## 根因

`packages/core/tests/markup-parser.ts` 是 Core 测试用的 `MarkupParser` 替身，它每次 `parseFromString` 都 `new Window()`（happy-dom 20）。happy-dom 会持有它解析出的每一个 `Document`，直到 window 被**异步** `close()`；而 `MarkupParser.parseFromString` 是同步的、必须返回一个还能用的 document，所以调用点没有任何时机去关它。

每构造一次测试用的 pipeline，这个 fixture 会被解析 2~3 次（`firstBucketIds()` 一次、pipeline 内部 `fetchRecent` 之后一次），于是 21 个用例累计几十次解析、每次 130 MB 不还，必然打穿堆。

这条路径完全不进产品：

- Node/CLI 宿主用 `packages/node-runtime/src/markup-parser.ts` 的 `LinkedomMarkupParser`（linkedom，实测无保留）。
- Obsidian 宿主用 `plugin/src/hosts/obsidian/markup-parser.ts`，即浏览器原生 `DOMParser`。
- happy-dom 在整个仓库只出现在测试侧（本文件 + `plugin/vitest.config.mts` 的 `environment`）。

novelty 阶段自身没有无界结构：`personalized-novelty.ts` 的模块级状态只有两个 `WeakSet`，代表作按 direction 分组、按调用上界裁剪，没有 N×N 比较，也不保留全文语料。

## 修法

1. `packages/core/tests/markup-parser.ts` 改用 linkedom —— 与 Node 宿主生产适配器同一个库。既消除泄漏，也让 Core 测试跑在真实 Node 宿主实际使用的解析器上，而不是第三种只存在于测试里的实现。
2. 新增 `packages/core/tests/markup-parser-retention.test.ts`：用 `WeakRef` + 强制 GC 断言「调用方丢弃后，解析器不再持有任何 document」，另一条用例断言持有期间 document 仍可用。这条断言在旧 happy-dom 版本下为红、在 linkedom 下为绿，锁住的是不变式而不是内存字节数。
3. `package.json` 把 `linkedom` 显式声明进根 devDependencies（此前 Core 测试是靠 node-runtime 的依赖提升拿到的）。

没有调大堆、没有 skip、没有缩减 fixture 规模、没有吞掉那条 unhandled rejection：`pipeline-novelty-stage.test.ts` 的用例数与数据规模一字未动，21 个用例全跑，峰值从 >4 GB 降到 252 MB。

## 验证

| 命令 | 结果 |
| --- | --- |
| `npm run test --workspace @arxiv-daily/core` | 全部批次通过 |
| `npm run typecheck --workspace @arxiv-daily/core` | 通过 |
| `npm test`（根目录全量） | 通过 |

## 顺带观察

- `scripts/run-core-tests.mjs` 的 8 文件分批，最初大概就是为压住这类内存增长而存在的。泄漏修掉后它仍然有用（隔离性），但不该再被当作内存对策。
- 887 KB 的 `arxiv-recent-astroph.html` 每次构造 pipeline 都要重新解析。现在不泄漏了，但这仍是该文件里最贵的一步；若日后再变慢，可以在文件级缓存一次解析结果。

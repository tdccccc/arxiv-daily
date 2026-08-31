# P7 — false-green-guards

goal_ref: ../goal.md
updated: 2026-08-31

## Outcome

验收在「宿主根本没起来」时必须判为**阻塞**并且**不产出任何断言结果**，而不是输出满屏 PASS 加一整套截图。三个具体的洞各自堵上：环境余量在启动前探测、截图落盘前确认拍到的是目标、走查前后确认应用处于可走查状态。

## 触发本阶段的那次假绿（2026-08-31）

本机 inotify 监视数配额被用满（上限 65536 全部占用）。Obsidian 启动后显示它自己的错误页——正文是 `ENOSPC: System limit for number of file watchers reached, watch '/home/tiandc/Desktop/plugin_test/'`，下面是 `Reload app` / `Open another vault` 两个按钮，**整个设置页根本没渲染出来**。那一轮的产出是：

- **17 项断言全部 PASS**，且带着精确到小数的测量值（`description 176px wide, 26.8 characters over 4 lines`）。
- **10 张截图全部静静落盘**，内容是错误页文字或纯色空白，没有一张是设置页。
- 整轮只因为「渲染进程零错误」这一条判红，退出码是 **1（失败）**——读起来像产品坏了——而不是 **2（阻塞）**。

把配额提到 524288 后在同一台机器重跑，17/17 真绿、截图全部正常：产品侧一直是好的，坏的是验收自己看不出环境已经塌了。

这与 P6 之前那次「部署漏拷 `styles.css`，几何测量一直在量测试库里的旧样式」是同一类：看起来在验证，实际没有在验证它以为在验证的东西。区别在于这次更危险——它不是漏了一项证据，而是伪造了一整套。

## Assumptions

- `/proc/sys/fs/inotify/max_user_watches` 只给上限，**不给当前用量**；单纯比上限无法区分「空闲的 524288」与「用满的 524288」。因此判据必须是功能探测。
- libuv 每个事件循环共用**一个** inotify 实例，`fs.watch` 每次调用消耗的是一个 **watch descriptor** 而不是一个 instance。已实测确认：同时持有 200 个 watcher 时 `/proc/self/fd` 下只有 1 个 inotify fd。所以批量探测打的正是耗尽的那种资源，而不会撞上 `max_user_instances`。
- inotify 对**已监视的同一路径**返回同一个 wd，不额外占额度。因此探测必须一个 watch 对一个新目录。
- Obsidian 的错误页不是唯一的环境崩溃形态；ENOSPC 只是其中一种。判据必须是「有没有挂载出可走查的 vault 窗口」这一正向能力，而不是任何一句错误文案。
- 真实 ENOSPC 需要 root 调低配额才能复现，本轮**不复现**（无 sudo，且不得修改系统配置）。三条守卫一律用注入式确定性测试取红。

## Approach

**一、`preflight.mjs` 功能探测监视器余量。** `probeFileWatchCapacity()` 在临时目录下建 `WATCH_HEADROOM = 128` 个子目录，逐个 `fs.watch`，随后**全部释放**（探测若留着监视，就会把它要测的余量本身吃掉）。`ENOSPC` → 阻塞，信息里带「拿到几个 / 要几个」和上限值，修复动作给 `sudo sysctl -w` 与 `/etc/sysctl.d/` 两步。128 是**下限不是预测**：Obsidian 实际要按 vault 目录数逐个建监视，需求更大；一个此刻连 128 个都给不出的环境，根本打不开 vault。非 `ENOSPC` 的失败（临时目录不可写、文件系统没有 inotify）返回 `measured: false` 而**不**报阻塞——宁可说「没测出来」，也不编一个站不住的阻塞。

**二、`screenshots.mjs` 落盘前置校验。** 顺序改为「先判定，后写盘」，因此拒绝不留下任何文件。按 selector 定位的目标要求：在文档里、可见（`display` / `visibility` / `opacity` / 真实 client rects 四条）、尺寸非零、且至少 `MIN_VISIBLE_FRACTION = 0.5` 落在被拍摄的视口内（`captureBeyondViewport` 是关的，视口外拍回来是空画布）。调用方自己量好矩形传进来的那条路径同样过后两条。拿回的 PNG 再做一次内容判定：**整幅只有一种颜色即拒绝**。

纯色判据**做了**，判据是「整幅一色」这一条，不是任何比例阈值、不是与基线比对。理由：这些截图每一张都必然含文字、边框或控件，整幅一色不可能是它声称的那个状态，无论是什么颜色；而合法的纯色**区域**完全不受影响——规则只在整帧再无他物时才触发。为此内置了一个 PNG 解码器（8 位、非隔行，颜色类型 0/2/4/6，五种 scanline filter 全支持），零新增依赖，只用 `node:zlib`。解不了的格式报 `measured: false`，绝不误判为失败。chunk CRC 不校验：字节是本进程自己启动的渲染进程经 loopback 送来的，要问的是图里有什么，不是传输有没有坏。

**三、`app-state.mjs` 应用错误态检测。** 新模块。`APP_USABILITY_EXPRESSION` 把渲染进程的能力读成**数据**：`app` 对象、`app.workspace`、`app.setting`、workspace 容器元素、`.workspace-leaf` 数量，外加页面正文文本与按钮文案。`judgeAppUsability()` 是纯函数，只按前五条**正向能力**判定；文本与按钮只用于**转述**给操作者看，从不参与判定——所以任何没预料到的环境崩溃都会一起被抓住，而不是只抓 ENOSPC 那一种。

`withAppUsable({ evaluate, run })` 把整轮走查夹在两次检查之间，接进 `session.mjs`。**第二次检查是关键**：中途塌掉的应用照样会返回一整套结果，而那套结果一文不值——每一条都是对着取代了 vault 窗口的东西断言的。这里抛异常把它整个丢弃，因此坏掉的宿主结构上不可能产出哪怕一条 PASS。异常带 `blockers`（与 preflight 同形），`acceptance.mjs` 据此走 `EXIT_BLOCKED`，并明说本轮不报告任何断言结果。

## Test strategy

- change kind: behavior change（验收自身的可信度加固）
- strategy: strict Red–Green–Refactor，三条守卫全部注入式确定性取红
- Red / baseline signal: 三个测试文件在实现前分别因缺 export / 缺 export / 缺模块而失败（原文见 Verification）
- Green / regression checks: `test:release-tools`、`npm test`、`check:boundaries` 全绿；真实桌面验收 17/17、退出码 0、十张截图照旧
- exception: 真实 ENOSPC 未复现——需要 root 调低配额，本 initiative 不修改系统配置。这是本阶段明确的验证边界。

## Tasks

- [x] `probeFileWatchCapacity()` 功能探测 + `preflight` 阻塞项与修复动作；上限值只作为上下文出现在信息里，不参与判定。
- [x] 截图前置校验（存在 / 可见 / 尺寸 / 在视口内）与「先判定后写盘」的顺序，拒绝时零文件落盘。
- [x] PNG 解码器与整幅纯色判据；实测确认能读渲染进程真实产出的 10 张图（8 位 RGB），而不是静默弃权。
- [x] `app-state.mjs` + `withAppUsable`，接入 `session.mjs` 与 `acceptance.mjs` 的退出码契约。
- [x] `README.md` 补「What is checked before anything is believed」一节与退出码 2 的新含义。

## Verification

### 三条守卫的红（注入）

- 监视器：`node --test scripts/tests/desktop-acceptance-preflight.test.mjs` →
  `SyntaxError: The requested module '../desktop-acceptance/preflight.mjs' does not provide an export named 'WATCH_HEADROOM'`
- 截图：`node --test scripts/tests/desktop-acceptance-screenshots.test.mjs` →
  `SyntaxError: The requested module '../desktop-acceptance/screenshots.mjs' does not provide an export named 'captureTargetExpression'`
- 错误态：`node --test scripts/tests/desktop-acceptance-app-state.test.mjs` →
  `Error [ERR_MODULE_NOT_FOUND]: Cannot find module '.../scripts/desktop-acceptance/app-state.mjs'`

转绿后：preflight 18 例、screenshots 19 例、app-state 10 例全通过。

### 真实运行中的反向证据（错误态守卫）

单元测试证明判据正确，但证明不了它接在了真实路径上。因此在 `REQUIRED` 里临时插入一条**永远不存在**的能力，跑真实桌面验收，得到：

```
desktop acceptance stopped: the application was not in a state where a walk means anything

  - Obsidian was not in a usable state before the walk: the renderer is not showing a
    usable vault window — it has no a capability that never exists (temporary reverse
    evidence). The page reads: "plugin_test arxiv-daily small_library …". It offers
    All91 / Starred1 / Detail summary1 / Refresh / Run today / Summarize by ID / More / Settings.
    → fix what the page above reports and rerun — …

No assertion result is reported from this run: every check that had already been made was
made against a renderer that was not showing the vault.
EXIT=2
```

**退出码 2，全程一条 PASS 都没有**，且页面实际内容被如实转述出来。临时改动已撤回。

### 纯色判据确实活着

拿真实运行刚产出的十张 PNG 逐一喂给 `judgeCapturedImage`：十张全部 `measured`（8 位 RGB，3 通道），没有一张走进「格式读不了 → 弃权」的分支。这条很重要——若 Chrome 的实际输出解不了，纯色判据会对每一张真图静默弃权，等于不存在。

### 门禁

- `npm run test:release-tools` — 301 tests / 301 pass / 0 fail，`check:product-units` OK
- `npm test` — 42 files / 677 tests / 0 fail
- `npm run check:boundaries` — OK
- `OBSIDIAN_TEST_VAULT=/home/tiandc/Desktop/plugin_test npm run test:desktop` — **17/17 PASS，退出码 0**，十张截图照旧。几何数字与 P5 记录逐字一致（`narrow panel 448px (window 700): 2 lines … description 176px wide, 26.8 characters over 4 lines`），说明三条守卫在环境正常时没有误伤，也没有改动任何既有断言。

### 未复现的部分（必须如实读）

**真实 ENOSPC 本轮没有复现。** 复现它需要 sudo 调低 `fs.inotify.max_user_watches` 并让配额真的耗尽，而本 initiative 不修改系统配置。因此监视器守卫的证据是：探测函数在注入「建监视失败」时判为阻塞（单测），以及探测函数在本机真实建立并释放 16 个监视（单测，非注入）。**「配额真的耗尽时 Obsidian 会被拦在启动前」这一条是推断，不是观测。** 同样地，错误态守卫见过的唯一一次真实错误页是 2026-08-31 那次事故本身，本轮的反向证据是靠临时篡改判据取得的。

## Abort / reshape triggers

- 如果为了让守卫转绿需要放宽既有 17 项断言中的任何一条或其容差，停止并汇报。未触发：既有断言与三个几何常量一字未改。
- 如果纯色判据在真实截图上产生误伤，退回到只做元素层面的校验并说明理由。未触发：十张真图全部判为多色。
- 如果错误态判据需要匹配具体错误文案才能生效，停止——那说明判据选错了层次。未触发：判据是正向能力，文案只被转述。
- 如果监视器探测本身占用的额度会影响被测环境，停止。未触发：探测释放它拿到的每一个监视，且只用 1 个 inotify 实例。

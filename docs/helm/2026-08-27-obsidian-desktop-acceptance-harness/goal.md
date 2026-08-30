# Obsidian 桌面验收自动化 harness（Obsidian desktop acceptance harness）

status: active
updated: 2026-08-30
owner: claude-code-session

## Intent

用 CDP 驱动真实 Obsidian 渲染进程，建立可重复、无人值守的桌面宿主验收 harness，让「实际桌面 UI 交互」从只能人工执行的验收项变成可复现的自动化证据。首个消费者是 `2026-08-17-pdf-hybrid-library-foundation` 的 P7 剩余验收项。

## Success criteria

- [x] harness 以隔离 XDG 配置启动真实 Obsidian，vault 列表只含指定测试 vault；用户真实 vault 与 `~/.config/obsidian` 全程不读不写。
- [x] 进程生命周期按 `setsid` 进程组建立与回收；用户常驻的真实 Obsidian 会话在 harness 运行前后均存活且未被影响。
- [x] 测试 vault 的可变状态（plugin settings store、workspace）在每轮运行前备份、运行后还原，失败路径同样还原。
- [x] harness 将当前分支构建部署进测试 vault，断言运行的是被测版本而非 vault 中的历史构建。
- [x] CDP 会话可断言插件完成加载，并收集全程 console error 与 pageerror；零错误是可检查的通过条件。
- [x] P7 四项桌面验收在真实宿主可断言：PDF `#page=N` 打开动作与页码降级、sidecar 默认关闭、启用后探测失败回退 PDF.js、旧 settings migration。
- [x] 单条命令可复现执行；不进入默认 `npm test`，不进入 plugin bundle，`check:boundaries`、`check:product-units`、lint 与 bundle 预算门禁维持通过。
- [x] 环境不具备（无显示、Obsidian 缺失、CDP 端口被占）时给出明确的阻塞原因并非零退出；脚本本身即等价的可复现手工步骤。
- [ ] 设置页这类「只能肉眼确认」的验收项可断言：分组顺序、行内按钮集合、真实布局几何（同一行 / 右对齐 / 未溢出容器）与就地授权对话框的出现与取消，并在关键状态落盘截图供人判断措辞与观感。

## Non-goals

- Windows 与 macOS 宿主验收；本 harness 只覆盖本机 Linux 桌面宿主。
- 外部 PDF 阅读器对 `#page=N` 的行为，以及 Obsidian 之外的任何宿主应用。
- 像素级视觉回归：不存基线图、不做截图比对，也不做主题渲染验收。**落盘截图本身是做的**（P5）——它是给人看的判断输入，不是断言的依据：没有任何断言读取图片，图片每轮覆盖，且位于 git 忽略目录。文案是否得体、时机是否突兀这类问题仍由人看图决定，harness 不代替。
- 替换或重写既有 Core / Plugin / Node / CLI 单元与集成测试。
- 修改 `2026-08-17-pdf-hybrid-library-foundation` 或任何并行 active Helm 的状态与验收结论。
- 把 harness 接入 CI 或任何远程执行环境。

## Constraints

- 测试 vault 只能是 `/home/tiandc/Desktop/plugin_test`；不得打开、枚举或写入用户的其它 vault。
- 严禁按进程名或命令行模式杀进程（`pkill -x obsidian`、`pkill -f <pattern>`）：前者会杀掉用户真实会话，后者会匹配到执行脚本自身。只允许 `setsid` 进程组 + `kill -PGID`。
- harness 必须与用户常驻的真实 Obsidian 会话共存，不独占 CDP 端口、显示或配置目录。
- 优先零新增运行时依赖（Node 22 内置 WebSocket）；不得引入会下载 Chromium 的 `puppeteer`。
- harness 代码不得被 plugin bundle 引用，不得违反 `check:boundaries` 的分层约束。
- 每个行为变更分块采用 Red–Green–Refactor，并在进入下一阶段前取得定向测试与相关回归证据。
- 本目标不修改并行 active Helm 的状态；与它们共享文件时保持改动隔离。

## Phases

<!-- Single source of truth for phase status. PN ↔ filename NN. Outcomes only — no steps. -->
1. P1 — 隔离启动、进程组回收与测试 vault 状态保护具备可复现证据 — status: done
2. P2 — CDP 会话层（连接、求值、诊断收集、信任对话框）稳定可复用 — status: done
3. P3 — P7 四项桌面验收场景在真实宿主产出断言级证据 — status: done
4. P4 — 集成为单条命令，门禁隔离与环境阻塞降级完成 — status: done
5. P5 — 个人文献库设置页场景与截图能力：分组顺序、按钮集合、两种面板宽度下的布局几何、就地授权对话框的出现与取消 — status: active

## Open questions

- 已解决：不预置信任状态。每轮出现的信任对话框正是诊断完整性的保证——插件在接受前不加载，harness 在接受前已启用诊断，因此整个插件启动期都在收集窗口内。`waitForPluginReady` 会在插件于 attach 前就已运行时报出 `loadedBeforeAttach`，把这条前提变成可检测项。

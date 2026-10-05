# arXiv Daily for DeepSeek Harness

左侧栏“设置”上方提供固定的“arxiv-daily”入口，空白新建页面也能打开。右侧栏入口页同时提供“arxiv-daily”标签，两处显示同一工作台。日历、论文列表、日报、详细总结和待读/收藏使用现有 arXiv Daily 核心及本地 Markdown。打开、搜索和标记不需要 Agent 对话，也不调用模型。

## 构建与安装

需要 Node.js **22.19+**（DSH 插件要求；独立 CLI/仓库要求 20.19+），以及原生模块构建所需的 CMake、C++ 编译器和 Node-API 头文件。从仓库根目录运行：

```sh
npm ci
npm ci --prefix extensions/dsh-arxiv-daily --ignore-scripts
node extensions/dsh-arxiv-daily/build.mjs
npm pack ./extensions/dsh-arxiv-daily/dist/package --pack-destination ./extensions/dsh-arxiv-daily/dist
```

当前产物是 `extensions/dsh-arxiv-daily/dist/dsh-arxiv-daily-0.1.17.tgz`。本地构建包含当前操作系统和架构的原生存储模块；当前验收环境为 Linux x64。其他平台需在对应系统构建并验证，不能将本机包当作跨平台包分发。

**DSH Desktop：** 打开“插件 / Plugins → 添加插件 / Add plugin”，输入生成的 `dsh-arxiv-daily-0.1.17.tgz` 的绝对路径，安装后选择“立即启用 / Enable now”。桌面 profile 由 Electron 管理，不要使用 `dsh plugin --profile desktop`。

**DSH Web：** 使用下面的命令安装到 Web profile：

```sh
dsh plugin --profile web add /absolute/path/dsh-arxiv-daily-0.1.17.tgz
```

重新启动 DSH，使新的插件包和客户端模块一起加载。包尚未发布到 npm；目前请安装本地产物。插件会注册自己的“arxiv-daily”侧栏标签。

## 从旧版本升级

用新包替换旧版本，配置、Markdown 和论文索引保持在原位置。Web profile：

```sh
dsh plugin --profile web add /absolute/path/dsh-arxiv-daily-0.1.17.tgz
```

然后退出并重新启动 DSH，使 Host 和客户端都加载新包。Desktop 使用插件管理器更新；若该版本的管理器不支持替换本地包，卸载旧插件后添加新包。插件卸载不会删除 arXiv Daily 的配置、Markdown 或论文索引。

## 使用

1. 点击左侧栏“设置”上方的“arxiv-daily”，无需先发消息；也可在右侧栏入口页选择“arxiv-daily”。
2. 首次使用会自动打开设置页。填写研究记录保存目录、模型 API 密钥与关注主题，点击“保存并开始使用”。已有 CLI 配置会直接沿用，不必再运行 `init`。
3. 工作台右上角“设置”遵循 Obsidian 1.13+ 主设置路径：LLM、arXiv categories、Research topics、Automatic detail notes、Timezone、Output & schedule、Personal library、Email delivery、Advanced、Help & feedback。分类、推理强度、时区、时间窗口等保留原选择控件；模型保留自由输入、候选列表及“Get models”。点击“保存设置”后生效，密钥留空保留，默认不回显；点击 Show 时按需显示已保存密钥，Hide 隐藏。独立运行必需的保存根目录放在原设置之前。

   Personal library 的 Choose folder / Build index / Revoke 复用共享文献库流程；Web 宿主通过路径框指定目录，首次处理会显示授权范围。Email delivery 的 Send verification / Send test 只有点击后才发送。Enable 开启后，仅在工作台进程运行时，按 Check every (minutes) 检查工作日报；原 CLI 外部 cron 配置独立保留，不会自动安装或修改系统任务。
4. 可使用 DSH 自带的侧栏放大或浮动功能。较窄时工作台通过“日历与筛选”切换导航。

模型使用 arXiv Daily 自己保存的 API 设置，与 DSH 对话模型分开。CLI 和 DSH 共用本机 CLI 配置：Linux/macOS 默认为 `~/.config/arxiv-daily/config.toml`（遵循 `XDG_CONFIG_HOME`），Windows 为 `%APPDATA%/arxiv-daily/config.toml`。保存目录决定 Markdown 和索引的位置，无需迁移到 DSH 会话目录。

工作台进程在同一插件实例的会话间共享。关闭阅读标签不会停止生成；禁用插件或退出 DSH 会停止它。插件重载后旧工作台链接会失效，再次点击“arxiv-daily”获取新链接。

旧版已在 DSH `0.1.7-alpha.1` 验证 Host 接口；本轮本机 DSH 已更新至 `0.2.0-rc.2`，在该版本验证固定入口的 SlotCore 注册和 Host 接口；未自动操作 Electron 桌面界面，其他版本仍需实测。

## 阅读与运行

- 左侧是可调宽度的日历和筛选，右侧默认显示论文列表。点击论文看概览，继续打开日报或论文总结；待读、已读与收藏独立保存。
- 正文、摘要、推荐理由和标题支持 Markdown 与 LaTeX 公式，包括 `$…$` 和 `$$…$$`。KaTeX 样式和字体随包提供，已有报告不必重新生成。外链图片和论文原文仍需要网络。
- Summary sources 及后续附录与正文分隔。页尾显示输入/输出/总 token、耗时和生成时间（UTC）；旧记录缺失值显示“未记录”，不以文件修改时间代替。概览引用的日报统计明确标注整份日报范围，不冒充单篇用量。
- 前进/后退在当前工作台阅读历史内跳转，恢复筛选、锚点和滚动位置；边界处按钮禁用。“返回列表”仍保留。这些按钮不控制 DSH 外部页面，也不跨工作台重启恢复历史。
- 生成日报或按 arXiv ID 生成论文总结时显示进度并可取消。等待公告、当日无更新、筛选无匹配与真正失败分别显示；等待公告不消耗失败重试额度。

1. 在“设置 → 个人文献库”连接目录并建立索引，按提示确认模型处理范围。
2. 点击左侧 **个人文献库** 浏览已识别的论文。目录筛选不调用模型；索引检索可选关键词、混合或语义模式，后两者使用配置的嵌入服务。已索引的非 arXiv PDF 也可显示。
3. 点击论文的 PDF 链接，在新页面打开本地文件（最多 25 MiB），使用浏览器或宿主的 PDF 阅读能力。
4. 点击 **方向审核 → 生成候选方向**，等待运行面板显示完成。打开审核页本身不调用模型。
5. 在“候选方向”中查看代表论文、编辑文本和依据、改名新主题、移动或删除候选；可预览代表论文的匹配情况。弱证据候选默认不选。
6. 选择方向并点击 **接受所选方向**，确认后保存到普通研究主题，之后参与日报筛选。已处理项不会重复添加；后续调整在研究主题设置完成。
7. “文献库概览”显示分析时间、已有方向覆盖、未归类与分析后新增论文。已修改方向的历史覆盖会提示重新验证。

生成与预览可在现有运行面板取消；预览不修改订阅。更完整的运行管理仍待补充。对应 [CLI 文献库命令](../../apps/cli/README.md#optional-personal-library) 继续可用。

## 实现边界

- `src/workbench-process.mjs` 只管理现有 CLI 的启动、就绪和停止，未来 Claude Mod 可复用这个边界；DSH 的 RPC/按钮代码留在 `host.mjs` 和 `client.mjs`。
- 两种入口都使用隔离 iframe，Web 仅向当前本机 DSH 的精确来源开放嵌入，Desktop 仅允许固定的 `dsh-app://app` 来源。工作台仍验证 capability、Host 和 API 请求 Origin。
- 目前面向本机 DSH；不支持远程 DSH 网页访问用户电脑的文献服务。
- 一次工作台生命周期只绑定一个网页来源；混用 localhost 和 127.0.0.1 时请回到原地址，或在任务结束后重新启用插件。
- 保留独立 CLI/浏览器入口，不复制论文库，不把研究数据存到会话历史中。
- 已有图形设置和首次使用引导，也保留 CLI 初始化；不自动切换 DSH 对话模型，尚未提供 Claude Mod 视图。

## 验证

```sh
npm ci --prefix extensions/dsh-arxiv-daily --ignore-scripts
node extensions/dsh-arxiv-daily/build.mjs
node --test extensions/dsh-arxiv-daily/tests/*.test.mjs
```

集成与验收记录见仓库的 `docs/helm/2026-10-02-dsh-research-plugin` 和 `docs/helm/2026-10-03-workbench-core-parity`。测试覆盖隔离的实际 DSH Host，不代表完成所有平台的 Electron 桌面视觉验收。

更新后必须退出并重新启动 `dsh web`，再刷新浏览器；仅替换磁盘上的插件包不会替换已经运行的工作台进程。

## 外观与界面语言

工作台右上角“设置 → 外观”集中管理主题和界面语言。修改后点击“保存设置”，界面立即重载所选语言和主题，保留当前阅读位置与筛选地址。界面语言不会改动论文总结语言、已有Markdown或用户研究主题。原始模型/网络诊断仍保留原文。

界面偏好保存在CLI配置旁的 `workbench-ui.json`，与业务TOML独立；侧栏布局和外观使用同一带锁的合并写入，不会互相覆盖。类型、默认值和纯校验，以及中英文词典统一由core提供，前端负责显示、宿主负责存储。Obsidian的原生设置控件和独立存储仍保留，未在此次变更中替换。

## 共享业务设置

模型、研究主题、输出、调度、文献库相关选项及邮件的字段描述由 core schema 统一提供，Obsidian 主设置页和工作台使用同一份顺序、名称、选项及条件。主题子字段和时区/运行窗口选项也共用定义。宿主仅保留原生控件、存储和运行时接线。

Obsidian 设置事务和工作台保存共用编辑校验：允许保存不完整草稿，真正获取模型、生成报告或发送邮件时再使用对应业务就绪校验。测试邮件及验证请求组装也共用 core 服务；既有模型列表已共用 LlmClient。Obsidian 原生存储、CLI TOML、工作台密钥留空保留、修订冲突和文献库授权仍由各宿主适配，不会互相复制或同步配置值。
